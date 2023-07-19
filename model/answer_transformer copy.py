''' 
Modified from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from data.config import CONF

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 0

# create attention mask
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# create masks for source and target
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)   

class AnswerTransformer(nn.Module):
    def __init__(self, answer_vocab):
        super().__init__()
        self.answer_vocab = answer_vocab
        self.testing = False
        self.seq2seq = Seq2SeqTransformer(3, 3, 256,  8, len(answer_vocab), 512)

    def forward(self, data_dict):
        if self.testing:
            return self.beam_search(data_dict, depth=1)

        # unpack
        answers_len = data_dict["answers_len"]
        batch_max_answer_len = torch.max(answers_len, dim=0)[0].item()
        src = data_dict["src"].permute(1, 0, 2) # num_proposals, batch_size, 256
        src_padding_mask = data_dict["src_mask"] # batch_size, num_proposals
        answers_to_vocab = data_dict['answers_to_vocab'] # batch_size, MAX_ANSWER_LEN
        answers_len = data_dict["answers_len"] # batch_size
        
        end_token = self.answer_vocab['<end>'] # 1

        # remove end token from target sequence
        answers_to_vocab_list = answers_to_vocab.tolist()
        for i, t in enumerate(answers_to_vocab_list):
            t = answers_to_vocab_list[i][:batch_max_answer_len]
            t.remove(end_token)
            answers_to_vocab_list[i] = t
            
        tgt_input = torch.tensor(answers_to_vocab_list, device=answers_to_vocab.device).permute(1, 0) # MAX_ANSWER_LEN - 1, batch_size

        # get masks
        src_mask, tgt_mask, _, tgt_padding_mask = create_mask(src, tgt_input)
        
        # run transformer model
        logits = self.seq2seq(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask) # MAX_ANSWER_LEN - 1, batch_size, vocab_size

        # combine output arrays into one
        logits = logits.permute(1, 0, 2)
        out_combined = torch.tensor([], device = logits.device)
        for i, length in enumerate(answers_len):
            sample = logits[i]
            out_combined = torch.cat([out_combined, sample[:length - 1]])
        data_dict['out_combined'] = out_combined

        return data_dict
            
    def beam_search(self, data_dict, depth=3):
        max_len = CONF.TRAIN.MAX_ANSWER_LEN
        start_token = self.answer_vocab['<start>'] # 1
        end_token = self.answer_vocab['<end>'] # 1
        src = data_dict["src"]
        softmax = nn.Softmax(dim=1)
        result = []
        
        # iterate over batch for every sample
        for i, src_item in enumerate(range(src.size()[0])):
            # reset B
            B = depth
            
            # encode source
            src_item = src[i].unsqueeze(0).permute(1, 0, 2) # MAX_TEXT_LEN, 1 (batch_size[i]), hidden_size
            src_seq_len = src_item.shape[0]
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
            memory = self.seq2seq.encode(src_item, src_mask) # MAX_TEXT_LEN, 1 (batch_size[i]), hidden_size
            memory = memory.to(DEVICE)
            
            # first pass
            ys = torch.ones(1, 1).fill_(start_token).type(torch.long).to(DEVICE) # 1, 1
            tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(DEVICE)
            out = self.seq2seq.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)

            # get probabilities for first word
            prob = softmax(self.seq2seq.generator(out[:, -1])) # 1 (batch_size[i]), vocab_size

            # get top B words
            values, indicies = torch.topk(prob, B, dim=1)
            
            # concat <start> and top B words to make B sentences
            values = values.squeeze(0).unsqueeze(-1).unsqueeze(-1)
            ys = ys.repeat(1, B) # 1, B
            initial = torch.cat([ys.unsqueeze(-1), indicies.unsqueeze(-1)], dim=-1).squeeze(0).unsqueeze(-1) # B, 2 (start & first word), 1

            # concat scores for sorting later
            initial = torch.cat([initial, values], dim=1)
            
            # store top B sentences for every sample in the batch
            result_sample = []
            n = 0
            # run beam search steps
            while (B > 0) and (n < max_len):
                n = n + 1

                # store current top sequences 
                new_initial = None

                # for every sentence in initial calculate top B next words
                for b in range(B):
                    seq_i = initial[b][:-1].long() 
                    p = initial[b][-1]
                    tgt_mask = generate_square_subsequent_mask(seq_i.size(0)).type(torch.bool).to(DEVICE)
                    out = self.seq2seq.decode(seq_i, memory, tgt_mask)
                    out = out.transpose(0, 1)

                    # probabilties of next word for sequence b
                    prob = softmax(self.seq2seq.generator(out[:, -1])) # 1 (batch_size[i]), vocab_size
                    values, indicies = torch.topk(prob, B, dim=1)

                    # create B copies of the sequence b and attach top B words
                    seq_i = seq_i.repeat(B, 1, 1) # B, seq_len, 1
                    values = values * p
                    indicies = indicies.squeeze(0).unsqueeze(-1).unsqueeze(-1)
                    values = values.squeeze(0).unsqueeze(-1).unsqueeze(-1)
                    seq_i = torch.cat([seq_i, indicies, values], dim=1)
                    if new_initial == None:
                        new_initial = seq_i
                    else:
                        new_initial = torch.cat([new_initial, seq_i], dim=0)

                # sort all new B * B sentences
                new_initial = new_initial.squeeze(-1)
                new_initial_list = new_initial.tolist()
                new_initial_sorted = sorted(new_initial_list, key=lambda x: -1 * x[-1])

                # take top B sentences
                new_initial_sorted = new_initial_sorted[:B]

                # filter the sentences that are done (<end> token or reached max_len)
                finished_seq = [item for item in new_initial_sorted if (item[-2] == end_token) or (len(item) >= max_len)]
                initial = [item for item in new_initial_sorted if (item[-2] != end_token) and (len(item) < max_len)]
                
                # update B
                B = len(initial)
                initial = torch.tensor(initial, device=DEVICE).unsqueeze(-1)
                result_sample.extend(finished_seq)
            
            # sort the B sentences created for the sample
            result_sample = sorted(result_sample, key=lambda x: -1 * x[-1])
            result.append(result_sample)
        
        # format answer
        pred_answers_tokens = [[answer[:-1] for answer in item] for item in result]
        pred_answers_tokens = [[[int(token_id) for token_id in answer] for answer in item] for item in pred_answers_tokens]
        pred_answers_probs = [[answer[-1] for answer in item] for item in result],
        data_dict["pred_answers"] = [[" ".join(self.answer_vocab.lookup_tokens(answer[1:-1])) for answer in item] for item in pred_answers_tokens]
        data_dict["pred_answers_prob"] = pred_answers_probs

        return data_dict
        
    # function to generate output sequence using greedy algorithm
    def greedy_decode(self, data_dict):
        max_len = CONF.TRAIN.MAX_ANSWER_LEN
        start_token = self.answer_vocab['<start>'] # 1
        end_token = self.answer_vocab['<end>'] # 1
        src = data_dict["src"]
        data_dict['ys'] = []
        data_dict['pred_answers'] = []
        softmax = nn.Softmax(dim=1)
        
        # iterate over batch
        for i, src_item in enumerate(range(src.size()[0])):
            src_item = src[i].unsqueeze(0).permute(1, 0, 2)
            src_seq_len = src_item.shape[0]
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

            memory = self.seq2seq.encode(src_item, src_mask)
            ys = torch.ones(1, 1).fill_(start_token).type(torch.long).to(DEVICE)
            for i in range(max_len-1):
                memory = memory.to(DEVICE)
                tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                            .type(torch.bool)).to(DEVICE)
                out = self.seq2seq.decode(ys, memory, tgt_mask)
                out = out.transpose(0, 1)
                prob = softmax(self.seq2seq.generator(out[:, -1]))
                _, next_word = torch.topk(prob, 1, dim=1)
                next_word = next_word.item()

                ys = torch.cat([ys, torch.ones(1, 1).type_as(src_item.data).fill_(next_word)], dim=0)
                if next_word == end_token:
                    break
                    
            data_dict['ys'].append(ys.reshape(-1).long().flatten()) 
            data_dict['pred_answers'].append([" ".join(self.answer_vocab.lookup_tokens(ys.reshape(-1).long().tolist()[1:-1]))])

        return data_dict

    def set_testing(self, value):
        self.testing = value