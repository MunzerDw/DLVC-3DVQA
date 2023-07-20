import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.beam_search import BeamSearch, Module

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 40

# create attention mask
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncodingText(nn.Module):
    def __init__(self, emb_size, dropout=0.1, maxlen=5000):
        super(PositionalEncodingText, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )

class Decoder(Module):
    def __init__(
        self,
        cfg,
        d_model,
        answer_vocabulary,
        answer_embeddings,
        nhead=6,
        d_hid=300,
        nlayers=2,
        dropout=0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.answer_vocabulary = answer_vocabulary
        self.register_buffer("answer_embeddings", torch.FloatTensor(answer_embeddings))

        # decoder for both modalities
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers)

        # map output to distribution over answer vocabulary
        self.map_to_vocab = nn.Sequential(
            nn.Linear(300, len(answer_vocabulary))
        )

        # positional encoding for text sequence
        self.positional_encoding_text = PositionalEncodingText(emb_size=300)

        # for greedy decoding
        self.testing = False
        # we use this attribute to determine whether we want to generate
        # greedy sentences for logging the accuracies during training with teacher forcing
        self.validation = False
        # we use this attribute for generating a baseline sample for CIDEr loss
        self.baseline = False

    def forward(
        self,
        batch,
        memory,
        answer_embeddings,
        memory_mask,
        answer_embeddings_mask,
        answer_embeddings_attn_mask,
    ):
        if self.testing:
            seq, seqLogprobs = self._sample(
                memory, memory_mask, greedy=True
            )
            return seq
        elif self.baseline:
            bs = BeamSearch(self, max_len=MAX_LEN, eos_idx=self.answer_vocabulary["<end>"], beam_size=3)
            seq, _, seqLogprobs = bs.apply(memory, out_size=3, return_probs=True, memory_mask=memory_mask)
            batch["seq_sample"] = seq
            batch["seq_logprobs_sample"] = seqLogprobs
            return seq
        elif False:
            seq, seqLogprobs = self._sample(
                memory, memory_mask, greedy=True
            )
            batch["seq_greedy"] = seq
            batch["seq_logprobs_greedy"] = seqLogprobs
            return seq
        else:
            if self.validation:
                seq, seqLogprobs = self._sample(
                    memory, memory_mask, greedy=True
                )
                seq = seq.tolist()
                pred_answers = []
                idxtotoken = self.answer_vocabulary.get_itos()
                for answer in seq:
                    seq_to_tokens = [idxtotoken[idx] for idx in answer if idx != 0]
                    seq_to_tokens_str = " ".join(seq_to_tokens)
                    pred_answers.append(seq_to_tokens_str)
                batch["pred_answers"] = pred_answers
                batch["seq_logprobs_greedy"] = seqLogprobs

            vqa_output = self._tf(
                memory,
                answer_embeddings,
                memory_mask,
                answer_embeddings_mask,
                answer_embeddings_attn_mask,
            )
            batch['out_combined'] = vqa_output.flatten(end_dim=1)
            return batch

    def _sample(self, memory, memory_mask, greedy=False):
        batch_size = memory.size(0)
        start_idx = self.answer_vocabulary["<start>"]
        end_idx = self.answer_vocabulary["<end>"]

        # we do + 1 because of the start token <start>
        seq = memory.new_zeros(batch_size, MAX_LEN + 1, dtype=torch.long)
        seqLogprobs = memory.new_zeros(
            batch_size, MAX_LEN + 1, len(self.answer_vocabulary)
        )

        seq[:, 0] = start_idx

        for t in range(MAX_LEN):
            tgt_mask = generate_square_subsequent_mask(t + 1).to(memory.device)
            seq_embeddings = (
                self.answer_embeddings[seq[:, : t + 1].reshape(-1), :].to(memory.device)
            ).reshape(batch_size, t + 1, 300)
            answer_embeddings_mask = (seq[:, : t + 1] == 0).bool()
            output = self.decoder(
                self.positional_encoding_text(seq_embeddings).transpose(0, 1),
                memory.transpose(0, 1),
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=answer_embeddings_mask,
                memory_key_padding_mask=memory_mask,
            ).transpose(0, 1)

            # map output vectors to vocabulary length
            output = self.map_to_vocab(output)
            output = output[:, -1]
            logprobs = F.log_softmax(output, dim=1)

            if greedy:
                sampleLogprobs, it = torch.max(logprobs.detach(), 1)
            else:
                # fetch prev distribution: shape NxM
                prob_prev = torch.exp(logprobs.detach()).cpu()
                it = torch.multinomial(prob_prev, 1).to(logprobs.device)
                # gather the logprobs at sampled positions
                sampleLogprobs = logprobs.gather(1, it)

            # and flatten indices for downstream processing
            it = it.view(-1).long()

            # stop when all finished
            if t == 0:
                unfinished = it != end_idx
            else:
                it = it * unfinished.type_as(it)
                unfinished = unfinished & (it != end_idx)
            seq[:, t + 1] = it
            seqLogprobs[:, t + 1] = logprobs
            if unfinished.sum() == 0:
                break

        seq = seq[:, 1:]
        seqLogprobs = seqLogprobs[:, 1:, :]

        return seq, seqLogprobs

    # for memory mesh transformer beam search algorithm
    def step(self, t, answer_embeddings, memory, **kwargs):
        start_idx = self.answer_vocabulary["<start>"]
        memory_mask = kwargs["memory_mask"]

        if t == 0:
            answer_embeddings = (
                torch.Tensor(self.answer_embeddings[start_idx].cpu())
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(memory.shape[0], 1, 1)
            ).to(memory)
        else:
            answer_embeddings = self.answer_embeddings[answer_embeddings].to(memory)
            memory = memory.repeat_interleave(
                answer_embeddings.shape[0] // memory_mask.shape[0], 0
            )
            memory_mask = memory_mask.repeat_interleave(
                answer_embeddings.shape[0] // memory_mask.shape[0], 0
            )

        answer_embeddings_attn_mask = generate_square_subsequent_mask(
            answer_embeddings.shape[1]
        ).to(memory)

        # shared decoder
        output = self.decoder(
            self.positional_encoding_text(answer_embeddings).transpose(0, 1),
            memory.transpose(0, 1),
            tgt_mask=answer_embeddings_attn_mask,
            memory_key_padding_mask=memory_mask,
        )

        output = output.transpose(0, 1)

        # map output vectors to vocabulary length
        output = self.map_to_vocab(output)
        output = output[:, -1, :]

        output = F.log_softmax(output, dim=-1)

        return output

    def _tf(
        self,
        memory,
        answer_embeddings,
        memory_mask,
        answer_embeddings_mask,
        answer_embeddings_attn_mask,
    ):
        # shared decoder
        output = self.decoder(
            answer_embeddings.transpose(0, 1),
            memory.transpose(0, 1),
            tgt_mask=answer_embeddings_attn_mask,
            tgt_key_padding_mask=answer_embeddings_mask,
            memory_key_padding_mask=memory_mask,
        )

        output = output.transpose(0, 1)

        # map output vectors to vocabulary length
        output = self.map_to_vocab(output)

        return output
    
    def set_testing(self, value):
        self.validation = value
