import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence 

from data.config import CONF

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# inspired by: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class AttnDecoderRNN(nn.Module):
    def __init__(self, 
        hidden_size, 
        vocab_size, 
        bi_hidden_size, 
        answer_vocab, 
        num_proposal=256, 
        emb_size=300, 
        mcan_flat_out_size=512, 
        dropout_p=0.5, 
        max_length=CONF.TRAIN.MAX_TEXT_LEN
    ):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size,
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.mcan_flat_out_size = mcan_flat_out_size
        self.testing = False
        self.bi_hidden_size = bi_hidden_size
        self.num_proposal = num_proposal
        self.num_layers = 1
        self.answer_vocab = answer_vocab
        self.use_teacher_forcing = True
        self.use_beam_search=False
        self.beam_search_depth = 1

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.relu = nn.ReLU()
        self.attn = nn.Sequential(
            nn.Linear(int(mcan_flat_out_size + emb_size), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.max_length)
        )
        self.attn_op = nn.Sequential(
            nn.Linear(int(mcan_flat_out_size + emb_size), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_proposal)
        )
        self.attn_combine = nn.Sequential(
            nn.Linear(int(hidden_size + self.bi_hidden_size + emb_size), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.mcan_flat_out_size)
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(mcan_flat_out_size, mcan_flat_out_size, num_layers=self.num_layers, batch_first=True)
        self.out = nn.Linear(mcan_flat_out_size, vocab_size)

    def forward(self, input, hidden, encoder_outputs, object_proposals, object_mask, lang_mask, answer_to_vocab=None):
        # use beam search for testing if set to True
        if self.use_beam_search and self.testing:
            return self.beam_search(input, hidden, encoder_outputs, object_proposals, object_mask, lang_mask, depth=self.beam_search_depth)

        # last predicted token as input to the GRU
        output = input # 1, 1
        # hidden vector of the GRU
        hidden = hidden.repeat(self.num_layers, 1, 1) # num_layers, 1, mcan_flat_out_size
        if not self.testing:
            # current answer length including <end> token
            answer_len = len(answer_to_vocab)

        # predicted scores for each step (used to calculate the loss)
        outputs = []
        # predicted tokens
        pred_answer = []
        # attention weights on the question
        out_attn_weights = []
        # attention weights on the object proposals
        out_attn_ob_weights = []

        if self.testing:
            # loop as long as the longest answer in the training dataset
            loop_range = CONF.TRAIN.MAX_ANSWER_LEN
            # get end token index to know when to stop
            end_token = self.answer_vocab['<end>'] # 1
        else:
            # loop over the lenth of the ground truth answer to predict word by word
            loop_range = answer_len - 1

        for i in range(loop_range):
            # get embedding of word from token
            output = self.embedding(output).view(1, 1, -1)[0] # 1, emb_size
            output = self.relu(output)
            output = self.dropout(output)
            
            # --------- ATTENTION ---------
            # calculate attention weights of object proposals
            attn_weights_op = self.attn_op(torch.cat((output, hidden[0]), 1)) # 1, num_proposal
            attn_weights_op = attn_weights_op.masked_fill(object_mask.bool(), -1e9)
            attn_weights_op = F.softmax(
                attn_weights_op, 
                dim=1
            ) # 1, num_proposal
            # save the attention weights
            out_attn_ob_weights.append(attn_weights_op)
            
            # apply attention weights to object proposals and sum them up
            attn_applied_op = torch.bmm(
                attn_weights_op.unsqueeze(0),
                object_proposals.unsqueeze(0)
            )  # 1, 1, hidden_size
            
            # calculate attention weights of question embeddings
            attn_weights = self.attn(torch.cat((output, hidden[0]), 1)) # 1, max_length
            attn_weights = attn_weights.masked_fill(lang_mask.bool(), -1e9)
            attn_weights = F.softmax(
                attn_weights, 
                dim=1
            ) # 1, max_length
            # save the attention weights
            out_attn_weights.append(attn_weights[lang_mask.unsqueeze(0) < 1])

            # apply attention weights to question embeddingts and sum them up
            attn_applied = torch.bmm(
                attn_weights.unsqueeze(0),
                encoder_outputs.unsqueeze(0)
            ) # 1, 1, bi_hidden_size

            # concatenate current word embedding and the attention outputs
            output = torch.cat((output, attn_applied[0], attn_applied_op[0]), 1) # hidden_size + bi_hidden_size + emb_size

            # reduce to size of hidden vectors (combine)
            output = self.attn_combine(output).unsqueeze(0) # 1, 1, hidden_size
            output = F.relu(output)
            output = self.dropout(output)

            # --------- DECODER ---------
            # run the Seq2Seq gru decoder
            output, hidden = self.gru(output, hidden) # 1, 1, mcan_flat_out_size # 1, 1, mcan_flat_out_size

            # get final scores for each word
            output_vocab = self.out(output[0]) # 1, 1, vocab_size

            # get index of word with highest score
            _, topi = output_vocab.topk(1) # 1, 1

            # assign input for next word
            # when using teacher forcing (during training only), we take the ground truth as the 
            # input for the next step. This results in faster convergence.
            # if we don't use teacher focing, we use the current predicted token as the input for 
            # the next step.
            if self.use_teacher_forcing and not self.testing:
                output = answer_to_vocab[i + 1].unsqueeze(0).unsqueeze(0)
            else:
                output = topi
            
            # save scores on the vocab
            outputs.append(output_vocab)
            
            # save top pred word
            pred_answer.append(topi.item())

            # if we are predicting and the token <end> is generated, then stop
            if self.testing and topi.item() == end_token:
                break

        # convert tokens to string
        pred_answer = [" ".join(self.answer_vocab.lookup_tokens(pred_answer[:-1]))]

        return outputs, pred_answer, out_attn_weights, out_attn_ob_weights
    
    # beam search (can be used instead of the forward(...) function)
    # to use beam search, set use_beam_search manually to True
    def beam_search(self, input, hidden, encoder_outputs, object_proposals, object_mask, lang_mask, depth=1):
        # last predicted token as input to the GRU
        output = input # 1, 1
        # hidden vector of the GRU
        hidden = hidden.repeat(self.num_layers, 1, 1) # num_layers, 1, mcan_flat_out_size

        max_len = CONF.TRAIN.MAX_ANSWER_LEN
        end_token = self.answer_vocab['<end>'] # 1
        softmax = nn.Softmax(dim=1)  
        
        # rename depth
        B = depth

        # first pass
        # get embedding of word from index
        output_context = self.embedding(output).view(1, 1, -1)[0] # 1, emb_size
        output_context = self.relu(output_context)
        output_context = self.dropout(output_context)

        # --------- ATTENTION ---------
        # calculate attention weights of object proposals
        attn_weights_op = self.attn_op(torch.cat((output_context, hidden[0]), 1)) # 1, num_proposal
        attn_weights_op = attn_weights_op.masked_fill(object_mask.bool(), -1e9)
        attn_weights_op = F.softmax(
            attn_weights_op, 
            dim=1
        ) # 1, num_proposal

        # apply attention weights to object proposals and sum them up
        attn_applied_op = torch.bmm(
            attn_weights_op.unsqueeze(0),
            object_proposals.unsqueeze(0)
        )  # 1, 1, hidden_size

        # calculate attention weights of question embeddings
        attn_weights = self.attn(torch.cat((output_context, hidden[0]), 1)) # 1, max_length
        attn_weights = attn_weights.masked_fill(lang_mask.bool(), -1e9)
        attn_weights = F.softmax(
            attn_weights, 
            dim=1
        ) # 1, max_length

        # apply attention weights to question embeddingts and sum them up
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        ) # 1, 1, bi_hidden_size

        # concatenate current word embedding and the attention outputs
        output_context = torch.cat((output_context, attn_applied[0], attn_applied_op[0]), 1) # hidden_size + bi_hidden_size + emb_size

        # reduce to size of hidden vectors
        output_context = self.attn_combine(output_context).unsqueeze(0) # 1, 1, hidden_size
        output_context = F.relu(output_context)
        output_context = self.dropout(output_context)

        # --------- DECODER --------- 
        # run the seq2seq gru model
        output_context, hidden = self.gru(output_context, hidden) # 1, 1, mcan_flat_out_size # 1, 1, mcan_flat_out_size

        # get probabilities for first word
        prob = softmax(self.out(output_context[0])) # 1, 1, vocab_size

        # get top B words
        values, indicies = torch.topk(prob, B, dim=1)

        # concat <start> and top B words to make B sentences
        values = values.squeeze(0).unsqueeze(-1).unsqueeze(-1)
        output = output.repeat(1, B) # 1, B
        initial = torch.cat([output.unsqueeze(-1), indicies.unsqueeze(-1)], dim=-1).squeeze(0).unsqueeze(-1) # B, 2 (start & first word), 1

        # concat scores for sorting later
        initial = torch.cat([initial, values], dim=1)

        # store top B sentences for every sample in the batch
        result_sample = []
        n = 0
        # run beam search steps until B is zero or we reach the max answer length
        while (B > 0) and (n < max_len):
            n = n + 1

            # store current top sequences 
            new_initial = None

            # for every sentence in initial calculate top B next words
            for b in range(B):
                seq_i = initial[b][:-1].long() 
                last_output_i = initial[b][-2].long() 
                # join probabilitz of sequence i
                p = initial[b][-1]

                # get embedding of word from index
                output_context = self.embedding(last_output_i).view(1, 1, -1)[0] # 1, emb_size
                output_context = self.relu(output_context)
                output_context = self.dropout(output_context)

                # --------- ATTENTION ---------
                # calculate attention weights of object proposals
                attn_weights_op = self.attn_op(torch.cat((output_context, hidden[0]), 1)) # 1, num_proposal
                attn_weights_op = attn_weights_op.masked_fill(object_mask.bool(), -1e9)
                attn_weights_op = F.softmax(
                    attn_weights_op, 
                    dim=1
                ) # 1, num_proposal

                # apply attention weights to object proposals and sum them up
                attn_applied_op = torch.bmm(
                    attn_weights_op.unsqueeze(0),
                    object_proposals.unsqueeze(0)
                )  # 1, 1, hidden_size

                # calculate attention weights of question embeddings
                attn_weights = self.attn(torch.cat((output_context, hidden[0]), 1)) # 1, max_length
                attn_weights = attn_weights.masked_fill(lang_mask.bool(), -1e9)
                attn_weights = F.softmax(
                    attn_weights, 
                    dim=1
                ) # 1, max_length

                # apply attention weights to question embeddingts and sum them up
                attn_applied = torch.bmm(
                    attn_weights.unsqueeze(0),
                    encoder_outputs.unsqueeze(0)
                ) # 1, 1, bi_hidden_size

                # concatenate current word embedding and the attention outputs
                output_context = torch.cat((output_context, attn_applied[0], attn_applied_op[0]), 1) # hidden_size + bi_hidden_size + emb_size

                # reduce to size of hidden vectors
                output_context = self.attn_combine(output_context).unsqueeze(0) # 1, 1, hidden_size
                output_context = F.relu(output_context)
                output_context = self.dropout(output_context)

                # --------- DECODER ---------
                # run the seq2seq gru model
                output_context, hidden = self.gru(output_context, hidden) # 1, 1, mcan_flat_out_size # 1, 1, mcan_flat_out_size

                # probabilties of next word for sequence b
                prob = softmax(self.out(output_context[0])) # 1, 1, vocab_size

                # get top B words
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

            # update (decrease) B
            B = len(initial)
            initial = torch.tensor(initial, device=DEVICE).unsqueeze(-1)
            result_sample.extend(finished_seq)

            # sort the B sentences created for the sample
            result_sample = sorted(result_sample, key=lambda x: -1 * x[-1])

        # format answer
        pred_answers_tokens = [answer[:-1] for answer in result_sample]
        pred_answers_tokens = [[int(token_id) for token_id in answer] for answer in pred_answers_tokens]
        pred_answers_probs = [answer[-1] for answer in result_sample]
        pred_answers = [" ".join(self.answer_vocab.lookup_tokens(answer[1:-1])) for answer in pred_answers_tokens]
        pred_answers_prob = pred_answers_probs

        return pred_answers_prob, pred_answers

class AnswerModule(nn.Module):
    def __init__(self, vocab_size, bi_hidden_size, answer_vocab, num_proposal=256, emb_size=300, hidden_size=256, mcan_flat_out_size=512, num_layers=1):
        super().__init__() 

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mcan_flat_out_size = mcan_flat_out_size
        self.bi_hidden_size = bi_hidden_size
        self.num_proposal = num_proposal
        self.answer_vocab = answer_vocab
        self.testing = False

        self.decoder = AttnDecoderRNN(vocab_size=vocab_size, hidden_size=hidden_size, mcan_flat_out_size=mcan_flat_out_size, emb_size=emb_size, bi_hidden_size=bi_hidden_size, num_proposal=num_proposal, answer_vocab=answer_vocab)
    
    def forward(self, data_dict):
        # unpack
        contexts = data_dict['contexts'] # batch_size, mcan_flat_out_size
        object_proposals = data_dict['object_proposals'] # batch_size, num_proposals, hidden_size
        object_masks = data_dict['bbox_mask']
        lang_feat = data_dict['lang_out'] # batch_size, num_words(max_question_length), hidden_size * num_dir
        lang_masks = data_dict['lang_mask'] # batch_size, num_words (max_question_length)
        if not self.testing:
            # get ground truth
            answers_to_vocab = data_dict['answers_to_vocab']
            answers_len = data_dict['answers_len']

        batch_size = lang_feat.size()[0]

        batch_outputs = []
        batch_pred_answers = []
        batch_attn_lang_weights = []
        batch_attn_ob_weights = []
        
        # get input <start>
        start_token = torch.full((1, 1), self.answer_vocab['<start>'], dtype=torch.long, device=DEVICE) # 1, 1

        # for every sample in the batch
        for i in range(batch_size):
            # reshape context
            context = contexts[i].unsqueeze(0).unsqueeze(0) # 1, 1, mcan_flat_out_size

            if not self.testing:
                # pack answer
                answers_to_vocab_i = pack_padded_sequence(answers_to_vocab[i].unsqueeze(0), answers_len[i].unsqueeze(0).cpu(), batch_first = True, enforce_sorted=False)[0]
            else:
                answers_to_vocab_i = None

            # get question encodings of the sample
            encoder_outputs = torch.zeros(CONF.TRAIN.MAX_TEXT_LEN, self.hidden_size, device=DEVICE) # max_text_length, hidden_size
            encoder_outputs[0:lang_feat.shape[1]] = lang_feat[i]

            # get question mask for the sample
            lang_mask = torch.ones(CONF.TRAIN.MAX_TEXT_LEN, device=DEVICE) # max_text_length, hidden_size
            lang_mask[0:lang_masks[i].shape[0]] = lang_masks[i]

            # run decoder
            outputs, pred_answer, attn_weights, attn_ob_weights = self.decoder(
                start_token, 
                context, 
                encoder_outputs, 
                object_proposals[i].squeeze(0), 
                object_masks[i], 
                lang_mask,
                answers_to_vocab_i
            )

            # save results
            batch_outputs.append(outputs)
            batch_pred_answers.append(pred_answer)
            batch_attn_lang_weights.append(attn_weights)
            batch_attn_ob_weights.append(attn_ob_weights)

        # save attention weights
        data_dict['attn_lang_weights'] = batch_attn_lang_weights
        data_dict['attn_ob_weights'] = batch_attn_ob_weights

        # save predicted answers
        data_dict['pred_answers'] = batch_pred_answers

        # combine output arrays into one
        data_dict['answer_pred_outputs'] = batch_outputs
        out_combined = torch.tensor([], device = DEVICE)
        for i, outputs in enumerate(batch_outputs):
            for ten in outputs:
                out_combined = torch.cat([out_combined, ten])
        out_combined = out_combined.squeeze(1)
        # save combined outputs for loss
        data_dict['out_combined'] = out_combined

        return data_dict
    
    def set_testing(self, value):
        self.testing = value
        self.decoder.testing = value