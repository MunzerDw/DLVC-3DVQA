""" 
Modified from: https://github.com/ATR-DBI/ScanQA/blob/main/models/lang_module.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence    

class LangModule(nn.Module):
    def __init__(self, num_object_class, use_lang_classifier=True, use_bidir=False, num_layers=1,
        emb_size=300, hidden_size=256, pdrop=0.1, word_pdrop=0.1):
        super().__init__() 

        self.num_object_class = num_object_class
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.num_layers = num_layers             

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=use_bidir,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.word_drop = nn.Dropout(pdrop)

        lang_size = hidden_size * 2 if use_bidir else hidden_size

        #
        # Language classifier
        #   num_object_class -> 18
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Dropout(p=pdrop),
                nn.Linear(lang_size, num_object_class),
                #nn.Dropout()
            )

    def make_mask(self, feature):
        """
        return a mask that is True for zero values and False for other values.
        """
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0) #.unsqueeze(-1) #.unsqueeze(2)        


    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        word_embs = data_dict["lang_feat"] # batch_size, MAX_TEXT_LEN (32), glove_size
        # dropout word embeddings
        word_embs = self.word_drop(word_embs)
        lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"].cpu(), batch_first=True, enforce_sorted=False)

        # encode description
        packed_output, (lang_last, _) = self.lstm(lang_feat)
        lang_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        data_dict["lang_out"] = lang_output # batch_size, num_words(max_question_length), hidden_size * num_dir

        # lang_last: (num_layers * num_directions, batch_size, hidden_size)
        _, batch_size, hidden_size = lang_last.size()
        lang_last = lang_last.view(self.num_layers, -1, batch_size, hidden_size) 
        # lang_last: num_directions, batch_size, hidden_size
        lang_last = lang_last[-1]
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir

        # store the encoded language features
        data_dict["lang_emb"] = lang_last # batch_size, hidden_size * num_dir
        data_dict["lang_mask"] = self.make_mask(lang_output) # batch_size, num_words (max_question_length)

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])
        return data_dict
