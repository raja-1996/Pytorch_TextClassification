
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from os.path import join

from Attention_Classification.WordAttn import WordAttn
from Attention_Classification.SentenceAttn import SentenceAttn

class HierarchicalAttention(nn.Module):

    def __init__(self, config):
        super(HierarchicalAttention, self).__init__()

        self.batch_size = config['batch_size']
        self.n_layers = config['n_layers']
        self.max_sents = config['max_sents']
        self.hidden_size = config['hidden_size']
        self.num_classes = config['num_classes']

        self.device = config['device']

        self.word_attn = WordAttn(config)
        self.sent_attn = SentenceAttn(config)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        path_to_model = join(config['model_dir'], config['model_name'])
        if os.path.exists(path_to_model):
            self.load_state_dict(torch.load(path_to_model, map_location=lambda storage, loc: storage))
            print('Model loaded from disk !!!! {}'.format(path_to_model))

    def init_hidden_state(self, input_ids):
        bz = input_ids.size(0)
        self.word_hidden_state = torch.zeros(2*self.n_layers, self.max_sents*bz, self.hidden_size).to(self.device)
        self.sent_hidden_state = torch.zeros(2*self.n_layers, bz, self.hidden_size).to(self.device)

    def forward(self, input, word_lengths, sent_lengths):
        # word_lengths => B, S
        # sent_lengths => B
        # input => Batch, Max_Sent, Max_Words
        # word_lengths => Batch, Max_Sent
        self.init_hidden_state(input)

        bs, ms, mw = input.size()
        input = input.view(ms*bs, mw)
        word_lengths = word_lengths.view(ms*bs)
        # word_lengths => Batch* Max_Sent

        output, self.word_hidden_state, word_attn_scores = self.word_attn(input, self.word_hidden_state, word_lengths)
        # output => S*B, 2*H
        output = output.view(bs, ms, -1)
        # output => B, S, 2*H

        word_attn_scores = word_attn_scores.view(bs, ms, -1)

        output, self.sent_hidden_state, sent_attn_scores = self.sent_attn(output, self.sent_hidden_state, sent_lengths)
        # output => B, 2*H
        
        logits = self.fc(output)
        # logits => B, C

        return logits, word_attn_scores, sent_attn_scores


