
import torch
import torch.nn as nn
import torch.nn.functional as F

from Attention_Classification.Attention import Attention


class SentenceAttn(nn.Module):

    def __init__(self, config):
        super(SentenceAttn, self).__init__()

        vocab_size = config['vocab_size']
        embed_size = config['embed_size']
        hidden_size = config['hidden_size']
        rnn_type = config['rnn_type']
        n_layers = config['n_layers']
        dropout = config['dropout']
        bidir = config['bidir']

        self.rnn = getattr(nn, rnn_type)(input_size=hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=n_layers,
                                         dropout=dropout,
                                         bidirectional=bidir)

        self.attn = Attention(config)
        self.fc = nn.Linear(2*hidden_size, hidden_size)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, input, hidden_state, lengths):

        input = input.permute(1, 0, 2)
        #input => B, S, 2*H

        outputs, hidden_state = self.rnn(input, hidden_state)
        # outputs => S, B, 2*H
        # Hidden => n_layers*2, B, 2*H

        outputs = outputs.permute(1, 0, 2)
        # outputs => B, S, 2*H

        outputs = self.fc(outputs)
        outputs = F.relu(outputs)
        outputs = self.dropout(outputs)
        
        outputs, attn_scores = self.attn(outputs, lengths)
        # ouputs => B, 2*H

        return outputs, hidden_state, attn_scores