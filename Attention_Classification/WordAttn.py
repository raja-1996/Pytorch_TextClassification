
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from Attention_Classification.Attention import Attention

class WordAttn(nn.Module):

    def __init__(self, config):
        super(WordAttn, self).__init__()

        vocab_size = config['vocab_size']
        embed_size = config['embed_size']
        hidden_size = config['hidden_size']
        rnn_type = config['rnn_type']
        n_layers = config['n_layers']
        dropout = config['dropout']
        bidir = config['bidir']

        load_pretrained_word_embeddings = config['load_pretrained_word_embeddings']

        if load_pretrained_word_embeddings:
            weight_matrix = self.load_matrix(config['weight_matrix_path'])
            vocab_size, embed_size = weight_matrix.shape
            print('vocab_size {}, Embed_size {}'.format(vocab_size, embed_size))
            
            self.embedding = nn.Embedding(vocab_size, embed_size).from_pretrained(weight_matrix)

            flag = True
            self.embedding.weight.requires_grad = flag
            print('Using Pretrained Word Embeddings and requires_grad is {}'.format(flag))
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = getattr(nn, rnn_type)(input_size=embed_size,
                                         hidden_size=hidden_size,
                                         num_layers=n_layers,
                                         dropout=dropout,
                                         bidirectional=bidir)

        self.attn = Attention(config)
        self.fc = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(config['dropout'])

    def load_matrix(self, weight_matrix_path):
        embedding_matrix = np.load(weight_matrix_path)
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
        return embedding_matrix

    def forward(self, input, hidden_state, lengths):

        embed = self.embedding(input)
        # Input S*B, W
        # Embed S*B, W, E

        embed = embed.permute(1, 0, 2)
        # Embed W, S*B, E

        outputs, hidden_state = self.rnn(embed, hidden_state)
        # outputs => W, S*B, 2*H
        # Hidden => n_layers*2, S*B, H

        outputs = outputs.permute(1, 0, 2)
        # outputs => S*B, W, 2*H

        outputs = self.fc(outputs)
        outputs = F.relu(outputs)
        outputs = self.dropout(outputs)

        outputs, attn_scores = self.attn(outputs, lengths)
        # ouputs => S*B, 2*H

        return outputs, hidden_state, attn_scores















