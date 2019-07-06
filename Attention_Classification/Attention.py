
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Attention(nn.Module):

    def __init__(self, config):
        super(Attention, self).__init__()

        self.device = config['device']
        self.attention_size = config['hidden_size']
        self.attn_vector = nn.Linear(self.attention_size, 1)
        # self.attn_vector = Parameter(torch.FloatTensor(self.attention_size))

    def get_mask(self, scores, lengths):

        max_len = scores.size(1)
        mask = torch.arange(0, max_len).unsqueeze(0)
        mask = mask.to(self.device)

        mask = mask<lengths.unsqueeze(1)
        mask = mask.float().clone().detach()

        return mask

    def forward(self, input, lengths):

        # input B, T, H
        # attn_vector H
        logits = self.attn_vector(input)
        logits = logits.squeeze(-1)
        # logits = input.matmul(self.attn_vector)

        # B, T
        scores: torch.Tensor = F.softmax(logits, dim=-1)
        #B, T
        mask = self.get_mask(scores, lengths)
        scores = scores*mask

        output = input.mul(scores.unsqueeze(-1).expand_as(input))
        output = output.sum(dim=1)

        return output, scores


