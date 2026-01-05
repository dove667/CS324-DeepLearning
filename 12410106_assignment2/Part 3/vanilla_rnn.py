from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_dim
        self.seq_length = seq_length
        self.W_xh = nn.Linear(input_dim, hidden_dim) # x(t)->h(t)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim) # h(t-1)->h(t)
        self.W_hy = nn.Linear(hidden_dim, output_dim) # h(t)->y(t) :predict of x(t)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None, batch_first=False):
        # x: (seq_len, batch, input) or (batch, seq_len, input) if batch_first=True
        if batch_first:
            x = x.transpose(0, 1)  # to (seq_len, batch, input)
        if hidden is None:
            hidden = x.new_zeros(x.size(1), self.hidden_size)
        for t in range(self.seq_length):
            hidden = self.tanh(self.W_xh(x[t]) + self.W_hh(hidden))
        output = self.W_hy(hidden)
        return output # (batch, output_dim)

