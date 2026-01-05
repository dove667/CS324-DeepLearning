from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init

################################################################################

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.hidden_size = hidden_dim
        
        # LSTM params
        # Input gate
        self.W_xi = nn.Linear(input_dim, hidden_dim)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim)

        # Forget gate
        self.W_xf = nn.Linear(input_dim, hidden_dim)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim)

        # Output gate
        self.W_xo = nn.Linear(input_dim, hidden_dim)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim)

        # Cell gate
        self.W_xc = nn.Linear(input_dim, hidden_dim)
        self.W_hc = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.W_hy = nn.Linear(hidden_dim, output_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        for layer in [
            self.W_xi, self.W_hi,
            self.W_xf, self.W_hf,
            self.W_xo, self.W_ho,
            self.W_xc, self.W_hc,
            self.W_hy,
        ]:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        # Implementation here ...
        # x shape: (seq_len, batch_size, input_size)
        # hidden: tuple of (h_0, c_0) where h_0: (batch_size, hidden_size), c_0: (batch_size, hidden_size)

        h = torch.zeros(x.size(1), self.hidden_size, device=x.device)
        c = torch.zeros(x.size(1), self.hidden_size, device=x.device)

        for t in range(x.size(0)):  # travers time step
            # Input gate
            i_t = self.sigmoid(self.W_xi(x[t]) + self.W_hi(h))
            # Forget gate
            f_t = self.sigmoid(self.W_xf(x[t]) + self.W_hf(h))
            # Output gate
            o_t = self.sigmoid(self.W_xo(x[t]) + self.W_ho(h))
            # Cell
            c_tilde = self.tanh(self.W_xc(x[t]) + self.W_hc(h))
            
            # Renew cell
            c = f_t * c + i_t * c_tilde
            # Update
            h = o_t * self.tanh(c)
            
        output = self.W_hy(h)
        return output
    # add more methods here if needed