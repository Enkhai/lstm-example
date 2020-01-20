import torch
from torch import nn


class CharRNN(nn.Module):

    def __init__(self, num_chars, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()

        self.num_chars = num_chars
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(num_chars, n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, num_chars)

    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    # initializes hidden and cell state to zero
    # hidden state: LTM, cell state: STM
    def init_hidden(self, batch_size, device):

        hidden = (torch.zeros((self.n_layers, batch_size, self.n_hidden)).to(device),
                  torch.zeros((self.n_layers, batch_size, self.n_hidden)).to(device))

        return hidden
