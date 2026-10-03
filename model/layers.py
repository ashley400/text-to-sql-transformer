import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, dim=10, hidden=40, p=0.2):
        super().__init__()
        self.w1= nn.Linear(dim, hidden)
        self.w2= nn.Linear(hidden, dim)
        self.activation= nn.ReLU()
        self.dropout= nn.Dropout(p)
        self.normalization= nn.LayerNorm(dim)

    def forward(self, x):
        y= self.w1(x)
        y= self.activation(y)
        y= self.w2(y)
        y= self.dropout(y)
        return self.normalization(x+y)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=256, d_ff=1024):
        super().__init__()
        self.w1= nn.Linear(d_model, d_ff)
        self.w2= nn.Linear(d_ff, d_model)
        self.activation= nn.ReLU()

    def forward(self, x):
        y= self.w1(x)
        y= self.activation(y)
        y= self.w2(y)
        return y