import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

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

class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, h=4, d_ff=1024,dropout=0.1):
        super().__init__()
        self.feedforwardnet=PositionwiseFeedForward(d_model,d_ff)
        self.attention=MultiHeadAttention(d_model,h)
        self.dropout= nn.Dropout(dropout)
        self.normalization1= nn.LayerNorm(d_model)
        self.normalization2= nn.LayerNorm(d_model)

    def forward(self,x,mask):
        output,_=self.attention(x,x,x,mask)
        x= self.normalization1(x+ self.dropout(output))
        ffn= self.feedforwardnet(x)
        x= self.normalization2(x+ self.dropout(ffn))
        return x


class Encoder(nn.Module):
    def __init__(self, N=3,d_model=256,h=4,d_ff=1024,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h,d_ff,dropout) for _ in range(N)])

    def forward(self, x,mask):
        for layer in self.layers:
            x =layer(x,mask)
        return x