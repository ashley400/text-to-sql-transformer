import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "starter")))
from model.layers import Encoder,Decoder
from embeddings import InputLayer, TokenEmbedding

class Transformer(nn.Module):
    def __init__(self,vocab_size,d_model=256,h=4,N=3,d_ff=1024,dropout=0.1,pad_id=0,max_len=512):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.src_input = InputLayer(self.embedding, d_model, max_len, dropout)
        self.tgt_input = InputLayer(self.embedding, d_model, max_len, dropout)
        self.encoder = Encoder(N, d_model, h, d_ff, dropout)
        self.decoder = Decoder(N, d_model, h, d_ff, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embedding.emb.weight
        nn.init.normal_(self.embedding.emb.weight, mean=0.0, std=d_model ** -0.5)
    def source_mask(self, source):
        return (source != self.pad_id)[:, None, None, :]
    def target_mask(self, target):
        T = target.size(1)
        not_pad = (target != self.pad_id)[:, None, None, :]
        no_future = torch.ones(T, T, dtype=torch.bool, device=target.device).tril()[None, None]
        return not_pad & no_future
    def encode(self, source):
        source_mask = self.source_mask(source)
        embedded = self.src_input(source)
        encoder_output = self.encoder(embedded, source_mask)
        return encoder_output, source_mask
    def decode(self, target, encoder_output, source_mask):
        target_mask = self.target_mask(target)
        embedded = self.tgt_input(target)
        hidden = self.decoder(embedded, encoder_output, source_mask, target_mask)
        return self.out_proj(hidden)
    def forward(self, source, target):
        encoder_output, source_mask = self.encode(source)
        return self.decode(target, encoder_output, source_mask)