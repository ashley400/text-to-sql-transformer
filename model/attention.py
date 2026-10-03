import math
import torch
import torch.nn as nn

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores=torch.matmul(q,k.transpose(-2,-1))
        scores=scores/math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, h=4):
        super().__init__()
        assert d_model% h==0
        self.h =h
        self.d_k =d_model // h
        self.w_q =nn.Linear(d_model, d_model)
        self.w_k= nn.Linear(d_model, d_model)
        self.w_v= nn.Linear(d_model, d_model)
        self.w_o=nn.Linear(d_model, d_model)
        self.attention=ScaledDotProduct()

    def forward(self, q, k, v, mask=None):
        batch=q.size(0)
        q=self.w_q(q).view(batch,-1,self.h,self.d_k).transpose(1, 2)
        k=self.w_k(k).view(batch,-1,self.h,self.d_k).transpose(1, 2)
        v=self.w_v(v).view(batch,-1,self.h,self.d_k).transpose(1, 2)
        output,weights= self.attention(q,k,v,mask)                    
        output=output.transpose(1, 2).contiguous().view(batch,-1,self.h*self.d_k)
        output=self.w_o(output)
        return output,weights


