import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
    
    def forward(self, x: torch.Tensor):
        # (batch_size, seq_len) -->  (batch_size, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        position_encoding = torch.zeros(self.seq_len, self.d_model) # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #(seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) /  self.d_model)) # (d_model/2) help with numeric stability (same result)
        position_encoding[:,0::2] = torch.sin(position * div_term) # sin to even position
        position_encoding[:,1::2] = torch.cos(position * div_term) # cos to odd position

        # add dimention for batch_size
        position_encoding = position_encoding.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", position_encoding)

    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch_size, seq_len, d_model)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # learnable
        self.bias = nn.Parameter(torch.zeros(features)) #learanble
    
    def forward(self, x):
        # x: (batch_size, seq_len, hiddeN_size)
        mean = x.mean(dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (batch_size, seq_len, 1)

        return self.alpha * (x - mean) / (std + self.eps) + self.bais
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(self.d_model,self.d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model)
    

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))




class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h # number of head

        assert d_model % h == 0, "d_model given is not divisible by h"

        self.d_k = d_model // h
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) #(batch_size, h, seq_len, seq_len)
        if mask is None:
            attention_score.masked_fill_(mask == 0, -1e9)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.W_Q(q) # (batch_size, seq_len, d_model)
        key = self.W_K(k) # (batch_size, seq_len, d_model)
        value = self.W_V(v) # (batch_size, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiheadAttentionBlock.attention(query, key, value, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.W_O(x)


