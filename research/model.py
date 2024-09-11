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
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) /  self.d_model)) # (d_model/2)

        position_encoding[:,0::2] = torch.sin(position * div_term) # sin to even position
        position_encoding[:,1::2] = torch.cos(position * div_term) # cos to odd position

        # add dimention for batch_size
        position_encoding = position_encoding.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", position_encoding)

    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch_size, seq_len, d_model)
        return self.dropout(x)
    




