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


