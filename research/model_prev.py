import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    '''
    Implementation of Layer Normalization as described in the paper
    Initialization:
        features: Dimension of the model
        eps: A small value to prevent division by zero
    Input:
        x: Tensor of shape (batch, seq_len, hidden_size)
    Output: 
        Tensor of shape (batch, seq_len, hidden_size)
    '''

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps # prevents division by zero
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    '''
    Implementation of Feed Forward Block as described in the paper
    Initialization:
        d_model: Dimension of the model
        d_ff: Dimension of the feed forward layer
        dropout: Dropout rate
    Input:
        x: Tensor of shape (batch, seq_len, d_model)
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    '''
    Implementation of Input Embeddings as described in the paper
    Initialization:
        d_model: Dimension of the model
        vocab_size: Size of the vocabulary
    Input:
        x: Tensor of shape (batch, seq_len)
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    '''
    Implementation of Positional Encoding as described in the paper
    Initialization:
        d_model: Dimension of the model
        seq_len: Maximum sequence length
        dropout: Dropout rate
    Input:
        x: Tensor of shape (batch, seq_len, d_model)
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector for positions [0, 1, 2, ..., max_len-1] of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term (the denominator in the formula)
        # Create a vector of shape (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * e^(10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * e^(10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    '''
    Implementation of Residual Connection as described in the paper
    Initialization:
        features: Dimension of the model
        dropout: Dropout rate
    Input:
        x: Tensor of shape (batch, seq_len, d_model)
        sublayer: A function that takes in a Tensor of shape (batch, seq_len, d_model) and returns a Tensor of the same shape
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    '''
    Implementation of Multi-Head Attention Block as described in the paper
    Initialization:
        d_model: Dimension of the model
        h: Number of heads
        dropout: Dropout rate
    Input:
        q: Tensor of shape (batch, seq_len, d_model)
        k: Tensor of shape (batch, seq_len, d_model)
        v: Tensor of shape (batch, seq_len, d_model)
        mask: Tensor of shape (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):
    '''
    Implementation of Encoder Block as described in the paper
    Initialization:
        features: Dimension of the model
        self_attention_block: An instance of MultiHeadAttentionBlock
        feed_forward_block: An instance of FeedForwardBlock
        dropout: Dropout rate
    Input:
        x: Tensor of shape (batch, seq_len, d_model)
        src_mask: Tensor of shape (batch, 1, 1, seq_len)
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    '''
    Implementation of Encoder as described in the paper
    Initialization:
        features: Dimension of the model
        layers: A ModuleList of EncoderBlock instances
    Input:
        x: Tensor of shape (batch, seq_len, d_model)
        mask: Tensor of shape (batch, 1, 1, seq_len)
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    '''
    Implementation of Decoder Block as described in the paper
    Initialization:
        features: Dimension of the model
        self_attention_block: An instance of MultiHeadAttentionBlock for self-attention
        cross_attention_block: An instance of MultiHeadAttentionBlock for cross-attention
        feed_forward_block: An instance of FeedForwardBlock
        dropout: Dropout rate
    Input:
        x: Tensor of shape (batch, seq_len, d_model)
        encoder_output: Tensor of shape (batch, seq_len, d_model)
        src_mask: Tensor of shape (batch, 1, 1, seq_len)
        tgt_mask: Tensor of shape (batch, 1, seq_len, seq_len)
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    '''
    Implementation of Decoder as described in the paper
    Initialization:
        features: Dimension of the model
        layers: A ModuleList of DecoderBlock instances
    Input:
        x: Tensor of shape (batch, seq_len, d_model)
        encoder_output: Tensor of shape (batch, seq_len, d_model)
        src_mask: Tensor of shape (batch, 1, 1, seq_len)
        tgt_mask: Tensor of shape (batch, 1, seq_len, seq_len)
    Output:
        Tensor of shape (batch, seq_len, d_model)
    '''

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    '''
    Implementation of Projection Layer as described in the paper
    Initialization:
        d_model: Dimension of the model
        vocab_size: Size of the vocabulary
    Input:
        x: Tensor of shape (batch, seq_len, d_model)
    Output:
        Tensor of shape (batch, seq_len, vocab_size)
    '''

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)
    
class Transformer(nn.Module):
    '''
    Implementation of Transformer as described in the paper
    Initialization:
        encoder: An instance of Encoder
        decoder: An instance of Decoder
        src_embed: An instance of InputEmbeddings for source language
        tgt_embed: An instance of InputEmbeddings for target language
        src_pos: An instance of PositionalEncoding for source language
        tgt_pos: An instance of PositionalEncoding for target language
        projection_layer: An instance of ProjectionLayer
    Input:
        src: Tensor of shape (batch, src_seq_len)
        src_mask: Tensor of shape (batch, 1, 1, src_seq_len
        tgt: Tensor of shape (batch, tgt_seq_len)
        tgt_mask: Tensor of shape (batch, 1, tgt_seq_len, tgt_seq
    Output:
        Tensor of shape (batch, tgt_seq_len, vocab_size)
    '''

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    '''
    Docstring for build_transformer
    Inputs:
        src_vocab_size: Size of the source vocabulary
        tgt_vocab_size: Size of the target vocabulary
        src_seq_len: Maximum sequence length for source language
        tgt_seq_len: Maximum sequence length for target language
        d_model: Dimension of the model
        N: Number of encoder and decoder blocks
        h: Number of heads in multi-head attention
        dropout: Dropout rate
        d_ff: Dimension of the feed forward layer
    Outputs:
        Transformer: An instance of the Transformer model
    '''
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer