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
    
    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, hiddeN_size)
        mean = x.mean(dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        var = x.var(dim=-1, keepdim=True) # (batch_size, seq_len, 1)

        return self.alpha * (x - mean) / torch.sqrt((var + self.eps) )+ self.bais
    
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
        self.attention_scores= None
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) #(batch_size, h, seq_len, seq_len)
        if mask is None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score =attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.W_Q(q) # (batch_size, seq_len, d_model)
        key = self.W_K(k) # (batch_size, seq_len, d_model)
        value = self.W_V(v) # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiheadAttentionBlock.attention(query, key, value, self.dropout)

        #(batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, -1, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.W_O(x) #batch_size, seq_len, d_model


class ResidualConnection(nn.Module):

    def __init__(self, feature: int, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm()
    
    def forward(self, x:float, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, num_blocks: int = 2):
        super().__init__()
        self.attention_block = self_attention
        self.feed_forward_blcok = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout)] for _ in range(num_blocks))
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask)) # here we use lambda since it is a little more complex than passing x into the module, multiple inputs are required
        x = self.residual_connections[1](x, self.feed_forward_block)

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        super.__init__()
        self.layers = layers # list of EncoderBlocks
        self.norm = nn.LayerNorm(features)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiheadAttentionBlock,
                 cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x,x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.proj(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionEncoding,
                 tgt_pos: PositionEncoding, proj_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_otuput: torch.Tensor, src_mask: torch.tesnor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_otuput, src_mask, tgt_mask)

    def project(self, x):
        return self.proj_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                    d_model: int = 512, N: int = 6, h: int = 8, dropout: float=0.1, d_ff: int = 2048):
    # embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # positional encoder
    src_pos = PositionEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionEncoding(d_model, tgt_seq_len, dropout)

    #Encoder
    encoder_blocks = []
    for _ in range(N):
        self_attention = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Decoder
    decoder_blocks = []
    for _ in range(N):
        self_attention = MultiheadAttentionBlock(d_model, h, dropout)
        cross_attention = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, self_attention, cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    project_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, project_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
