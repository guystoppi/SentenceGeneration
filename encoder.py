from torch import nn
import torch
import math

from attention import MultiAttentionHeads

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EncoderBlock(nn.Module):

    def __init__(self, num_heads, zvx_size, qk_size, dropout=0.4):
        super(EncoderBlock, self).__init__()
        self.selfatt = MultiAttentionHeads(num_heads, zvx_size, qk_size, zvx_size) # Self Attention
        self.drop1 = nn.Dropout(p=dropout)
        self.layernorm1 = nn.LayerNorm(zvx_size)

        self.outlinear = nn.Linear(zvx_size, zvx_size)
        self.drop2 = nn.Dropout(p=dropout)
        self.layernorm2 = nn.LayerNorm(zvx_size)

    def forward(self, x_vectors):
        z_vectors = self.selfatt(x_vectors)
        z_vectors = self.drop1(z_vectors)
        z_prime_vectors = self.layernorm1(x_vectors + z_vectors)

        r_vectors = self.outlinear(z_prime_vectors)
        r_vectors = self.drop2(r_vectors)
        r_prime_vectors = self.layernorm2(z_prime_vectors + r_vectors)

        return r_prime_vectors

class Encoder(nn.Module):

    def __init__(self, num_blocks, num_heads, zvx_size, qk_size, dropout):
        super(Encoder, self).__init__()
        
        self.encoders = nn.Sequential(*[
            EncoderBlock(num_heads, zvx_size, qk_size, dropout) for _ in range(num_blocks)
        ])


    def forward(self, x_vectors):
        
        intermediate = x_vectors
        for enc in self.encoders:
            intermediate = enc(intermediate)

        return intermediate
