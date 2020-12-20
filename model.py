from torch import nn
import torch
from encoder import Encoder, PositionalEncoding
from memory import MemoryLSTM

class Model(nn.Module):
    
    def __init__(self, vocab_size, num_blocks, num_heads, zvx_size, qk_size, dropout=0.4, use_memory=True):
        super(Model, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, zvx_size)
        self.pos_enc = PositionalEncoding(zvx_size)

        self.use_memory = use_memory
        if use_memory:
            self.mem1 = MemoryLSTM(zvx_size, zvx_size, nlayers=4, hiddensize=512, dropout=0.4)

        self.encoder = Encoder(num_blocks, num_heads, zvx_size, qk_size, dropout)
        self.final1 = nn.Linear(zvx_size, vocab_size)

    def step(self):
      if self.use_memory:
          self.mem1.step()

    def reset(self):
      if self.use_memory:
          self.mem1.reset()

    def forward(self, word_vectors):

        x_vectors = self.embeddings(word_vectors)

        posenc_vectors = self.pos_enc(x_vectors)

        if self.use_memory:
            z_posvectors = self.encoder(posenc_vectors)
            mem_vectors = self.mem1(posenc_vectors)
            z_memvectors = self.encoder(mem_vectors)
            out_vectors = torch.mean(z_posvectors + z_memvectors, dim=-2)
        else:
            z_vectors = self.encoder(posenc_vectors) # B x W x V
            out_vectors = torch.mean(z_vectors, dim=-2)

        out_vectors = self.final1(out_vectors)

        return out_vectors