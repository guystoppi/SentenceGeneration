from torch import nn
import torch

import numpy as np

class AttentionHead(nn.Module):

    def __init__(self, x_size, qk_size, zv_size):
        super(AttentionHead, self).__init__()

        self.query = nn.Linear(x_size, qk_size)
        self.keys = nn.Linear(x_size, qk_size)
        self.value = nn.Linear(x_size, zv_size)
        self.qk_size = qk_size

    def forward(self, x_vectors):
        q_vectors = self.query(x_vectors)
        k_vectors = self.keys(x_vectors)
        v_vectors = self.value(x_vectors)

        e_matrix = torch.matmul(q_vectors, torch.transpose(k_vectors, -1,-2))
        e_matrix /= np.sqrt(self.qk_size)
        e_matrix = torch.nn.functional.softmax(e_matrix, dim=-1)

        z_vectors = torch.matmul(e_matrix, v_vectors)

        return z_vectors

class MultiAttentionHeads(nn.Module):

    def __init__(self, num_heads, x_size, qk_size, zv_size):
        super(MultiAttentionHeads, self).__init__()

        self.heads = [
            AttentionHead(x_size, qk_size, zv_size) for _ in range(num_heads)
        ]

        for j in range(len(self.heads)):
            self.add_module("head%d" % j, self.heads[j])

        self.compresslinear = nn.Linear(num_heads * zv_size, zv_size)

    def forward(self, x_vectors):

        z_vectors = [head(x_vectors) for head in self.heads]

        super_z_vector = torch.cat(z_vectors, dim=-1)

        final_z_vector = self.compresslinear(super_z_vector)

        return final_z_vector

class EncoderDecoderAttentionHead(nn.Module):

    def __init__(self, num_heads, x_size, qk_size, zv_size):
        super(nn.Module, self).__init__()

        self.query = nn.Linear(x_size, qk_size)

    def forward(self, x_vectors, enc_output):

        q_vectors = self.query(x_vectors)
        k_vectors = torch.matmul(enc_output[0], x_vectors)
        v_vectors = torch.matmul(enc_output[1], x_vectors)

        e_matrix = torch.matmul(q_vectors, torch.transpose(k_vectors, -1,-2))
        e_matrix /= torch.sqrt(qk_size)
        e_matrix = torch.nn.functional.softmax(e_matrix, dim=1)

        z_vectors = torch.matmul(e_matrix, torch.transpose(v_vectors, -1,-2))

        return z_vectors
        
class MultiEncDecHeads(nn.Module):

    def __init__(self, num_heads, x_size, qk_size, zv_size):
        super(nn.Module, self).__init__()

        self.heads = [
            EncoderDecoderAttentionHead(x_size, qk_size, zv_size) for _ in range(num_heads)
        ]

        self.compresslinear = nn.Linear(num_heads * zv_size, zv_size)

    def forward(self, x_vectors, enc_output):

        z_vectors = [head(x_vectors, enc_output) for head in self.heads]

        super_z_vector = torch.cat(z_vectors, dim=1)

        final_z_vector = self.compresslinear(super_z_vector)

        return final_z_vector