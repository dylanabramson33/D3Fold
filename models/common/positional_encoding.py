import math
import torch
from torch import nn

##TODO: should probably refactor to make  more general
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.register_buffer('pe', self.compute_embeddings(max_len, d_model))

    def compute_embeddings(self, num_embeddings, embed_dim):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, data, pad_index=-100):
        indices = data.residue_index
        no_padded_indices = torch.abs(indices)
        #in case pad_index is negative
        min_indices = no_padded_indices.min(dim=1).values.unsqueeze(-1)
        indices_zero_centered = indices - min_indices
        slice_ = self.pe[indices_zero_centered]
        zero_tensor = torch.zeros_like(slice_)
        slice_ = torch.where(indices.unsqueeze(-1) != pad_index, slice_, zero_tensor)
        return slice_