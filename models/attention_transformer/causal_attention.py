import torch
import torch.nn as nn

def create_causal_mask(seq_len, device):
    """
    Create a causal mask for self-attention.

    Args:
        seq_len: Length of the input sequence.
        device: Device to create the mask on (e.g., 'cpu' or 'cuda').

    Returns:
        A boolean mask tensor of shape (seq_len, seq_len).
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(device)
    return mask

class CausalAttention(nn.Module):
    def __init__(self, num_tokens, embed_dim, num_heads):
        super(CausalAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} should be divisible by the number of heads {num_heads}.")
        self.embed_dim = embed_dim
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask):
        # replace padding index with num_tokens - 1
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return attn_output

class AttentionBlock(nn.Module):
    def __init__(self, num_tokens, embed_dim, num_heads, hidden_size):
        super(AttentionBlock, self).__init__()
        self.attention = CausalAttention(num_tokens, embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim)
        )

    def forward(self, x, mask):
        x = x + self.attention(x, mask)
        x = self.norm1(x)
        x = x + self.feedforward(x)
        return x