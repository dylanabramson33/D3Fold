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

def create_block_lowert_mask(seq_len, block_size, device):
    block = torch.ones((block_size,block_size)).to(device)

    # Assume all blocks are of the same size (2x2 in this case)
    block_size = block.size(0)

    # Define the number of block rows and columns
    num_blocks = seq_len

    # Initialize a large zero matrix
    result = torch.zeros(block_size * num_blocks, block_size * num_blocks, device=device)

    # Fill in the lower triangular part with the blocks
    # blocks = [[A, None, None, None], [A, A, None, None], [A, A, A, None], [A, A, A,A]]
    # do similar pattern to above but for num_blocks
    blocks = [[block if i >= j else None for j in range(num_blocks)] for i in range(num_blocks)]

    for i in range(num_blocks):
        for j in range(i + 1):
            if blocks[i][j] is not None:
                result[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[i][j]
    return result

class CausalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CausalAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} should be divisible by the number of heads {num_heads}.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # replace padding index with num_tokens - 1
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn_output, _ = self.attn(q, k, v, attn_mask=~mask if isinstance(mask, type(None)) else None)
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