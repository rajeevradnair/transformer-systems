import torch
from torch import nn


vocab_size = 20
hidden_width = 8

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=hidden_width,
)

invalid_token_ids = torch.tensor(
    [[1, 4, vocab_size, 3]],
    dtype=torch.long,
)

print("vocab_size:", vocab_size)
print("valid token range:", f"0 through {vocab_size - 1}")
print("observed token IDs:", invalid_token_ids.tolist())

embedding(invalid_token_ids)