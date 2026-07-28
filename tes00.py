import torch

x = torch.arange(1, 13)
x=x.reshape(1,3,2,2)


# [B, S, num_heads, head_dim]
# → [B, num_heads, S, head_dim]
xt = x.transpose(1, 2)

print()
print(x)
print()
print(xt)
print(x.shape)
print(xt.shape)