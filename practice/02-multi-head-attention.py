# %% [markdown]
# # Drill 2: Multi-Head Attention
# Implement multi-head self-attention from scratch.
# Input: (B, T, D). Output: (B, T, D).
# Steps:
# 1. Create W_q, W_k, W_v, W_o projection matrices (D x D)
# 2. Project input to Q, K, V
# 3. Reshape (B, T, D) -> (B, H, T, d_k) where d_k = D // H
# 4. Attention scores per head: Q @ K^T / sqrt(d_k)
# 5. Optional causal mask, then softmax
# 6. Weighted sum: weights @ V -> (B, H, T, d_k)
# 7. Recombine heads: (B, H, T, d_k) -> (B, T, D)
# 8. Output projection: result @ W_o

# %%
import torch
import torch.nn as nn
import math

# %%
class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, causal=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        # TODO: create W_q (d_model, d_model)
        # TODO: create W_k (d_model, d_model)
        # TODO: create W_v (d_model, d_model)
        # TODO: create W_o (d_model, d_model)
        pass

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        # TODO step 1: project Q, K, V  -> each (B, T, D)
        # TODO step 2: reshape to (B, T, H, d_k) then transpose to (B, H, T, d_k)
        # TODO step 3: scores = Q @ K^T / sqrt(d_k)  -> (B, H, T, T)
        # TODO step 4: if self.causal, apply upper-triangular mask with -inf
        # TODO step 5: softmax over last dim
        # TODO step 6: weights @ V  -> (B, H, T, d_k)
        # TODO step 7: transpose back to (B, T, H, d_k) then reshape to (B, T, D)
        # TODO step 8: output projection  -> (B, T, D)
        raise NotImplementedError

# %%
# TEST 1: shape check
x = torch.randn(2, 5, 64)
model = MyMultiHeadAttention(d_model=64, num_heads=8)
out = model(x)
assert out.shape == (2, 5, 64), f'Wrong shape: {out.shape}'
print('PASS: shape check (8 heads)')

# %%
# TEST 2: different head counts
for h in [1, 2, 4, 8, 16]:
    m = MyMultiHeadAttention(d_model=64, num_heads=h)
    o = m(x)
    assert o.shape == (2, 5, 64), f'Wrong shape with {h} heads: {o.shape}'
print('PASS: all head counts')

# %%
# TEST 3: causal mode shape check
model_c = MyMultiHeadAttention(d_model=64, num_heads=8, causal=True)
out_c = model_c(x)
assert out_c.shape == (2, 5, 64), f'Wrong shape: {out_c.shape}'
print('PASS: causal shape check')
