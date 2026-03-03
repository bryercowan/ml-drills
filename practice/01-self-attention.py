# %% [markdown]
# # Drill 1: Self-Attention
# Implement single-head self-attention from scratch.
# Input: (B, T, D). Output: (B, T, D).
# Steps:
# 1. Create W_q, W_k, W_v projection matrices (D x D)
# 2. Project input to Q, K, V
# 3. Compute attention scores: Q @ K^T / sqrt(d)
# 4. If causal, apply upper-triangular mask (-inf)
# 5. Softmax over last dim
# 6. Weighted sum: weights @ V

# %%
import torch
import torch.nn as nn
import math

# %%
class MySelfAttention(nn.Module):
    def __init__(self, d_model, causal=False):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        # TODO: create W_q (d_model, d_model)
        # TODO: create W_k (d_model, d_model)
        # TODO: create W_v (d_model, d_model)
        pass

    def forward(self, x):
        # x: (B, T, D)
        # TODO step 1: project Q = x @ W_q, K = x @ W_k, V = x @ W_v
        # TODO step 2: compute scores = Q @ K^T / sqrt(d_model)  -> (B, T, T)
        # TODO step 3: if self.causal, create upper-triangular mask and fill scores with -inf
        # TODO step 4: softmax over last dimension
        # TODO step 5: output = weights @ V  -> (B, T, D)
        raise NotImplementedError

# %%
# TEST 1: shape check
x = torch.randn(2, 5, 64)
model = MySelfAttention(64, causal=False)
out = model(x)
assert out.shape == (2, 5, 64), f'Wrong shape: {out.shape}'
print('PASS: shape check (non-causal)')

# %%
# TEST 2: shape check causal
model_c = MySelfAttention(64, causal=True)
out_c = model_c(x)
assert out_c.shape == (2, 5, 64), f'Wrong shape: {out_c.shape}'
print('PASS: shape check (causal)')

# %%
# TEST 3: causal correctness — changing a future token should not affect past outputs
model_c2 = MySelfAttention(64, causal=True)
model_c2.eval()
x1 = torch.randn(1, 5, 64)
x2 = x1.clone()
x2[0, 3, :] += 10.0  # modify token at position 3
with torch.no_grad():
    out1 = model_c2(x1)
    out2 = model_c2(x2)
# positions 0,1,2 should be identical (they can't see position 3)
assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5), 'Causal mask broken: past tokens changed when future token modified'
# position 3 should differ (it sees itself, which changed)
assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-3), 'Position 3 should have changed'
print('PASS: causal correctness')
