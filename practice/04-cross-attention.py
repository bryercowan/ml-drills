# %% [markdown]
# # Drill 4: Cross-Attention
# Implement cross-attention from scratch.
# Queries come from one sequence, keys/values from another.
# Input: queries (B, T_q, D), context (B, T_kv, D). Output: (B, T_q, D).
# Steps:
# 1. Create W_q, W_k, W_v projection matrices (D x D)
# 2. Q = queries @ W_q, K = context @ W_k, V = context @ W_v
# 3. Scores = Q @ K^T / sqrt(d)  -> (B, T_q, T_kv) — NOT square!
# 4. Softmax over last dim (T_kv)
# 5. Output = weights @ V  -> (B, T_q, D)

# %%
import torch
import torch.nn as nn
import math

# %%
class MyCrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # TODO: create W_q (d_model, d_model)
        # TODO: create W_k (d_model, d_model)
        # TODO: create W_v (d_model, d_model)
        pass

    def forward(self, queries_input, context):
        # queries_input: (B, T_q, D) — the sequence that asks questions
        # context:       (B, T_kv, D) — the sequence that provides answers
        # TODO step 1: Q = queries_input projected  -> (B, T_q, D)
        # TODO step 2: K = context projected         -> (B, T_kv, D)
        # TODO step 3: V = context projected         -> (B, T_kv, D)
        # TODO step 4: scores = Q @ K^T / sqrt(d)   -> (B, T_q, T_kv)
        # TODO step 5: weights = softmax(scores)     -> (B, T_q, T_kv)
        # TODO step 6: output = weights @ V          -> (B, T_q, D)
        raise NotImplementedError

# %%
# TEST 1: basic shape check
B, T_q, T_kv, D = 2, 3, 220, 64
queries = torch.randn(B, T_q, D)
context = torch.randn(B, T_kv, D)
model = MyCrossAttention(d_model=D)
out = model(queries, context)
assert out.shape == (B, T_q, D), f'Wrong shape: {out.shape}, expected ({B}, {T_q}, {D})'
print('PASS: shape check')

# %%
# TEST 2: different sequence lengths
for tq, tkv in [(1, 100), (10, 10), (50, 5)]:
    q = torch.randn(1, tq, 64)
    c = torch.randn(1, tkv, 64)
    o = model(q, c)
    assert o.shape == (1, tq, 64), f'Wrong shape for T_q={tq}, T_kv={tkv}: {o.shape}'
print('PASS: variable sequence lengths')

# %%
# TEST 3: output depends on context
c1 = torch.randn(1, 10, 64)
c2 = torch.randn(1, 10, 64)
q = torch.randn(1, 3, 64)
model.eval()
with torch.no_grad():
    o1 = model(q, c1)
    o2 = model(q, c2)
assert not torch.allclose(o1, o2, atol=1e-3), 'Output should change when context changes'
print('PASS: context sensitivity')
