# Drill 04 — Cross-Attention

## Why This Matters

You already built this. In your CNN policy, **action tokens** (Q) attend to
**ConvNeXt spatial features** (K, V). That's cross-attention — Q and K/V come
from different sources.

Where it shows up:
- **Your policy**: 3 action tokens query 220 spatial tokens (11×20 ConvNeXt grid)
- **Encoder-decoder transformers**: decoder queries attend to encoder output
- **Stable Diffusion**: noised image tokens query CLIP text embeddings

You already know self-attention from Drill 01. Cross-attention is the same math
with one structural twist.

## The One Key Difference

In **self-attention**, Q, K, V all come from the same input → score matrix is
square: `(B, T, T)`.

In **cross-attention**, Q comes from one source, K/V from another → score matrix
is **rectangular**: `(B, T_q, T_kv)` where `T_q ≠ T_kv`.

Your real dimensions:
```
queries_input: (B, 3, D)     ← 3 action tokens (what you're predicting)
context:       (B, 220, D)   ← 220 spatial positions from ConvNeXt (11×20 grid)
scores:        (B, 3, 220)   ← each action token scores ALL spatial positions
output:        (B, 3, D)     ← output matches query length, NOT context length
```

That's it. Same `softmax(QK^T / √d) V` — just Q and K/V have different sources.

## Building Block 1: Two Separate Inputs

Self-attention has one input. Cross-attention has two.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

B, D = 2, 64  # batch size, model dim

# Source 1: what's asking the questions (action tokens)
T_q = 3
queries_input = torch.randn(B, T_q, D)

# Source 2: what's being attended to (spatial features from ConvNeXt)
T_kv = 220
context = torch.randn(B, T_kv, D)

print(f"queries_input: {queries_input.shape}")  # (2, 3, 64)
print(f"context:       {context.shape}")         # (2, 220, 64)
print(f"\nThese are DIFFERENT tensors with DIFFERENT sequence lengths.")
```

## Building Block 2: Q from Queries, K/V from Context

This is the asymmetry. Three projections, but they take different inputs.

```python
d_head = 32

W_q = nn.Linear(D, d_head, bias=False)
W_k = nn.Linear(D, d_head, bias=False)
W_v = nn.Linear(D, d_head, bias=False)

# Q comes from queries_input
Q = W_q(queries_input)   # (B, 3, d_head)

# K and V come from context
K = W_k(context)          # (B, 220, d_head)
V = W_v(context)          # (B, 220, d_head)

print(f"Q: {Q.shape}  ← from queries_input (3 tokens)")
print(f"K: {K.shape}  ← from context (220 tokens)")
print(f"V: {V.shape}  ← from context (220 tokens)")
print(f"\nQ has {T_q} rows, K/V have {T_kv} rows. No requirement they match.")
```

## Building Block 3: Non-Square Score Matrix

`QK^T` produces a **(B, T_q, T_kv)** matrix. Each row is one query token's
attention distribution over ALL context positions.

```python
# Q: (B, 3, d_head) @ K^T: (B, d_head, 220) → (B, 3, 220)
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
print(f"scores: {scores.shape}")  # (B, 3, 220) — NOT square!

# Each of the 3 action tokens has 220 attention weights
weights = F.softmax(scores, dim=-1)
print(f"weights: {weights.shape}")

# Verify: each query's weights sum to 1 across the 220 context positions
print(f"\nRow sums (should be 1.0): {weights[0].sum(dim=-1)}")
```

## Building Block 4: Output Shape Matches Queries

The output has **T_q** tokens (3), not T_kv (220). Each output token is a
weighted combination of context values, with weights determined by that query.

```python
# weights: (B, 3, 220) @ V: (B, 220, d_head) → (B, 3, d_head)
output = torch.matmul(weights, V)
print(f"output: {output.shape}")  # (B, 3, 32) — matches query length!
print(f"\nStarted with {T_q} query tokens, got {output.shape[1]} output tokens.")
print(f"Each output is a weighted mix of {T_kv} context vectors.")
```

## Complete CrossAttention Module

Putting it together with multi-head support and output projection.

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q from queries, K/V from context — the defining asymmetry
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, queries_input, context):
        """
        queries_input: (B, T_q, D)   — e.g. action tokens
        context:       (B, T_kv, D)  — e.g. ConvNeXt spatial features
        returns:       (B, T_q, D)   — output matches query length
        """
        B, T_q, _ = queries_input.shape
        T_kv = context.shape[1]

        # Project: Q from queries, K/V from context
        Q = self.W_q(queries_input)  # (B, T_q, D)
        K = self.W_k(context)        # (B, T_kv, D)
        V = self.W_v(context)        # (B, T_kv, D)

        # Reshape to multi-head: (B, n_heads, T, d_head)
        Q = Q.view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)

        # Attention: (B, n_heads, T_q, d_head) @ (B, n_heads, d_head, T_kv)
        #          → (B, n_heads, T_q, T_kv)  ← rectangular!
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        weights = F.softmax(scores, dim=-1)

        # (B, n_heads, T_q, T_kv) @ (B, n_heads, T_kv, d_head)
        # → (B, n_heads, T_q, d_head)
        out = torch.matmul(weights, V)

        # Recombine heads: (B, T_q, D)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.W_o(out)
```

## Your Turn

Implement `CrossAttention` from scratch below. No peeking above.

Checklist:
- [ ] Q projected from `queries_input`, K/V projected from `context`
- [ ] Score matrix shape is `(B, n_heads, T_q, T_kv)` — rectangular
- [ ] Output shape is `(B, T_q, D)` — matches queries, not context

```python
class MyCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, queries_input, context):
        # YOUR CODE HERE
        pass
```

## Test

```python
# Test with your real dimensions
B, D, n_heads = 2, 64, 4
T_q, T_kv = 3, 220  # 3 action tokens, 220 spatial positions (11x20)

queries_input = torch.randn(B, T_q, D)
context = torch.randn(B, T_kv, D)

# Test reference implementation
ref = CrossAttention(D, n_heads)
ref_out = ref(queries_input, context)
assert ref_out.shape == (B, T_q, D), f"Expected {(B, T_q, D)}, got {ref_out.shape}"

# Test your implementation
my = MyCrossAttention(D, n_heads)
my_out = my(queries_input, context)
assert my_out.shape == (B, T_q, D), f"Expected {(B, T_q, D)}, got {my_out.shape}"

print(f"Input shapes:  queries={queries_input.shape}, context={context.shape}")
print(f"Output shape:  {my_out.shape}")
print(f"\n3 action tokens queried 220 spatial positions → 3 output tokens.")
print("All assertions passed.")
```

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 4: Cross-Attention
Implement cross-attention from scratch.
Queries come from one sequence, keys/values from another.
Input: queries (B, T_q, D), context (B, T_kv, D). Output: (B, T_q, D).
Steps:
1. Create W_q, W_k, W_v projection matrices (D x D)
2. Q = queries @ W_q, K = context @ W_k, V = context @ W_v
3. Scores = Q @ K^T / sqrt(d)  -> (B, T_q, T_kv) — NOT square!
4. Softmax over last dim (T_kv)
5. Output = weights @ V  -> (B, T_q, D)

```python
import torch
import torch.nn as nn
import math
```

```python
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
```

```python
# TEST 1: basic shape check
B, T_q, T_kv, D = 2, 3, 220, 64
queries = torch.randn(B, T_q, D)
context = torch.randn(B, T_kv, D)
model = MyCrossAttention(d_model=D)
out = model(queries, context)
assert out.shape == (B, T_q, D), f'Wrong shape: {out.shape}, expected ({B}, {T_q}, {D})'
print('PASS: shape check')
```

```python
# TEST 2: different sequence lengths
for tq, tkv in [(1, 100), (10, 10), (50, 5)]:
    q = torch.randn(1, tq, 64)
    c = torch.randn(1, tkv, 64)
    o = model(q, c)
    assert o.shape == (1, tq, 64), f'Wrong shape for T_q={tq}, T_kv={tkv}: {o.shape}'
print('PASS: variable sequence lengths')
```

```python
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
```
