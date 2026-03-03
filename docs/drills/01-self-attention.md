# Self-Attention from Scratch

## Why This Matters

You already fine-tuned attention layers with LoRA on your VLA for GUI automation.
You know *where* attention lives in the architecture and *how* to adapt it.
Now you're going to know *what it actually computes* — line by line, tensor by tensor.

By the end of this notebook you will:
- Build scaled dot-product self-attention from raw matrix multiplies
- Understand every shape: `(B, T, D)` → `(B, T, T)` → `(B, T, D)`
- Add a causal mask so token `i` can only attend to tokens `≤ i`
- Write the whole thing yourself from a blank class signature

## The Intuition

Self-attention answers one question per token: **"Which other tokens should I pay attention to?"**

Each token gets three roles:
- **Query (Q)** — "I'm looking for…"
- **Key (K)** — "Here's what I contain…"
- **Value (V)** — "Here's what I'll give you if you pick me."

The flow:

```
  x  ──┬──  @ W_q  ──▶  Q  ─────────────┐
        │                                  ▼
        ├──  @ W_k  ──▶  K  ──▶  K^T ──▶ Q @ K^T  ──▶  scale  ──▶  softmax  ──▶  weights
        │                                                                            │
        └──  @ W_v  ──▶  V  ─────────────────────────────────────────────────────▶  weights @ V  ──▶  output
```

Shapes at each step (batch size B, sequence length T, model dim D):

| Tensor       | Shape       |
|-------------|-------------|
| x           | (B, T, D)   |
| Q, K, V     | (B, T, D)   |
| Q @ K^T     | (B, T, T)   |
| weights     | (B, T, T)   |
| weights @ V | (B, T, D)   |

That T×T matrix is the core insight — it's a token-to-token relevance map.

```python
import torch
import math

torch.manual_seed(42)

# Dimensions we'll use throughout
B = 2   # batch size
T = 4   # sequence length (number of tokens)
D = 8   # embedding / model dimension

# Random "token embeddings" — pretend these came from an embedding layer
x = torch.randn(B, T, D)
print(f"x shape: {x.shape}")  # (B, T, D) = (2, 4, 8)
```

## Building Block 1: Q, K, V Projections

Each projection is just a learned linear transform: `x @ W + b`.
We map from D → D (same size in, same size out — for single-head attention).

```python
# Weight matrices — in practice these are nn.Linear layers
W_q = torch.randn(D, D)  # (D, D) — query projection
W_k = torch.randn(D, D)  # (D, D) — key projection
W_v = torch.randn(D, D)  # (D, D) — value projection

# Project: (B, T, D) @ (D, D) = (B, T, D)
Q = x @ W_q  # each token gets a query vector
K = x @ W_k  # each token gets a key vector
V = x @ W_v  # each token gets a value vector

print(f"Q shape: {Q.shape}")  # (2, 4, 8)
print(f"K shape: {K.shape}")  # (2, 4, 8)
print(f"V shape: {V.shape}")  # (2, 4, 8)
print()
print("Q[0, 0] (query vector for batch 0, token 0):")
print(Q[0, 0])
```

## Building Block 2: Attention Scores

`Q @ K^T` gives us a score for every (query-token, key-token) pair.

Entry `[i, j]` in the resulting T×T matrix means:
**"How relevant is token j's content (key) to what token i is looking for (query)?"**

```python
# (B, T, D) @ (B, D, T) = (B, T, T)
# K.transpose(-2, -1) swaps the last two dims: (B, T, D) -> (B, D, T)
scores = Q @ K.transpose(-2, -1)

print(f"scores shape: {scores.shape}")  # (2, 4, 4) — T×T matrix per batch
print()
print("scores[0] — token-to-token relevance for batch 0:")
print(scores[0])
print()
print("scores[0, 0, :] — how much token 0 cares about tokens [0,1,2,3]:")
print(scores[0, 0, :])
```

## Building Block 3: Scaling by √d_k

Problem: dot products grow with dimension D. Large values push softmax into
regions where gradients vanish (everything near 0 or 1).

Fix: divide by √D to keep variance ≈ 1 regardless of dimension.

This is why it's called **scaled** dot-product attention.

```python
d_k = D  # dimension of key vectors

# Show the problem: compare magnitudes before and after scaling
print(f"Before scaling — max |score|: {scores.abs().max().item():.2f}")
print(f"sqrt(d_k) = sqrt({d_k}) = {math.sqrt(d_k):.2f}")
print()

scaled_scores = scores / math.sqrt(d_k)  # (B, T, T) — same shape, smaller values

print(f"After scaling  — max |score|: {scaled_scores.abs().max().item():.2f}")
print()
print("Scaled scores[0]:")
print(scaled_scores[0])
```

## Building Block 4: Softmax → Attention Weights

Softmax along the last dimension (the key dimension) converts raw scores into
a probability distribution. Each row sums to 1.

Row `i` tells us: **"How should token i distribute its attention across all tokens?"**

```python
# softmax over last dim: each ROW of the T×T matrix sums to 1
weights = torch.softmax(scaled_scores, dim=-1)  # (B, T, T)

print(f"weights shape: {weights.shape}")  # (2, 4, 4)
print()
print("weights[0] — attention weights for batch 0:")
print(weights[0])
print()
print("Row sums (should all be 1.0):")
print(weights[0].sum(dim=-1))
```

## Building Block 5: Weighted Sum → Output

Multiply the attention weights by V. Each output token is a weighted average
of all value vectors, where the weights come from Q·K similarity.

This is the actual information mixing — token i's output is a blend of
every other token's value, weighted by relevance.

```python
# (B, T, T) @ (B, T, D) = (B, T, D)
output = weights @ V

print(f"output shape: {output.shape}")  # (2, 4, 8) — same shape as input x
print()
print("output[0, 0] — token 0's new representation (weighted mix of all V's):")
print(output[0, 0])
```

## Building Block 6: Causal Mask

For autoregressive models (GPT-style), token `i` must NOT attend to future tokens `j > i`.

We create a lower-triangular mask and set the upper-triangle scores to `-inf`
**before** softmax. `softmax(-inf) = 0`, so future tokens get zero weight.

```
mask (T=4):          after masked_fill:
  1 0 0 0              score  -inf  -inf  -inf
  1 1 0 0              score  score -inf  -inf
  1 1 1 0              score  score score -inf
  1 1 1 1              score  score score score
```

```python
# Lower-triangular mask: True where we ALLOW attention
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # (T, T)
print("Causal mask (True = allowed):")
print(mask)
print()

# Start from raw scaled scores again
print("Scaled scores BEFORE masking (batch 0):")
print(scaled_scores[0].detach())
print()

# Set positions where mask is False (upper triangle) to -inf
masked_scores = scaled_scores.masked_fill(~mask, float('-inf'))  # (B, T, T)
print("Scaled scores AFTER masking (batch 0):")
print(masked_scores[0].detach())
print()

# Now softmax — -inf positions become 0
causal_weights = torch.softmax(masked_scores, dim=-1)  # (B, T, T)
print("Causal attention weights (batch 0):")
print(causal_weights[0])
print()
print("Row sums (still 1.0):")
print(causal_weights[0].sum(dim=-1))
print()

# Final output with causal masking
causal_output = causal_weights @ V  # (B, T, D)
print(f"Causal output shape: {causal_output.shape}")  # (2, 4, 8)
```

## Putting It Together: Complete SelfAttention

Here's the full module. Every line is commented with the tensor shape.

```python
import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention with optional causal mask."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model                          # D

        self.W_q = nn.Linear(d_model, d_model)          # (D) -> (D)
        self.W_k = nn.Linear(d_model, d_model)          # (D) -> (D)
        self.W_v = nn.Linear(d_model, d_model)          # (D) -> (D)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, T, D = x.shape                               # unpack dimensions

        Q = self.W_q(x)                                  # (B, T, D)
        K = self.W_k(x)                                  # (B, T, D)
        V = self.W_v(x)                                  # (B, T, D)

        scores = Q @ K.transpose(-2, -1)                 # (B, T, T)
        scores = scores / math.sqrt(D)                   # (B, T, T) — scale

        if causal:
            mask = torch.tril(torch.ones(T, T,           # (T, T) lower triangle
                              dtype=torch.bool,
                              device=x.device))
            scores = scores.masked_fill(~mask, float('-inf'))  # mask future

        weights = torch.softmax(scores, dim=-1)          # (B, T, T) — row sums = 1
        output = weights @ V                             # (B, T, D) — weighted mix

        return output                                    # (B, T, D) same shape as x


# Quick test
attn = SelfAttention(d_model=D)
y = attn(x, causal=True)
print(f"Input shape:  {x.shape}")   # (2, 4, 8)
print(f"Output shape: {y.shape}")   # (2, 4, 8)
print("Passed basic shape check.")
```

---

## Your Turn

Rewrite `SelfAttention` from scratch. No peeking above.

The docstring lists every step. Fill in the body, then run the test cell below.

```python
class MySelfAttention(nn.Module):
    """Write single-head self-attention from scratch.

    __init__:
        1. Store d_model
        2. Create W_q, W_k, W_v as nn.Linear(d_model, d_model)

    forward(x, causal=False):
        1. Unpack B, T, D from x.shape
        2. Compute Q, K, V via the linear layers          — each (B, T, D)
        3. Compute attention scores: Q @ K^T               — (B, T, T)
        4. Scale by sqrt(D)                                — (B, T, T)
        5. If causal: create tril mask, masked_fill -inf   — (B, T, T)
        6. Softmax over last dim                           — (B, T, T)
        7. Weighted sum: weights @ V                       — (B, T, D)
        8. Return output                                   — (B, T, D)
    """

    def __init__(self, d_model: int):
        super().__init__()
        raise NotImplementedError("Your code here")

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        raise NotImplementedError("Your code here")
```

### Tests

Run this cell to verify your implementation. Both checks must print PASSED.

```python
def test_my_self_attention():
    torch.manual_seed(123)
    B, T, D = 2, 6, 16
    x_test = torch.randn(B, T, D)

    model = MySelfAttention(d_model=D)

    # ── Test 1: output shape ──
    out = model(x_test, causal=False)
    assert out.shape == (B, T, D), f"Shape mismatch: expected {(B, T, D)}, got {out.shape}"
    print("TEST 1 PASSED — output shape is correct.")

    # ── Test 2: causal masking ──
    out_causal = model(x_test, causal=True)
    assert out_causal.shape == (B, T, D), f"Shape mismatch with causal=True"

    # Verify causality: changing a future token should NOT affect earlier outputs
    x_modified = x_test.clone()
    x_modified[:, -1, :] = torch.randn(B, D)  # change last token
    out_modified = model(x_modified, causal=True)

    # Token 0's output should be identical (it can't see the last token)
    diff = (out_causal[:, 0, :] - out_modified[:, 0, :]).abs().max().item()
    assert diff < 1e-5, f"Causal mask broken: token 0 changed by {diff} when future token modified"
    print("TEST 2 PASSED — causal mask correctly blocks future tokens.")
    print()
    print("All tests passed. You can write self-attention from scratch now.")

test_my_self_attention()
```

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 1: Self-Attention
Implement single-head self-attention from scratch.
Input: (B, T, D). Output: (B, T, D).
Steps:
1. Create W_q, W_k, W_v projection matrices (D x D)
2. Project input to Q, K, V
3. Compute attention scores: Q @ K^T / sqrt(d)
4. If causal, apply upper-triangular mask (-inf)
5. Softmax over last dim
6. Weighted sum: weights @ V

```python
import torch
import torch.nn as nn
import math
```

```python
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
```

```python
# TEST 1: shape check
x = torch.randn(2, 5, 64)
model = MySelfAttention(64, causal=False)
out = model(x)
assert out.shape == (2, 5, 64), f'Wrong shape: {out.shape}'
print('PASS: shape check (non-causal)')
```

```python
# TEST 2: shape check causal
model_c = MySelfAttention(64, causal=True)
out_c = model_c(x)
assert out_c.shape == (2, 5, 64), f'Wrong shape: {out_c.shape}'
print('PASS: shape check (causal)')
```

```python
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
```
