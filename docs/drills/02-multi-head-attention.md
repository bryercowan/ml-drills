# Drill 2: Multi-Head Attention

## Why This Matters

In Drill 1 you built single-head self-attention: one set of Q, K, V projections
producing one attention pattern over the sequence.

But your EvoCUA-8B model has **32 attention heads**. When you LoRA-tune the
`q_proj`, `k_proj`, `v_proj`, and `o_proj` weight matrices, every one of those
32 heads is affected. Understanding what happens *inside* multi-head attention
will help you reason about what your LoRA adapters are actually learning.

**Goal:** Build `MultiHeadAttention` from scratch, reusing the scaled-dot-product
attention you already know.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```

## The Core Idea

Single-head attention computes **one** set of attention weights.  
Multi-head attention runs **H independent attention functions in parallel**,
each with its own learned projections, then concatenates the results.

Why bother?

| Single head (D=128)            | 4 heads (d_k=32 each)                    |
|--------------------------------|-------------------------------------------|
| One 128-dim attention pattern  | Four 32-dim patterns, each specialising   |
| Must encode everything at once | Head 1: syntax, Head 2: coreference, ...  |

Each head operates on a **slice** of the embedding dimension.  
Total compute is the same — we just split D into H pieces of size d_k = D // H.

Let's build it step by step.

## Building Block 1: The Reshape Trick

We project X into Q, K, V of shape `(B, T, D)` — same as Drill 1.  
Then we **reshape** to separate the head dimension:

```
(B, T, D)  →  view(B, T, H, d_k)  →  transpose(1, 2)  →  (B, H, T, d_k)
```

**Why transpose?** We want each head to be its own "mini-batch" so that the
matrix multiply `Q @ K^T` broadcasts cleanly over both the B and H dimensions.
After transpose, dims 0 and 1 are both batch-like, and dims 2, 3 are the
(sequence, features) that attention operates on.

```python
# --- The reshape trick, step by step ---

B, T, D = 2, 6, 128   # batch=2, seq_len=6, embed_dim=128
H = 4                  # number of heads
d_k = D // H           # per-head dimension = 32

x = torch.randn(B, T, D)

# Step 1: project (using a simple linear — no bias for clarity)
W_q = nn.Linear(D, D, bias=False)
Q = W_q(x)
print(f"After projection:  Q.shape = {Q.shape}")        # (2, 6, 128)

# Step 2: view to expose the head dimension
Q = Q.view(B, T, H, d_k)
print(f"After view:         Q.shape = {Q.shape}")        # (2, 6, 4, 32)

# Step 3: transpose so H comes before T
Q = Q.transpose(1, 2)
print(f"After transpose:    Q.shape = {Q.shape}")        # (2, 4, 6, 32)

# Now Q[b, h, :, :] is head h's query matrix for batch element b.
# This is exactly a (T, d_k) matrix — same layout as single-head attention!
```

## Building Block 2: Per-Head Scaled Dot-Product Attention

Once Q, K, V all have shape `(B, H, T, d_k)`, the same formula from Drill 1
works **unchanged**:

```
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)   # (B, H, T, T)
weights = softmax(scores, dim=-1)
context = weights @ V                            # (B, H, T, d_k)
```

PyTorch broadcasts the matmul over the B and H dimensions automatically.
Each head gets its own T×T attention matrix — exactly what we wanted.

```python
# --- Per-head attention (all heads at once) ---

B, T, D, H = 2, 6, 128, 4
d_k = D // H

# Simulate Q, K, V already reshaped to (B, H, T, d_k)
Q = torch.randn(B, H, T, d_k)
K = torch.randn(B, H, T, d_k)
V = torch.randn(B, H, T, d_k)

# Scaled dot-product attention — identical to Drill 1!
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, H, T, T)
print(f"scores.shape  = {scores.shape}")               # (2, 4, 6, 6)

weights = F.softmax(scores, dim=-1)
print(f"weights.shape = {weights.shape}")               # (2, 4, 6, 6)

context = weights @ V                                   # (B, H, T, d_k)
print(f"context.shape = {context.shape}")               # (2, 4, 6, 32)

# Verify: each head has its own attention pattern
print(f"\nHead 0 attends differently from Head 1:")
print(f"  weights[0,0,0,:3] = {weights[0,0,0,:3].tolist()}")
print(f"  weights[0,1,0,:3] = {weights[0,1,0,:3].tolist()}")
```

## Building Block 3: Recombine Heads + Output Projection

After per-head attention we have `(B, H, T, d_k)`.
We need to get back to `(B, T, D)` so the rest of the transformer sees
a normal embedding.

```
(B, H, T, d_k) → transpose(1,2) → (B, T, H, d_k) → contiguous() → view(B, T, D)
```

**Why `contiguous()`?** `transpose` doesn't move data in memory — it just
changes stride metadata. `view` requires contiguous memory, so we call
`contiguous()` first (or use `reshape`, which does it implicitly).

Finally, we multiply by **W_o** `(D, D)` — the output projection. This is the
`o_proj` that your LoRA adapter also tunes. It lets the model learn how to
**mix information across heads**.

```python
# --- Recombine heads ---

B, T, D, H = 2, 6, 128, 4
d_k = D // H

# Simulated per-head output
context = torch.randn(B, H, T, d_k)
print(f"Per-head context:   {context.shape}")         # (2, 4, 6, 32)

# Step 1: transpose H and T back
context = context.transpose(1, 2)
print(f"After transpose:    {context.shape}")         # (2, 6, 4, 32)

# Step 2: contiguous + view to merge heads back into D
context = context.contiguous().view(B, T, D)
print(f"After view:         {context.shape}")         # (2, 6, 128)

# Step 3: output projection W_o
W_o = nn.Linear(D, D, bias=False)
output = W_o(context)
print(f"After W_o:          {output.shape}")          # (2, 6, 128)
```

## Putting It All Together: MultiHeadAttention

Here's the complete module with all three building blocks combined.
Read through and match each section to the blocks above.

```python
class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads       # per-head dimension

        # --- Projections (these are what LoRA adapts) ---
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # output projection

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, H, T, d_k)"""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) → (B, T, D)"""
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Project into Q, K, V  — shape (B, T, D) each
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into heads       — shape (B, H, T, d_k) each
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # 3. Scaled dot-product attention (per head, broadcasted)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)                       # (B, H, T, T)
        context = weights @ V                                     # (B, H, T, d_k)

        # 4. Merge heads and apply output projection
        output = self._merge_heads(context)                       # (B, T, D)
        output = self.W_o(output)                                 # (B, T, D)

        return output


# --- Quick test ---
B, T, D, H = 2, 10, 128, 4
mha = MultiHeadAttention(d_model=D, n_heads=H)
x = torch.randn(B, T, D)
out = mha(x)
print(f"Input:  {x.shape}")    # (2, 10, 128)
print(f"Output: {out.shape}")  # (2, 10, 128)
assert out.shape == x.shape, "Shape mismatch!"
print("Shape check passed.")
```

## Connecting to EvoCUA-8B

Your model uses **d_model=4096** and **n_heads=32**, giving **d_k=128** per head.

When you set up LoRA with `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`,
you're injecting low-rank adapters into exactly the four `nn.Linear` layers above
(`W_q`, `W_k`, `W_v`, `W_o`). Each LoRA adapter is shared across all 32 heads —
its update gets split into 32 slices by the reshape trick.

That's why a rank-64 LoRA on `q_proj` can still meaningfully change all 32 heads:
each head only "sees" a rank-2 perturbation (64 / 32), but that's enough to shift
attention patterns during fine-tuning.

## Your Turn

Implement `MultiHeadAttention` from scratch using the scaffolding below.
Don't peek at the solution above — use the building blocks you practiced.

```python
class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO 1: store d_model, n_heads, compute d_k
        # TODO 2: create W_q, W_k, W_v, W_o as nn.Linear(d_model, d_model)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # TODO 3: project x into Q, K, V  (each: B, T, D)

        # TODO 4: reshape to (B, H, T, d_k) using view + transpose

        # TODO 5: compute scores = Q @ K^T / sqrt(d_k)

        # TODO 6: softmax over the last dimension

        # TODO 7: multiply weights @ V to get context (B, H, T, d_k)

        # TODO 8: merge heads back: transpose, contiguous, view → (B, T, D)

        # TODO 9: apply output projection W_o

        return output
```

## Test Your Implementation

Run this cell after filling in `MyMultiHeadAttention`.
It checks that the output shape matches the input shape.

```python
# --- Test your implementation ---
B, T, D, H = 2, 10, 128, 4

my_mha = MyMultiHeadAttention(d_model=D, n_heads=H)
x = torch.randn(B, T, D)
out = my_mha(x)

assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print("All checks passed!")
```

## Bonus: Causal (Autoregressive) Mask

Decoder-only models like EvoCUA-8B use a **causal mask** so that each token
can only attend to itself and earlier tokens. Here's how to create one and
pass it to the attention module.

```python
# --- Causal mask ---
T = 6
causal_mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
print("Causal mask (1 = attend, 0 = block):")
print(causal_mask[0, 0])

# Use it with our MHA
B, D, H = 1, 128, 4
mha = MultiHeadAttention(d_model=D, n_heads=H)
x = torch.randn(B, T, D)
out = mha(x, mask=causal_mask)
print(f"\nCausal MHA output shape: {out.shape}")
```

## What's Next

In **Drill 3** we'll add positional encoding and layer normalization,
then stack attention + feed-forward into a full transformer block.
That's the repeating unit your EvoCUA-8B stacks 32 times.

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 2: Multi-Head Attention
Implement multi-head self-attention from scratch.
Input: (B, T, D). Output: (B, T, D).
Steps:
1. Create W_q, W_k, W_v, W_o projection matrices (D x D)
2. Project input to Q, K, V
3. Reshape (B, T, D) -> (B, H, T, d_k) where d_k = D // H
4. Attention scores per head: Q @ K^T / sqrt(d_k)
5. Optional causal mask, then softmax
6. Weighted sum: weights @ V -> (B, H, T, d_k)
7. Recombine heads: (B, H, T, d_k) -> (B, T, D)
8. Output projection: result @ W_o

```python
import torch
import torch.nn as nn
import math
```

```python
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
```

```python
# TEST 1: shape check
x = torch.randn(2, 5, 64)
model = MyMultiHeadAttention(d_model=64, num_heads=8)
out = model(x)
assert out.shape == (2, 5, 64), f'Wrong shape: {out.shape}'
print('PASS: shape check (8 heads)')
```

```python
# TEST 2: different head counts
for h in [1, 2, 4, 8, 16]:
    m = MyMultiHeadAttention(d_model=64, num_heads=h)
    o = m(x)
    assert o.shape == (2, 5, 64), f'Wrong shape with {h} heads: {o.shape}'
print('PASS: all head counts')
```

```python
# TEST 3: causal mode shape check
model_c = MyMultiHeadAttention(d_model=64, num_heads=8, causal=True)
out_c = model_c(x)
assert out_c.shape == (2, 5, 64), f'Wrong shape: {out_c.shape}'
print('PASS: causal shape check')
```
