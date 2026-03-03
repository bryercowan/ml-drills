# Drill 08 — LayerNorm from Scratch

## Why This Matters

LayerNorm appears **twice** in every transformer block — once before self-attention,
once before the feed-forward network. Your Qwen3-VL backbone applies it dozens of times
per forward pass.

Unlike BatchNorm, LayerNorm normalizes across the **feature dimension** of each individual
token, so it works identically at any batch size and any sequence length. That's why
transformers use it universally.

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
# A small batch of 2 sequences, 3 tokens each, 4-dim features
x = torch.randn(2, 3, 4)
print("Input shape:", x.shape)
print(x)
```

## Intuition

For each token's feature vector (the last dimension):
1. Compute the **mean** and **variance** across that vector
2. Subtract mean, divide by std → zero mean, unit variance
3. Apply learned **scale** (γ) and **shift** (β) so the network can undo the normalization where needed

Each token is normalized **independently** — no dependence on other tokens or other samples.

## Building Block 1: Mean & Variance

```python
# Compute mean and variance along the LAST dimension (features)
# keepdim=True so we can broadcast during subtraction/division
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, unbiased=False, keepdim=True)  # LayerNorm uses biased variance (1/N not 1/(N-1))

print("mean shape:", mean.shape)   # (2, 3, 1) — one value per token
print("var  shape:", var.shape)
print()
print("First token mean:", mean[0, 0].item())
print("First token var: ", var[0, 0].item())
```

## Building Block 2: Normalize

```python
eps = 1e-5  # small constant for numerical stability

x_norm = (x - mean) / torch.sqrt(var + eps)

# Verify: each token should now have ~0 mean and ~1 variance
print("After normalization:")
print("Token [0,0] mean:", x_norm[0, 0].mean().item())
print("Token [0,0] std: ", x_norm[0, 0].std(unbiased=False).item())
```

## Building Block 3: Learnable Scale (γ) and Shift (β)

Why not stop at normalization? Because some features **should** be larger than others.
For example, one dimension might encode "is this a noun?" (binary-ish) while another
encodes position (could be large). Forcing unit variance everywhere would lose that.

γ and β are **per-feature** learnable parameters that let the network recover any
scale/shift it needs.

```python
d = x.shape[-1]  # feature dimension = 4

gamma = nn.Parameter(torch.ones(d))   # initialized to 1 (identity scale)
beta  = nn.Parameter(torch.zeros(d))  # initialized to 0 (no shift)

out = gamma * x_norm + beta
print("gamma:", gamma.data)
print("beta: ", beta.data)
print("Output (same as x_norm initially since gamma=1, beta=0):")
print(torch.allclose(out.data, x_norm))
```

## Building Block 4: LayerNorm vs BatchNorm

| | BatchNorm | LayerNorm |
|---|---|---|
| **Normalizes across** | Batch dimension (per feature) | Feature dimension (per sample) |
| **Depends on batch?** | Yes — needs batch statistics | No — each sample independent |
| **Train vs eval** | Different behavior (running stats) | Same behavior always |
| **Best for** | CNNs (spatial features) | Transformers (variable-length sequences) |

Key insight: BatchNorm computes stats **across samples** for each feature.
LayerNorm computes stats **across features** for each sample. That's the whole difference.

## Complete LayerNorm Implementation

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# Quick test
ln = LayerNorm(4)
out = ln(x)
print("Output shape:", out.shape)
print("Token [0,0] mean:", out[0, 0].mean().item())
```

## Your Turn

Implement `MyLayerNorm` below from memory. Don't peek above!

```python
class MyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # TODO: store eps, create gamma (ones) and beta (zeros) parameters
        pass

    def forward(self, x):
        # TODO: compute mean, var, normalize, then scale and shift
        pass
```

## Test: Compare Against `nn.LayerNorm`

```python
# Run this to check your implementation
torch.manual_seed(99)
test_x = torch.randn(4, 8, 64)  # batch=4, seq=8, dim=64

ref = nn.LayerNorm(64)
mine = MyLayerNorm(64)

# Copy weights so outputs should match exactly
mine.gamma.data = ref.weight.data.clone()
mine.beta.data = ref.bias.data.clone()

ref_out = ref(test_x)
my_out = mine(test_x)

if my_out is None:
    print("Your forward() returned None — fill in the TODO!")
else:
    match = torch.allclose(ref_out, my_out, atol=1e-5)
    if match:
        print("PASS — your LayerNorm matches nn.LayerNorm!")
    else:
        max_diff = (ref_out - my_out).abs().max().item()
        print(f"FAIL — max difference: {max_diff}")
```

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 8: Layer Normalization
Implement LayerNorm from scratch (no nn.LayerNorm).

Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta

- Compute mean and variance along the LAST dimension (keepdim=True)
- gamma (scale): learnable parameter initialized to ones, shape (d_model,)
- beta (shift): learnable parameter initialized to zeros, shape (d_model,)
- eps: small constant for numerical stability (default 1e-5)

```python
import torch
import torch.nn as nn
```

```python
class MyLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # TODO: create gamma as nn.Parameter of ones, shape (d_model,)
        # TODO: create beta as nn.Parameter of zeros, shape (d_model,)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> (..., d_model)"""
        # TODO: compute mean along last dim (keepdim=True)
        # TODO: compute variance along last dim (keepdim=True)
        # TODO: normalize: (x - mean) / sqrt(var + eps)
        # TODO: scale and shift: gamma * normalized + beta
        pass
```

```python
# TEST: compare against nn.LayerNorm
torch.manual_seed(42)
d = 64
my_ln = MyLayerNorm(d)
ref_ln = nn.LayerNorm(d)

# Copy parameters so both use same gamma/beta
with torch.no_grad():
    ref_ln.weight.copy_(my_ln.gamma)
    ref_ln.bias.copy_(my_ln.beta)

x = torch.randn(4, 10, d)
out = my_ln(x)
ref = ref_ln(x)

assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
assert torch.allclose(out, ref, atol=1e-5), f"Max diff: {(out - ref).abs().max().item()}"
print(f"Max difference: {(out - ref).abs().max().item():.2e}")
print("layer norm: ALL TESTS PASSED")
```
