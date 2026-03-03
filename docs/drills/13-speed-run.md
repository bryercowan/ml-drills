# Speed Run — All Implementations From Scratch

Open this notebook. Set a **60-minute timer**. Implement everything without looking at the other notebooks.

This is your interview simulation. No hints. No building blocks. Just you and the spec.

**Rules:**
1. No peeking at other notebooks or external references
2. Each function must pass its assertions
3. Track your time per drill — know where you're slow

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```

## 1. Self-Attention

**Input:** `Q, K, V` each `(batch, seq_len, d_k)`, optional `mask` (boolean, True = ignore)
**Output:** `(batch, seq_len, d_k)` — attended values
**Key:** Scale by `sqrt(d_k)`. If masked, set those scores to `-inf` before softmax.

```python
def self_attention(Q, K, V, mask=None):
    # YOUR CODE HERE
    pass


# --- Tests ---
B, T, D = 2, 5, 16
Q = K = V = torch.randn(B, T, D)
out = self_attention(Q, K, V)
assert out.shape == (B, T, D)

# Causal mask test
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
out_masked = self_attention(Q, K, V, mask=mask)
assert out_masked.shape == (B, T, D)
print("1. Self-Attention passed")
```

## 2. Multi-Head Attention

**Input:** `x` `(batch, seq_len, d_model)`, `n_heads` int
**Output:** `(batch, seq_len, d_model)`
**Key:** Project to Q/K/V, split into heads, attend per head, concat, project out. Use `nn.Linear` for projections.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x, mask=None):
        # YOUR CODE HERE
        pass


# --- Tests ---
mha = MultiHeadAttention(64, 4)
x = torch.randn(2, 10, 64)
out = mha(x)
assert out.shape == (2, 10, 64)
print("2. Multi-Head Attention passed")
```

## 3. Cross-Attention

**Input:** `x_q` `(batch, seq_q, d_model)`, `x_kv` `(batch, seq_kv, d_model)`, `n_heads` int
**Output:** `(batch, seq_q, d_model)`
**Key:** Q comes from `x_q`, K and V come from `x_kv`. Otherwise identical to multi-head attention.

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x_q, x_kv, mask=None):
        # YOUR CODE HERE
        pass


# --- Tests ---
ca = CrossAttention(64, 4)
x_q = torch.randn(2, 8, 64)
x_kv = torch.randn(2, 12, 64)
out = ca(x_q, x_kv)
assert out.shape == (2, 8, 64)
print("3. Cross-Attention passed")
```

## 4. LoRA Linear Layer

**Input:** `x` `(batch, in_features)`, rank `r`, frozen base weight
**Output:** `(batch, out_features)`
**Key:** `y = W_frozen @ x + (B @ A) @ x`. A is `(r, in)` init N(0, σ²), B is `(out, r)` init zeros. Freeze `W`.

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass


# --- Tests ---
lora = LoRALinear(64, 32, r=4)
x = torch.randn(2, 64)
out = lora(x)
assert out.shape == (2, 32)
# At init, LoRA contribution should be zero (B is zeros)
frozen_out = lora.weight @ x.T  # (32, 2)
assert torch.allclose(out.T, frozen_out, atol=1e-5), "LoRA should be zero-contribution at init"
print("4. LoRA Linear passed")
```

## 5. Numerically Stable Softmax

**Input:** `x` arbitrary shape tensor, `dim` int
**Output:** same shape, sums to 1 along `dim`
**Key:** Subtract `x.max(dim, keepdim=True)` before exp. No `torch.softmax`.

```python
def softmax(x, dim=-1):
    # YOUR CODE HERE
    pass


# --- Tests ---
x = torch.randn(2, 5)
out = softmax(x, dim=-1)
assert out.shape == x.shape
assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)
# Stability test: large values shouldn't produce nan/inf
x_big = torch.tensor([[1000.0, 1001.0, 1002.0]])
out_big = softmax(x_big, dim=-1)
assert not torch.isnan(out_big).any() and not torch.isinf(out_big).any()
print("5. Softmax passed")
```

## 6. Cross-Entropy Loss

**Input:** `logits` `(batch, num_classes)`, `targets` `(batch,)` integer class indices
**Output:** scalar loss
**Key:** `loss = -log(softmax(logits))[target]`, averaged over batch. Use log-sum-exp trick.

```python
def cross_entropy(logits, targets):
    # YOUR CODE HERE
    pass


# --- Tests ---
logits = torch.randn(4, 10)
targets = torch.tensor([3, 7, 0, 2])
loss = cross_entropy(logits, targets)
expected = F.cross_entropy(logits, targets)
assert loss.shape == ()
assert torch.allclose(loss, expected, atol=1e-5), f"Got {loss.item()}, expected {expected.item()}"
print("6. Cross-Entropy passed")
```

## 7. Layer Normalization

**Input:** `x` `(batch, ..., d)`, `d` is normalized dimension (last dim)
**Output:** same shape, normalized to zero mean / unit variance along last dim, then scale+shift
**Key:** `y = gamma * (x - mean) / sqrt(var + eps) + beta`. Learnable `gamma` (init 1) and `beta` (init 0).

```python
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass


# --- Tests ---
ln = LayerNorm(64)
x = torch.randn(2, 10, 64)
out = ln(x)
assert out.shape == x.shape
mean = out.mean(dim=-1)
var = out.var(dim=-1, unbiased=False)
assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
assert torch.allclose(var, torch.ones_like(var), atol=1e-2)
print("7. Layer Norm passed")
```

## 8. Transformer Block (Pre-Norm)

**Input:** `x` `(batch, seq_len, d_model)`, `n_heads` int, `d_ff` int (feedforward hidden dim)
**Output:** `(batch, seq_len, d_model)`
**Key:** Pre-norm style: `x = x + MHA(LN(x))`, then `x = x + FFN(LN(x))`. FFN = Linear→ReLU→Linear.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x, mask=None):
        # YOUR CODE HERE
        pass


# --- Tests ---
block = TransformerBlock(64, 4, 256)
x = torch.randn(2, 10, 64)
out = block(x)
assert out.shape == (2, 10, 64)
print("8. Transformer Block passed")
```

## 9. Training Loop

**Spec:** Write a training function that takes a model, dataloader, optimizer, loss_fn, device, and num_epochs. Returns list of average losses per epoch.
**Requirements:** Move batches to device. Zero grads. Forward. Loss. Backward. Step. Track running loss.
**Signature:** `train(model, dataloader, optimizer, loss_fn, device, num_epochs) -> list[float]`

```python
def train(model, dataloader, optimizer, loss_fn, device, num_epochs):
    # YOUR CODE HERE
    pass


# --- Tests ---
from torch.utils.data import DataLoader, TensorDataset
model = nn.Linear(8, 2)
dataset = TensorDataset(torch.randn(100, 8), torch.randint(0, 2, (100,)))
loader = DataLoader(dataset, batch_size=16)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = train(model, loader, optimizer, F.cross_entropy, 'cpu', 3)
assert len(losses) == 3
assert all(isinstance(l, float) for l in losses)
print("9. Training Loop passed")
```

---

**If you finished all 9 in under 60 minutes, you're ready.**

Go back and redo any that took more than 8 minutes. Those are your weak spots.

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 13: Speed Run — 60 minutes, all implementations, no peeking
Start a timer. Implement all 9 from scratch.
No looking at previous drills. No references. Just you and PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

start_time = time.time()

# ---------------------------------------------------------------------------
# 1/9  SELF-ATTENTION
# ---------------------------------------------------------------------------
```

## 1. Self-Attention
Input: (B, T, D) -> Output: (B, T, D)
Must apply causal mask (no attending to future positions)

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # TODO: Q, K, V projections and output projection
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D) with causal masking."""
        raise NotImplementedError
```

```python
# TEST 1: Self-Attention
B, T, D = 2, 8, 32
sa = SelfAttention(D)
x = torch.randn(B, T, D)
out = sa(x)
assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
print("1/9 Self-Attention PASSED")

# ---------------------------------------------------------------------------
# 2/9  MULTI-HEAD ATTENTION
# ---------------------------------------------------------------------------
```

## 2. Multi-Head Attention
Input: (B, T, D), H heads -> Output: (B, T, D)
Split D into H heads, each head_dim = D // H

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: Q, K, V projections, output projection
        # TODO: store n_heads, head_dim
        raise NotImplementedError

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D) with optional causal mask."""
        raise NotImplementedError
```

```python
# TEST 2: Multi-Head Attention
B, T, D, H = 2, 8, 32, 4
mha = MultiHeadAttention(D, H)
x = torch.randn(B, T, D)
out = mha(x)
assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
print("2/9 Multi-Head Attention PASSED")

# ---------------------------------------------------------------------------
# 3/9  CROSS-ATTENTION
# ---------------------------------------------------------------------------
```

## 3. Cross-Attention
Q from decoder (B, T_q, D), K/V from encoder (B, T_kv, D) -> (B, T_q, D)
No causal mask (attend to all encoder positions)

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: separate projections for Q (from decoder) and K,V (from encoder)
        raise NotImplementedError

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """x_q: (B, T_q, D), x_kv: (B, T_kv, D) -> (B, T_q, D)."""
        raise NotImplementedError
```

```python
# TEST 3: Cross-Attention
B, T_q, T_kv, D, H = 2, 6, 10, 32, 4
ca = CrossAttention(D, H)
x_q = torch.randn(B, T_q, D)
x_kv = torch.randn(B, T_kv, D)
out = ca(x_q, x_kv)
assert out.shape == (B, T_q, D), f"Expected {(B, T_q, D)}, got {out.shape}"
print("3/9 Cross-Attention PASSED")

# ---------------------------------------------------------------------------
# 4/9  LoRA LINEAR
# ---------------------------------------------------------------------------
```

## 4. LoRA Linear Layer
Frozen base weight W, low-rank adaptation: W + (alpha/r) * B @ A
A: (r, in), B: (out, r) — B is zero-initialized so output starts as base

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        # TODO: base weight (frozen), lora_A (random init), lora_B (zero init)
        # TODO: store scaling = alpha / rank
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)."""
        raise NotImplementedError
```

```python
# TEST 4: LoRA Linear
lora = LoRALinear(32, 64, rank=4)
x = torch.randn(2, 8, 32)
out = lora(x)
assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"
# At init, B=0 so LoRA contribution should be zero
base_out = F.linear(x, lora.base_weight)
assert torch.allclose(out, base_out, atol=1e-5), "At init, LoRA output should equal base output"
print("4/9 LoRA Linear PASSED")

# ---------------------------------------------------------------------------
# 5/9  SOFTMAX
# ---------------------------------------------------------------------------
```

## 5. Numerically Stable Softmax
Input: (*, D) -> Output: (*, D)
Subtract max for stability: exp(x - max(x)) / sum(exp(x - max(x)))

```python
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax along given dimension.

    Args:
        x: input tensor of any shape
        dim: dimension to apply softmax over

    Returns:
        tensor of same shape, values in (0,1), summing to 1 along dim
    """
    # TODO: subtract max for stability
    # TODO: exponentiate
    # TODO: normalize
    raise NotImplementedError
```

```python
# TEST 5: Softmax
x = torch.randn(4, 10)
out = my_softmax(x)
assert out.shape == x.shape, f"Shape mismatch"
assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5), "Should sum to 1"
assert (out > 0).all(), "All values should be positive"
# Stability test with large values
x_large = torch.tensor([1000.0, 1001.0, 1002.0])
out_large = my_softmax(x_large)
assert not torch.isnan(out_large).any(), "Should handle large values without NaN"
assert torch.allclose(out_large.sum(), torch.tensor(1.0), atol=1e-5)
print("5/9 Softmax PASSED")

# ---------------------------------------------------------------------------
# 6/9  CROSS-ENTROPY LOSS
# ---------------------------------------------------------------------------
```

## 6. Cross-Entropy Loss
Input: logits (B, C), targets (B,) integer class labels -> scalar loss
Formula: -log(softmax(logits)[target]) averaged over batch

```python
def my_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss.

    Args:
        logits: (B, C) raw scores
        targets: (B,) integer class indices

    Returns:
        scalar loss
    """
    # TODO: log-softmax for stability
    # TODO: gather target class log-probs
    # TODO: return negative mean
    raise NotImplementedError
```

```python
# TEST 6: Cross-Entropy
torch.manual_seed(42)
logits = torch.randn(8, 5)
targets = torch.randint(0, 5, (8,))
loss = my_cross_entropy(logits, targets)
expected = F.cross_entropy(logits, targets)
assert torch.allclose(loss, expected, atol=1e-5), f"Expected {expected.item():.6f}, got {loss.item():.6f}"
assert loss.shape == (), "Should be scalar"
print("6/9 Cross-Entropy PASSED")

# ---------------------------------------------------------------------------
# 7/9  LAYER NORM
# ---------------------------------------------------------------------------
```

## 7. Layer Normalization
Input: (B, T, D) -> Output: (B, T, D)
Normalize over last dim: (x - mean) / sqrt(var + eps) * gamma + beta

```python
class MyLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        # TODO: learnable gamma (ones) and beta (zeros)
        # TODO: store eps
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) -> (..., D) normalized over last dimension."""
        raise NotImplementedError
```

```python
# TEST 7: Layer Norm
ln = MyLayerNorm(32)
x = torch.randn(2, 8, 32)
out = ln(x)
assert out.shape == (2, 8, 32), f"Shape mismatch"
# After layernorm, each vector should be ~zero mean, ~unit variance
mean = out.mean(dim=-1)
var = out.var(dim=-1, unbiased=False)
assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), "Mean should be ~0"
assert torch.allclose(var, torch.ones_like(var), atol=1e-3), "Variance should be ~1"
print("7/9 Layer Norm PASSED")

# ---------------------------------------------------------------------------
# 8/9  TRANSFORMER BLOCK
# ---------------------------------------------------------------------------
```

## 8. Transformer Block (Pre-Norm)
Pre-norm architecture: x = x + attn(norm1(x)), x = x + ffn(norm2(x))
FFN: Linear(D, 4D) -> GELU -> Linear(4D, D)

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: layer norm 1, multi-head attention, layer norm 2, FFN
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        raise NotImplementedError
```

```python
# TEST 8: Transformer Block
B, T, D, H = 2, 8, 32, 4
block = TransformerBlock(D, H)
x = torch.randn(B, T, D)
out = block(x)
assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
# Should be different from input (transformation happened)
assert not torch.allclose(out, x), "Output should differ from input"
print("8/9 Transformer Block PASSED")

# ---------------------------------------------------------------------------
# 9/9  TRAINING LOOP
# ---------------------------------------------------------------------------
```

## 9. Training Loop Function
Input: model, dataloader (list of (x, y) batches), epochs, lr
Standard loop: forward, loss, backward, step, zero_grad
Return list of average losses per epoch

```python
def train_loop(
    model: nn.Module,
    dataloader: list,
    epochs: int = 3,
    lr: float = 1e-3,
) -> list:
    """Basic training loop.

    Args:
        model: nn.Module with forward(x) -> logits
        dataloader: list of (input, target) tuples
        epochs: number of training epochs
        lr: learning rate

    Returns:
        list of average loss per epoch
    """
    # TODO: set up optimizer (Adam)
    # TODO: epoch loop
    # TODO: batch loop — forward, compute cross-entropy, backward, step, zero_grad
    # TODO: track and return average loss per epoch
    raise NotImplementedError
```

```python
# TEST 9: Training Loop
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(16, 4)
    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)
model = TinyModel()
fake_data = [(torch.randn(8, 16), torch.randint(0, 4, (8,))) for _ in range(5)]
losses = train_loop(model, fake_data, epochs=10, lr=1e-2)
assert len(losses) == 10, f"Expected 10 epoch losses, got {len(losses)}"
assert losses[-1] < losses[0], f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
print(f"9/9 Training Loop PASSED (loss: {losses[0]:.4f} -> {losses[-1]:.4f})")
```

## Results

```python
elapsed = time.time() - start_time
minutes = elapsed / 60
print(f"\nCompleted in {minutes:.1f} minutes")
if minutes < 60:
    print("All 9 in under 60 min = ready for interviews")
else:
    print("Keep practicing — target is under 60 minutes!")
```
