# %% [markdown]
# # Drill 13: Speed Run — 60 minutes, all implementations, no peeking
# Start a timer. Implement all 9 from scratch.
# No looking at previous drills. No references. Just you and PyTorch.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

start_time = time.time()

# ---------------------------------------------------------------------------
# 1/9  SELF-ATTENTION
# ---------------------------------------------------------------------------

# %% [markdown]
# ## 1. Self-Attention
# Input: (B, T, D) -> Output: (B, T, D)
# Must apply causal mask (no attending to future positions)

# %%
class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # TODO: Q, K, V projections and output projection
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D) with causal masking."""
        raise NotImplementedError

# %%
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

# %% [markdown]
# ## 2. Multi-Head Attention
# Input: (B, T, D), H heads -> Output: (B, T, D)
# Split D into H heads, each head_dim = D // H

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: Q, K, V projections, output projection
        # TODO: store n_heads, head_dim
        raise NotImplementedError

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D) with optional causal mask."""
        raise NotImplementedError

# %%
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

# %% [markdown]
# ## 3. Cross-Attention
# Q from decoder (B, T_q, D), K/V from encoder (B, T_kv, D) -> (B, T_q, D)
# No causal mask (attend to all encoder positions)

# %%
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: separate projections for Q (from decoder) and K,V (from encoder)
        raise NotImplementedError

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """x_q: (B, T_q, D), x_kv: (B, T_kv, D) -> (B, T_q, D)."""
        raise NotImplementedError

# %%
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

# %% [markdown]
# ## 4. LoRA Linear Layer
# Frozen base weight W, low-rank adaptation: W + (alpha/r) * B @ A
# A: (r, in), B: (out, r) — B is zero-initialized so output starts as base

# %%
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        # TODO: base weight (frozen), lora_A (random init), lora_B (zero init)
        # TODO: store scaling = alpha / rank
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)."""
        raise NotImplementedError

# %%
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

# %% [markdown]
# ## 5. Numerically Stable Softmax
# Input: (*, D) -> Output: (*, D)
# Subtract max for stability: exp(x - max(x)) / sum(exp(x - max(x)))

# %%
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

# %%
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

# %% [markdown]
# ## 6. Cross-Entropy Loss
# Input: logits (B, C), targets (B,) integer class labels -> scalar loss
# Formula: -log(softmax(logits)[target]) averaged over batch

# %%
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

# %%
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

# %% [markdown]
# ## 7. Layer Normalization
# Input: (B, T, D) -> Output: (B, T, D)
# Normalize over last dim: (x - mean) / sqrt(var + eps) * gamma + beta

# %%
class MyLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        # TODO: learnable gamma (ones) and beta (zeros)
        # TODO: store eps
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) -> (..., D) normalized over last dimension."""
        raise NotImplementedError

# %%
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

# %% [markdown]
# ## 8. Transformer Block (Pre-Norm)
# Pre-norm architecture: x = x + attn(norm1(x)), x = x + ffn(norm2(x))
# FFN: Linear(D, 4D) -> GELU -> Linear(4D, D)

# %%
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: layer norm 1, multi-head attention, layer norm 2, FFN
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        raise NotImplementedError

# %%
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

# %% [markdown]
# ## 9. Training Loop Function
# Input: model, dataloader (list of (x, y) batches), epochs, lr
# Standard loop: forward, loss, backward, step, zero_grad
# Return list of average losses per epoch

# %%
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

# %%
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

# %% [markdown]
# ## Results

# %%
elapsed = time.time() - start_time
minutes = elapsed / 60
print(f"\nCompleted in {minutes:.1f} minutes")
if minutes < 60:
    print("All 9 in under 60 min = ready for interviews")
else:
    print("Keep practicing — target is under 60 minutes!")
