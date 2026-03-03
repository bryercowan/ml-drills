# %% [markdown]
# # Speed Run — All Implementations From Scratch
#
# Open this notebook. Set a **60-minute timer**. Implement everything without looking at the other notebooks.
#
# This is your interview simulation. No hints. No building blocks. Just you and the spec.
#
# **Rules:**
# 1. No peeking at other notebooks or external references
# 2. Each function must pass its assertions
# 3. Track your time per drill — know where you're slow

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# %% [markdown]
# ## 1. Self-Attention
#
# **Input:** `Q, K, V` each `(batch, seq_len, d_k)`, optional `mask` (boolean, True = ignore)
# **Output:** `(batch, seq_len, d_k)` — attended values
# **Key:** Scale by `sqrt(d_k)`. If masked, set those scores to `-inf` before softmax.

# %%
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

# %% [markdown]
# ## 2. Multi-Head Attention
#
# **Input:** `x` `(batch, seq_len, d_model)`, `n_heads` int
# **Output:** `(batch, seq_len, d_model)`
# **Key:** Project to Q/K/V, split into heads, attend per head, concat, project out. Use `nn.Linear` for projections.

# %%
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

# %% [markdown]
# ## 3. Cross-Attention
#
# **Input:** `x_q` `(batch, seq_q, d_model)`, `x_kv` `(batch, seq_kv, d_model)`, `n_heads` int
# **Output:** `(batch, seq_q, d_model)`
# **Key:** Q comes from `x_q`, K and V come from `x_kv`. Otherwise identical to multi-head attention.

# %%
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

# %% [markdown]
# ## 4. LoRA Linear Layer
#
# **Input:** `x` `(batch, in_features)`, rank `r`, frozen base weight
# **Output:** `(batch, out_features)`
# **Key:** `y = W_frozen @ x + (B @ A) @ x`. A is `(r, in)` init N(0, σ²), B is `(out, r)` init zeros. Freeze `W`.

# %%
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

# %% [markdown]
# ## 5. Numerically Stable Softmax
#
# **Input:** `x` arbitrary shape tensor, `dim` int
# **Output:** same shape, sums to 1 along `dim`
# **Key:** Subtract `x.max(dim, keepdim=True)` before exp. No `torch.softmax`.

# %%
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

# %% [markdown]
# ## 6. Cross-Entropy Loss
#
# **Input:** `logits` `(batch, num_classes)`, `targets` `(batch,)` integer class indices
# **Output:** scalar loss
# **Key:** `loss = -log(softmax(logits))[target]`, averaged over batch. Use log-sum-exp trick.

# %%
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

# %% [markdown]
# ## 7. Layer Normalization
#
# **Input:** `x` `(batch, ..., d)`, `d` is normalized dimension (last dim)
# **Output:** same shape, normalized to zero mean / unit variance along last dim, then scale+shift
# **Key:** `y = gamma * (x - mean) / sqrt(var + eps) + beta`. Learnable `gamma` (init 1) and `beta` (init 0).

# %%
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

# %% [markdown]
# ## 8. Transformer Block (Pre-Norm)
#
# **Input:** `x` `(batch, seq_len, d_model)`, `n_heads` int, `d_ff` int (feedforward hidden dim)
# **Output:** `(batch, seq_len, d_model)`
# **Key:** Pre-norm style: `x = x + MHA(LN(x))`, then `x = x + FFN(LN(x))`. FFN = Linear→ReLU→Linear.

# %%
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

# %% [markdown]
# ## 9. Training Loop
#
# **Spec:** Write a training function that takes a model, dataloader, optimizer, loss_fn, device, and num_epochs. Returns list of average losses per epoch.
# **Requirements:** Move batches to device. Zero grads. Forward. Loss. Backward. Step. Track running loss.
# **Signature:** `train(model, dataloader, optimizer, loss_fn, device, num_epochs) -> list[float]`

# %%
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

# %% [markdown]
# ---
#
# **If you finished all 9 in under 60 minutes, you're ready.**
#
# Go back and redo any that took more than 8 minutes. Those are your weak spots.
