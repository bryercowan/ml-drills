# %% [markdown]
# # Drill 0: PyTorch Fundamentals
#
# **Do this BEFORE Drill 1.** Everything else builds on these basics.
#
# This covers the 15 things you need to be fluent in before implementing
# attention, LoRA, training loops, etc. Each one is small. Run every cell.

# %%
import torch
import torch.nn as nn
import math

# %% [markdown]
# ---
# ## 1. Tensors — What They Are
#
# A tensor is just an N-dimensional array. It's the fundamental data type.
# - 0D tensor = scalar (just a number)
# - 1D tensor = vector
# - 2D tensor = matrix
# - 3D tensor = a batch of matrices (this is most of what you'll see)

# %%
scalar = torch.tensor(3.14)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
batch = torch.randn(2, 3, 4)  # 2 matrices, each 3 rows x 4 cols

print(f"scalar:  shape={scalar.shape}, ndim={scalar.ndim}, value={scalar.item()}")
print(f"vector:  shape={vector.shape}, ndim={vector.ndim}")
print(f"matrix:  shape={matrix.shape}, ndim={matrix.ndim}")
print(f"batch:   shape={batch.shape}, ndim={batch.ndim}")
print()
print("Matrix:")
print(matrix)

# %% [markdown]
# ---
# ## 2. Shape — Reading (B, T, D)
#
# ML code is ALL about shapes. You'll see `(B, T, D)` everywhere:
# - **B** = batch size (how many examples processed at once)
# - **T** = sequence length (how many tokens)
# - **D** = dimension (size of each token's vector)
#
# When you see `(2, 5, 64)` think: "2 sequences, each with 5 tokens, each token is a 64-dim vector"

# %%
B, T, D = 2, 5, 64
x = torch.randn(B, T, D)

print(f"x.shape = {x.shape}")
print(f"x.shape[0] = {x.shape[0]}  (batch size)")
print(f"x.shape[1] = {x.shape[1]}  (sequence length)")
print(f"x.shape[2] = {x.shape[2]}  (dimension)")
print(f"Number of elements: {x.numel()}")  # 2 * 5 * 64 = 640

# %% [markdown]
# ---
# ## 3. Creating Tensors
#
# The ones you'll use constantly:

# %%
# Random normal (mean=0, std=1) — used for random data, initialization
a = torch.randn(3, 4)
print("randn(3,4):", a.shape)

# Random uniform between 0 and 1
b = torch.rand(3, 4)
print("rand(3,4):", b.shape)

# Zeros and ones
c = torch.zeros(3, 4)
d = torch.ones(3, 4)
print("zeros:", c[0])
print("ones:", d[0])

# Empty (uninitialized — faster but garbage values)
e = torch.empty(3, 4)

# Like another tensor (same shape and device)
f = torch.randn_like(c)  # same shape as c, filled with randn
print("randn_like:", f.shape)

# Range
g = torch.arange(0, 10)  # [0, 1, 2, ..., 9]
print("arange:", g)

h = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
print("linspace:", h)

# %% [markdown]
# ---
# ## 4. Indexing — Getting Pieces of a Tensor
#
# Same as numpy. You'll use this constantly to inspect specific tokens,
# batches, or features.

# %%
x = torch.randn(2, 5, 64)  # (B=2, T=5, D=64)

# Single element
print("x[0, 0, 0] =", x[0, 0, 0].item())  # first batch, first token, first feature

# Slice a whole token vector
print("x[0, 0] shape:", x[0, 0].shape)  # (64,) — first token of first batch

# First batch, all tokens
print("x[0] shape:", x[0].shape)  # (5, 64)

# All batches, first token
print("x[:, 0] shape:", x[:, 0].shape)  # (2, 64)

# All batches, all tokens, first 8 features
print("x[:, :, :8] shape:", x[:, :, :8].shape)  # (2, 5, 8)

# Last element of last dim
print("x[..., -1] shape:", x[..., -1].shape)  # (2, 5)

# %% [markdown]
# ---
# ## 5. Matrix Multiply — The @ Operator
#
# **This is the most important operation in ML.** Neural nets are layers
# of matrix multiplications with nonlinearities in between.
#
# `@` is PyTorch's matrix multiply operator (`torch.matmul`).

# %%
# 2D: standard matrix multiply
A = torch.randn(3, 4)  # 3 rows, 4 cols
B_mat = torch.randn(4, 5)  # 4 rows, 5 cols
C = A @ B_mat               # (3, 4) @ (4, 5) = (3, 5)
print(f"A: {A.shape}, B: {B_mat.shape} → A @ B: {C.shape}")
print()

# The inner dimensions must match: (3, *4*) @ (*4*, 5) = (3, 5)
#                                        ^^^     ^^^
#                                     these must match

# 3D: batched matrix multiply (this is what attention uses)
# Each batch gets its own matmul, independently
X = torch.randn(2, 3, 4)  # 2 batches of (3, 4) matrices
Y = torch.randn(2, 4, 5)  # 2 batches of (4, 5) matrices
Z = X @ Y                  # 2 batches of (3, 5) matrices
print(f"X: {X.shape}, Y: {Y.shape} → X @ Y: {Z.shape}")
print()

# This is equivalent to:
Z_manual_0 = X[0] @ Y[0]  # batch 0
Z_manual_1 = X[1] @ Y[1]  # batch 1
print(f"Manual batch 0 matches: {torch.allclose(Z[0], Z_manual_0)}")

# %% [markdown]
# ---
# ## 6. Transpose — Swapping Dimensions
#
# `.transpose(dim0, dim1)` swaps two dimensions.
# `.T` is shorthand for 2D matrices only.
# You'll use `K.transpose(-2, -1)` in attention to get K^T.

# %%
x = torch.randn(2, 5, 64)
print(f"x shape:               {x.shape}")           # (2, 5, 64)
print(f"x.transpose(1, 2):     {x.transpose(1, 2).shape}")  # (2, 64, 5) — swapped T and D
print(f"x.transpose(-2, -1):   {x.transpose(-2, -1).shape}")  # same thing, using negative indices
print()

# In attention: Q @ K^T
# Q is (B, T, D), K is (B, T, D)
# K.transpose(-2, -1) is (B, D, T)
# Q @ K^T = (B, T, D) @ (B, D, T) = (B, T, T) ← the attention score matrix
Q = torch.randn(2, 5, 64)
K = torch.randn(2, 5, 64)
scores = Q @ K.transpose(-2, -1)
print(f"Q @ K^T: {Q.shape} @ {K.transpose(-2, -1).shape} = {scores.shape}")  # (2, 5, 5)

# %% [markdown]
# ---
# ## 7. View & Reshape — Changing Shape Without Changing Data
#
# `.view()` and `.reshape()` rearrange elements into a new shape.
# Total elements must stay the same: 2×6 = 3×4 = 12.
#
# You'll use this heavily in multi-head attention to split D into H heads.

# %%
x = torch.arange(12)
print(f"Original: {x.shape} → {x}")
print()

# Reshape to different shapes (same 12 elements)
print(f"view(3, 4):\n{x.view(3, 4)}")
print(f"\nview(2, 6):\n{x.view(2, 6)}")
print(f"\nview(2, 2, 3):\n{x.view(2, 2, 3)}")
print()

# -1 means "figure out this dimension for me"
print(f"view(-1, 4): {x.view(-1, 4).shape}")  # (3, 4) — 12/4 = 3
print(f"view(2, -1): {x.view(2, -1).shape}")   # (2, 6) — 12/2 = 6

# %% [markdown]
# ---
# ## 8. The Multi-Head Reshape Trick
#
# This is the trickiest shape manipulation you'll encounter.
# In multi-head attention, you split D into H heads of d_k each:
#
# `(B, T, D) → view → (B, T, H, d_k) → transpose → (B, H, T, d_k)`
#
# After attention, reverse it:
#
# `(B, H, T, d_k) → transpose → (B, T, H, d_k) → view → (B, T, D)`

# %%
B, T, D, H = 2, 5, 64, 8
d_k = D // H  # 64 // 8 = 8 per head

x = torch.randn(B, T, D)
print(f"Start:     {x.shape}")          # (2, 5, 64)

# Step 1: split D into H heads
x2 = x.view(B, T, H, d_k)
print(f"After view:      {x2.shape}")   # (2, 5, 8, 8) — 5 tokens, 8 heads, 8 dims each

# Step 2: move heads to batch-like position so each head is independent
x3 = x2.transpose(1, 2)
print(f"After transpose: {x3.shape}")   # (2, 8, 5, 8) — now (B, H, T, d_k)

# After attention, reverse:
x4 = x3.transpose(1, 2)                 # (2, 5, 8, 8)
x5 = x4.contiguous().view(B, T, D)      # (2, 5, 64) — back to original shape
print(f"Reversed:  {x5.shape}")

# WHY contiguous()? transpose() returns a "view" that may not be
# stored contiguously in memory. view() requires contiguous memory.
# contiguous() makes a fresh copy if needed.

# %% [markdown]
# ---
# ## 9. nn.Module — How PyTorch Models Work
#
# `nn.Module` is the base class for all neural network layers/models.
# Every model you build inherits from it.
#
# Two things you MUST implement:
# 1. `__init__`: create the layers
# 2. `forward`: define the computation

# %%
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # MUST call this first
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)   # linear: x @ W^T + b
        x = self.relu(x)  # activation: max(0, x)
        x = self.fc2(x)   # linear
        return x

model = SimpleModel(64, 128, 10)
x = torch.randn(4, 64)     # 4 samples, 64 features
out = model(x)              # calls model.forward(x)
print(f"Input: {x.shape} → Output: {out.shape}")

# nn.Module gives you for free:
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# %% [markdown]
# ---
# ## 10. nn.Linear — The Fundamental Layer
#
# `nn.Linear(in_features, out_features)` is a learnable matrix multiply + bias:
# `y = x @ W^T + b`
#
# W has shape (out_features, in_features). **Note the transpose!**
# This is so y = x @ W.T works when x is (batch, in_features).

# %%
linear = nn.Linear(64, 128)  # 64 in, 128 out

print(f"Weight shape: {linear.weight.shape}")  # (128, 64) — note: (out, in)
print(f"Bias shape:   {linear.bias.shape}")     # (128,)

x = torch.randn(4, 64)    # 4 samples, 64 features
y = linear(x)             # calls forward: x @ W.T + b
print(f"\nInput: {x.shape} → Output: {y.shape}")  # (4, 128)

# This is equivalent to:
y_manual = x @ linear.weight.T + linear.bias
print(f"Manual matches: {torch.allclose(y, y_manual, atol=1e-6)}")

# %% [markdown]
# ---
# ## 11. Gradients & requires_grad
#
# PyTorch tracks operations on tensors to compute gradients (for backprop).
# `requires_grad=True` tells PyTorch: "I might want gradients for this."
#
# Parameters of nn.Module automatically have requires_grad=True.

# %%
# Manual tensor with gradient tracking
w = torch.randn(3, 3, requires_grad=True)
x = torch.randn(3, 3)

# Forward
y = x @ w
loss = y.sum()

# Backward — computes dL/dw
loss.backward()
print(f"w.grad shape: {w.grad.shape}")
print(f"w.grad:\n{w.grad}")
print()

# With nn.Module, this happens automatically:
model = nn.Linear(3, 3)
x = torch.randn(2, 3)
y = model(x)
loss = y.sum()
loss.backward()
print(f"model weight grad: {model.weight.grad.shape}")
print(f"model bias grad:   {model.bias.grad.shape}")

# %% [markdown]
# ---
# ## 12. torch.no_grad() and Freezing
#
# During inference or when you freeze layers, you don't want gradient computation.
# `torch.no_grad()` disables it (saves memory and speed).

# %%
model = nn.Linear(64, 10)
x = torch.randn(4, 64)

# With gradients (training mode)
y1 = model(x)
print(f"requires_grad during training: {y1.requires_grad}")

# Without gradients (inference)
with torch.no_grad():
    y2 = model(x)
    print(f"requires_grad during inference: {y2.requires_grad}")

# Freezing a parameter (like your frozen vision encoder)
for param in model.parameters():
    param.requires_grad = False

y3 = model(x)
print(f"requires_grad after freezing: {y3.requires_grad}")

# %% [markdown]
# ---
# ## 13. Common Tensor Operations
#
# Quick reference for operations you'll use in every drill.

# %%
x = torch.randn(2, 5, 64)

# Reduce operations (along a dimension)
print(f"x.mean(dim=-1) shape: {x.mean(dim=-1).shape}")           # (2, 5) — mean over last dim
print(f"x.mean(dim=-1, keepdim=True): {x.mean(dim=-1, keepdim=True).shape}")  # (2, 5, 1) — keep dim for broadcasting
print(f"x.sum(dim=1) shape: {x.sum(dim=1).shape}")               # (2, 64) — sum over tokens
print(f"x.max(dim=-1).values shape: {x.max(dim=-1).values.shape}")  # (2, 5) — max over features
print()

# Softmax (turns values into probabilities that sum to 1)
logits = torch.randn(2, 5)
probs = torch.softmax(logits, dim=-1)
print(f"softmax: {probs[0]}")
print(f"sum: {probs[0].sum()}")  # 1.0
print()

# Cumulative product (used in diffusion schedules)
a = torch.tensor([0.99, 0.98, 0.97, 0.96])
print(f"cumprod: {torch.cumprod(a, dim=0)}")  # 0.99, 0.99*0.98, 0.99*0.98*0.97, ...
print()

# Masking
mask = torch.tril(torch.ones(4, 4))  # lower triangular
print(f"tril mask:\n{mask}")

scores = torch.randn(4, 4)
masked = scores.masked_fill(mask == 0, float('-inf'))
print(f"\nMasked scores:\n{masked}")
print(f"\nAfter softmax (upper triangle → 0):\n{torch.softmax(masked, dim=-1)}")

# %% [markdown]
# ---
# ## 14. Broadcasting
#
# When tensors have different shapes, PyTorch "broadcasts" the smaller one
# to match. This lets you add a bias vector (D,) to a batch (B, T, D)
# without explicit looping.
#
# Rules: dimensions are compared from right to left. They match if they're
# equal OR one of them is 1.

# %%
x = torch.randn(2, 5, 64)  # (2, 5, 64)
bias = torch.randn(64)       # (64,)

# This works! bias is broadcast: (64,) → (1, 1, 64) → (2, 5, 64)
y = x + bias
print(f"x: {x.shape} + bias: {bias.shape} → {y.shape}")

# keepdim=True matters for broadcasting:
mean_keepdim = x.mean(dim=-1, keepdim=True)  # (2, 5, 1)
mean_no_keepdim = x.mean(dim=-1)              # (2, 5)

centered = x - mean_keepdim  # works: (2,5,64) - (2,5,1) broadcasts
print(f"Centered with keepdim: {centered.shape}")

# This would fail without keepdim:
# x - mean_no_keepdim  # error: (2,5,64) - (2,5) doesn't broadcast

# %% [markdown]
# ---
# ## 15. Device (CPU/GPU/MPS)
#
# Tensors live on a specific device. All tensors in an operation
# must be on the same device.

# %%
x_cpu = torch.randn(3, 3)
print(f"Default device: {x_cpu.device}")

# Move to GPU if available
if torch.cuda.is_available():
    x_gpu = x_cpu.to('cuda')
    print(f"CUDA: {x_gpu.device}")
elif torch.backends.mps.is_available():
    x_mps = x_cpu.to('mps')
    print(f"MPS: {x_mps.device}")
else:
    print("No GPU available, CPU only")

# Model + data must be on the same device:
# model = model.to('cuda')
# x = x.to('cuda')
# out = model(x)  # both on cuda, this works

# %% [markdown]
# ---
# ## You're Ready
#
# If you ran every cell and nothing felt confusing, go start Drill 1.
#
# If anything felt shaky, re-run that section and experiment:
# - Change the shapes and see what breaks
# - Print intermediate values
# - Try to predict the output shape before running the cell
#
# The most important ones for Drill 1 (self-attention):
# - **Section 5**: `@` operator (matrix multiply)
# - **Section 6**: `transpose` (you'll need K^T)
# - **Section 8**: the multi-head reshape trick
# - **Section 13**: `softmax`, `masked_fill`, `tril`
