# %% [markdown]
# # Drill 0: PyTorch Basics — Check Yourself
#
# Run each cell. Predict the output BEFORE running. If you get it wrong, go
# back to learn/00-pytorch-basics.py and re-read that section.

# %%
import torch
import torch.nn as nn
import math

# %% [markdown]
# ## Question 1: What's the shape?

# %%
x = torch.randn(4, 10, 256)
# Predict: what is x.shape[1]?
# Predict: what is x.numel()?
print(f"x.shape[1] = {x.shape[1]}")
print(f"x.numel() = {x.numel()}")

# %% [markdown]
# ## Question 2: Matrix multiply shapes

# %%
A = torch.randn(2, 5, 64)
B = torch.randn(2, 64, 10)
C = A @ B
# Predict: what is C.shape?
print(f"C.shape = {C.shape}")

# %%
Q = torch.randn(2, 5, 64)
K = torch.randn(2, 5, 64)
scores = Q @ K.transpose(-2, -1)
# Predict: what is scores.shape?
print(f"scores.shape = {scores.shape}")

# %% [markdown]
# ## Question 3: The reshape trick

# %%
x = torch.randn(2, 5, 64)
B, T, D, H = 2, 5, 64, 8
d_k = D // H
x2 = x.view(B, T, H, d_k)
x3 = x2.transpose(1, 2)
# Predict: x2.shape? x3.shape?
print(f"x2.shape = {x2.shape}")
print(f"x3.shape = {x3.shape}")

# %% [markdown]
# ## Question 4: Softmax + masking

# %%
scores = torch.randn(1, 4, 4)
mask = torch.tril(torch.ones(4, 4))
masked_scores = scores.masked_fill(mask == 0, float('-inf'))
weights = torch.softmax(masked_scores, dim=-1)
# Predict: what does weights[0, 0, :] look like? (first row)
# Predict: what does weights[0, 0, :].sum() equal?
print(f"weights[0, 0, :] = {weights[0, 0, :]}")
print(f"sum = {weights[0, 0, :].sum()}")

# %% [markdown]
# ## Question 5: nn.Linear shapes

# %%
linear = nn.Linear(64, 128)
# Predict: linear.weight.shape?
# Predict: linear.bias.shape?
print(f"weight: {linear.weight.shape}")
print(f"bias: {linear.bias.shape}")

x = torch.randn(4, 64)
y = linear(x)
# Predict: y.shape?
print(f"y.shape = {y.shape}")

# %% [markdown]
# ## Question 6: Broadcasting

# %%
x = torch.randn(2, 5, 64)
mean = x.mean(dim=-1, keepdim=True)
# Predict: mean.shape?
print(f"mean.shape = {mean.shape}")

centered = x - mean
# Predict: centered.shape?
print(f"centered.shape = {centered.shape}")

# %% [markdown]
# ## Question 7: Gradients

# %%
w = torch.randn(3, requires_grad=True)
y = (w ** 2).sum()
y.backward()
# Predict: what is w.grad? (derivative of sum(w^2) w.r.t. w is 2*w)
print(f"w = {w.data}")
print(f"w.grad = {w.grad}")
print(f"2*w = {2*w.data}")
print(f"match: {torch.allclose(w.grad, 2*w.data)}")

# %% [markdown]
# ---
# If you got every prediction right, go to drill 01. If not, review
# the section in learn/00-pytorch-basics.py before moving on.
