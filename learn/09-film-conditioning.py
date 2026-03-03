# %% [markdown]
# # Drill 09 — FiLM (Feature-wise Linear Modulation)
#
# **Why this matters for your architecture:**
# - In your CNN policy, **FiLM provides GLOBAL conditioning** — it tells every feature map
#   *"what to do"* (e.g. "we're placing a road tile") by scaling and shifting activations.
# - **Cross-attention provides SPATIAL conditioning** — *"where to do it"*.
# - FiLM is also how diffusion models inject **timestep conditioning** into the denoiser.
#
# It's the simplest conditioning mechanism: just a learned scale and shift.

# %% [markdown]
# ## Intuition
#
# Given features `x` of shape `(B, D)` and a conditioning vector `c` of shape `(B, C)`:
#
# ```
# gamma = Linear(c)   # (B, C) -> (B, D)  — learned scale
# beta  = Linear(c)   # (B, C) -> (B, D)  — learned shift
# output = gamma * x + beta
# ```
#
# That's it. Two linear layers predict per-feature scale and shift from the conditioning signal.

# %%
import torch
import torch.nn as nn
torch.manual_seed(42)

# %% [markdown]
# ## Building Block 1: The Conditioning Projections
#
# Two linear layers map the conditioning vector to gamma and beta.
# Each has shape `(B, D)` — one scale and one shift per feature dimension.

# %%
B, C, D = 2, 8, 16  # batch, conditioning_dim, feature_dim

cond = torch.randn(B, C)  # conditioning signal

gamma_proj = nn.Linear(C, D)
beta_proj  = nn.Linear(C, D)

gamma = gamma_proj(cond)  # (B, D)
beta  = beta_proj(cond)   # (B, D)

print(f"cond:  {cond.shape}")
print(f"gamma: {gamma.shape}")
print(f"beta:  {beta.shape}")

# %% [markdown]
# ## Building Block 2: The Modulation
#
# Element-wise multiply by gamma, then add beta. Watch how it transforms the features.

# %%
x = torch.randn(B, D)  # input features

# FiLM modulation — one line!
out = gamma * x + beta

print(f"x  : mean={x.mean():.3f}, std={x.std():.3f}")
print(f"out: mean={out.mean():.3f}, std={out.std():.3f}")
print(f"\nFirst sample, first 4 features:")
print(f"  x     = {x[0, :4].tolist()}")
print(f"  gamma = {gamma[0, :4].tolist()}")
print(f"  beta  = {beta[0, :4].tolist()}")
print(f"  out   = {out[0, :4].tolist()}")

# %% [markdown]
# ## Building Block 3: FiLM on Spatial Feature Maps
#
# For CNN feature maps `(B, D, H, W)`, we reshape gamma/beta to `(B, D, 1, 1)` so they
# broadcast across all spatial positions — **same scale/shift everywhere** (global conditioning).
#
# This is exactly how diffusion models inject timestep `t`:
# `t -> sinusoidal_embed -> MLP -> gamma, beta -> modulate each ResBlock`

# %%
H, W = 4, 4
feat_map = torch.randn(B, D, H, W)  # CNN feature map

# Reshape for broadcasting: (B, D) -> (B, D, 1, 1)
gamma_spatial = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
beta_spatial  = beta.unsqueeze(-1).unsqueeze(-1)    # (B, D, 1, 1)

out_spatial = gamma_spatial * feat_map + beta_spatial
print(f"feat_map: {feat_map.shape}")
print(f"gamma:    {gamma_spatial.shape}  (broadcasts over H, W)")
print(f"output:   {out_spatial.shape}")

# %% [markdown]
# ## Complete FiLM Layer

# %%
class FiLM(nn.Module):
    """Feature-wise Linear Modulation.
    
    Maps conditioning vector -> (gamma, beta) and applies
    output = gamma * features + beta.
    Works for both flat (B, D) and spatial (B, D, H, W) features.
    """
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj  = nn.Linear(cond_dim, feat_dim)
        # Initialize gamma to 1, beta to 0 (identity at init)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, features, cond):
        gamma = self.gamma_proj(cond)  # (B, D)
        beta  = self.beta_proj(cond)   # (B, D)
        # If features are spatial (B, D, H, W), unsqueeze for broadcasting
        if features.dim() > 2:
            gamma = gamma.view(*gamma.shape, *([1] * (features.dim() - 2)))
            beta  = beta.view(*beta.shape, *([1] * (features.dim() - 2)))
        return gamma * features + beta

# %% [markdown]
# ## Your Turn
#
# Implement `FiLMBlock` — a FiLM-conditioned residual block:
# 1. Conv2d -> BatchNorm -> FiLM -> ReLU
# 2. Add residual connection

# %%
# YOUR CODE HERE
class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        pass  # TODO

    def forward(self, x, cond):
        pass  # TODO

# ---- Tests ----
film = FiLM(cond_dim=8, feat_dim=16)
x_flat = torch.randn(2, 16)
x_spatial = torch.randn(2, 16, 4, 4)
c = torch.randn(2, 8)

# At init: gamma=1, beta=0, so output ≈ input
out_flat = film(x_flat, c)
assert out_flat.shape == (2, 16), f"Expected (2, 16), got {out_flat.shape}"
assert torch.allclose(out_flat, x_flat, atol=1e-6), "FiLM should be identity at init"

out_sp = film(x_spatial, c)
assert out_sp.shape == (2, 16, 4, 4), f"Expected (2, 16, 4, 4), got {out_sp.shape}"
assert torch.allclose(out_sp, x_spatial, atol=1e-6), "Spatial FiLM should be identity at init"

# After a gradient step gamma/beta should change
loss = out_sp.sum()
loss.backward()
print("All tests passed!")
