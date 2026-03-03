# %% [markdown]
# # Drill 8: Layer Normalization
# Implement LayerNorm from scratch (no nn.LayerNorm).
#
# Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
#
# - Compute mean and variance along the LAST dimension (keepdim=True)
# - gamma (scale): learnable parameter initialized to ones, shape (d_model,)
# - beta (shift): learnable parameter initialized to zeros, shape (d_model,)
# - eps: small constant for numerical stability (default 1e-5)

# %%
import torch
import torch.nn as nn

# %%
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

# %%
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
