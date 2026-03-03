# %% [markdown]
# # Drill 12: Sinusoidal Positional Encoding
# Implement the fixed positional encoding from "Attention Is All You Need".
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# %%
import torch
import math

# %%
def sinusoidal_pe(max_seq_len: int, d_model: int) -> torch.Tensor:
    """Generate sinusoidal positional encodings.

    Args:
        max_seq_len: maximum sequence length
        d_model: embedding dimension

    Returns:
        (max_seq_len, d_model) tensor of positional encodings
    """
    # TODO: create position indices (0 to max_seq_len-1), shape (max_seq_len, 1)
    # TODO: compute div_term = exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))
    # TODO: even dimensions (0, 2, 4, ...) get sin(pos * div_term)
    # TODO: odd dimensions (1, 3, 5, ...) get cos(pos * div_term)
    # TODO: return the full (max_seq_len, d_model) encoding
    raise NotImplementedError

# %% [markdown]
# ## Tests

# %%
# TEST: Sinusoidal Positional Encoding
pe = sinusoidal_pe(50, 64)

# Shape check
assert pe.shape == (50, 64), f"Expected (50, 64), got {pe.shape}"

# Values should be in [-1, 1] (sin and cos range)
assert pe.min() >= -1.0, f"Min value {pe.min().item()} should be >= -1"
assert pe.max() <= 1.0, f"Max value {pe.max().item()} should be <= 1"

# Position 0 should have sin(0)=0 for even dims
assert torch.allclose(pe[0, 0], torch.tensor(0.0), atol=1e-6), "PE[0,0] should be sin(0)=0"

# Position 0 should have cos(0)=1 for odd dims
assert torch.allclose(pe[0, 1], torch.tensor(1.0), atol=1e-6), "PE[0,1] should be cos(0)=1"

# Different positions should have different encodings
assert not torch.allclose(pe[0], pe[1]), "Different positions should have different encodings"

# Test with different sizes
pe2 = sinusoidal_pe(100, 128)
assert pe2.shape == (100, 128), f"Expected (100, 128), got {pe2.shape}"
assert pe2.min() >= -1.0 and pe2.max() <= 1.0, "Values out of [-1, 1] range"

# d_model=2 edge case
pe3 = sinusoidal_pe(10, 2)
assert pe3.shape == (10, 2), f"Expected (10, 2), got {pe3.shape}"

print(f"Shape: {pe.shape}")
print(f"Value range: [{pe.min().item():.4f}, {pe.max().item():.4f}]")
print(f"PE[0, :4] = {pe[0, :4].tolist()}")
print(f"PE[1, :4] = {pe[1, :4].tolist()}")
print("All positional encoding tests passed!")
