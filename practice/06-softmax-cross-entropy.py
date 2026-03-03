# %% [markdown]
# # Drill 6a: Softmax
# Implement numerically stable softmax from scratch.
# - Subtract max along last dim before exp (prevents overflow)
# - Apply along last dim (dim=-1) for any input shape
# - Return tensor of same shape with values summing to 1 along last dim

# %%
import torch
import torch.nn.functional as F

# %%
def my_softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax along last dimension."""
    # TODO: subtract max along last dim (keepdim=True) for stability
    # TODO: exponentiate
    # TODO: divide by sum along last dim (keepdim=True)
    pass

# %%
# TEST: softmax
x = torch.randn(4, 10)
out = my_softmax(x)
ref = torch.softmax(x, dim=-1)
assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
assert torch.allclose(out, ref, atol=1e-6), f"Max diff: {(out - ref).abs().max().item()}"
assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-6), "Rows don't sum to 1"

# also test 3D input
x3d = torch.randn(2, 3, 5)
out3d = my_softmax(x3d)
ref3d = torch.softmax(x3d, dim=-1)
assert torch.allclose(out3d, ref3d, atol=1e-6), "Failed on 3D input"
print("softmax: ALL TESTS PASSED")

# %% [markdown]
# # Drill 6b: Cross-Entropy Loss
# Implement cross-entropy loss using your my_softmax.
# - Apply softmax to logits to get probabilities
# - For each sample, pick the probability of the correct class
# - Compute -log(prob) and return the mean over the batch

# %%
def my_cross_entropy(logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
    """Cross-entropy loss. logits: (B, C), targets: (B,) class indices."""
    # TODO: get probabilities using my_softmax
    # TODO: gather/index the probability of the correct class for each sample
    # TODO: compute -log(correct_probs).mean()
    pass

# %%
# TEST: cross-entropy
torch.manual_seed(42)
logits = torch.randn(8, 5)
targets = torch.randint(0, 5, (8,))
out = my_cross_entropy(logits, targets)
ref = F.cross_entropy(logits, targets)
assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
assert torch.allclose(out, ref, atol=1e-5), f"Mismatch: {out.item():.6f} vs {ref.item():.6f}"
print(f"cross_entropy: {out.item():.6f} vs reference {ref.item():.6f}")
print("cross_entropy: ALL TESTS PASSED")
