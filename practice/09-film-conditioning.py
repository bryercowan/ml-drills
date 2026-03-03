# %% [markdown]
# # Drill 9: FiLM Conditioning
# Implement Feature-wise Linear Modulation (FiLM).
#
# FiLM modulates feature maps using a conditioning signal:
#   output = gamma * features + beta
#
# where gamma and beta are predicted from a conditioning vector
# via learned linear projections.
#
# - scale_net: Linear(cond_dim -> feature_dim) predicts gamma
# - shift_net: Linear(cond_dim -> feature_dim) predicts beta
# - Forward: take features (B, feature_dim) and cond (B, cond_dim),
#   predict gamma and beta from cond, apply: gamma * features + beta

# %%
import torch
import torch.nn as nn

# %%
class MyFiLM(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        # TODO: scale_net = Linear(cond_dim, feature_dim) to predict gamma
        # TODO: shift_net = Linear(cond_dim, feature_dim) to predict beta
        pass

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        features: (B, feature_dim)
        cond: (B, cond_dim)
        returns: (B, feature_dim)
        """
        # TODO: predict gamma from cond using scale_net
        # TODO: predict beta from cond using shift_net
        # TODO: return gamma * features + beta
        pass

# %%
# TEST: shape and basic behavior
torch.manual_seed(0)
feat_dim, cond_dim, batch = 128, 32, 4

film = MyFiLM(feat_dim, cond_dim)
features = torch.randn(batch, feat_dim)
cond = torch.randn(batch, cond_dim)

out = film(features, cond)
assert out.shape == (batch, feat_dim), f"Expected ({batch}, {feat_dim}), got {out.shape}"

# Different conditioning should give different outputs
cond2 = torch.randn(batch, cond_dim)
out2 = film(features, cond2)
assert not torch.allclose(out, out2), "Different conditioning should produce different outputs"

# Zero conditioning should still work (bias terms)
cond_zero = torch.zeros(batch, cond_dim)
out_zero = film(features, cond_zero)
assert out_zero.shape == (batch, feat_dim), "Should work with zero conditioning"

print(f"Output shape: {out.shape}")
print("FiLM conditioning: ALL TESTS PASSED")
