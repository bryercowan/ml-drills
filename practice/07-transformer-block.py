# %% [markdown]
# # Drill 7: Transformer Block (Pre-Norm)
# Implement a single pre-norm transformer block.
#
# Architecture:
#   x = x + attn(norm1(x))    # self-attention with residual
#   x = x + ffn(norm2(x))     # feed-forward with residual
#
# Components:
#   - norm1, norm2: nn.LayerNorm(d_model)
#   - attn: nn.MultiheadAttention(d_model, num_heads, batch_first=True)
#   - ffn: Sequential(Linear(D, 4D), GELU, Linear(4D, D))
#
# Forward should apply a causal mask so positions can only attend to past.
# You may use nn.LayerNorm, nn.MultiheadAttention, nn.Linear, nn.GELU.

# %%
import torch
import torch.nn as nn

# %%
class MyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: create norm1 (LayerNorm)
        # TODO: create norm2 (LayerNorm)
        # TODO: create attn (MultiheadAttention with batch_first=True)
        # TODO: create ffn (Sequential: Linear(D,4D) -> GELU -> Linear(4D,D))
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        # TODO: generate causal mask (upper triangular) of shape (seq_len, seq_len)
        #       Use torch.nn.Transformer.generate_square_subsequent_mask or manual
        # TODO: pre-norm self-attention with residual
        #       norm -> attn(query, key, value, attn_mask=mask) -> add residual
        # TODO: pre-norm FFN with residual
        #       norm -> ffn -> add residual
        pass

# %%
# TEST: shape check
torch.manual_seed(0)
block = MyTransformerBlock(d_model=64, num_heads=4)
x = torch.randn(2, 10, 64)  # (batch=2, seq=10, d=64)
out = block(x)
assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"
# Check that output is different from input (something happened)
assert not torch.allclose(x, out), "Output should differ from input"
print(f"Output shape: {out.shape}")
print("transformer block: ALL TESTS PASSED")
