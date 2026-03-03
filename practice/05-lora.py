# %% [markdown]
# # Drill 5: LoRA (Low-Rank Adaptation)
# Implement a LoRA wrapper around a frozen nn.Linear.
# Core idea: output = base(x) + (x @ A^T) @ B^T * scaling
# A is initialized small random, B is initialized ZERO.
# At init, LoRA contributes nothing (B=0), so output == base output.
# Steps:
# 1. Store base_linear and freeze it (requires_grad=False)
# 2. Create A: nn.Parameter of shape (r, in_features), init small random
# 3. Create B: nn.Parameter of shape (out_features, r), init ZERO
# 4. Compute scaling = alpha / r
# 5. Forward: base_out + (x @ A^T) @ B^T * scaling

# %%
import torch
import torch.nn as nn

# %%
class MyLoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, alpha=32):
        super().__init__()
        # TODO step 1: store base_linear and freeze all its parameters
        # TODO step 2: create A as nn.Parameter, shape (r, in_features), init with small random (e.g. kaiming or normal with std=0.01)
        # TODO step 3: create B as nn.Parameter, shape (out_features, r), init ZERO
        # TODO step 4: compute and store self.scaling = alpha / r
        pass

    def forward(self, x):
        # x: (..., in_features)
        # TODO step 1: base_out = base linear forward
        # TODO step 2: lora_out = (x @ A^T) @ B^T * scaling
        # TODO step 3: return base_out + lora_out
        raise NotImplementedError

# %%
# TEST 1: shape check
base = nn.Linear(64, 128)
lora = MyLoRALinear(base, r=16, alpha=32)
x = torch.randn(2, 5, 64)
out = lora(x)
assert out.shape == (2, 5, 128), f'Wrong shape: {out.shape}'
print('PASS: shape check')

# %%
# TEST 2: at init, output matches base (B=0 so LoRA adds nothing)
base2 = nn.Linear(64, 128)
lora2 = MyLoRALinear(base2, r=8, alpha=16)
x2 = torch.randn(3, 10, 64)
with torch.no_grad():
    base_out = base2(x2)
    lora_out = lora2(x2)
assert torch.allclose(base_out, lora_out, atol=1e-6), 'At init, LoRA output should match base output (B is zero)'
print('PASS: init equivalence')

# %%
# TEST 3: base weights are frozen
for p in lora2.base_linear.parameters():
    assert not p.requires_grad, 'Base linear params should be frozen'
print('PASS: base frozen')

# %%
# TEST 4: only LoRA params are trainable
trainable = [n for n, p in lora2.named_parameters() if p.requires_grad]
assert len(trainable) == 2, f'Expected 2 trainable params (A, B), got {len(trainable)}: {trainable}'
print(f'PASS: trainable params = {trainable}')
