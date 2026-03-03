# Drill 05 — LoRA: Low-Rank Adaptation from Scratch

## Why This Matters — A Real Discovery

You LoRA-tuned **EvoCUA-8B** (Meituan's GUI agent) with `r=64`, `alpha=128`,
targeting the attention projections: **q_proj, k_proj, v_proj, o_proj**.

The key discovery: when you also included the MLP layers (`gate_proj`, `up_proj`,
`down_proj`), **spatial grounding collapsed**. The model forgot WHERE to click
on the screen. Why? Because Meituan had carefully trained those MLP weights on
82K GUI interaction trajectories — they encode spatial-geometric reasoning about
UI layouts. Overwriting them destroyed that knowledge.

**Attention-only LoRA** preserved the spatial "where to click" knowledge in the
MLPs while adapting "what to attend to" in the attention heads. This is the
surgical precision that makes LoRA powerful: you choose WHICH weight matrices
to adapt and leave the rest frozen.

This notebook builds LoRA from scratch so you understand every piece.

## The Core Intuition

**Full fine-tuning** updates every element of a weight matrix $W \in \mathbb{R}^{out \times in}$.
That's $out \times in$ parameters per layer.

**LoRA's insight**: the *change* to $W$ during fine-tuning is low-rank.
We don't need a full-rank update — a rank-$r$ approximation is enough:

$$\Delta W = B \cdot A$$

where:
- $A \in \mathbb{R}^{r \times in}$ — projects input DOWN to rank $r$
- $B \in \mathbb{R}^{out \times r}$ — projects back UP to output dim

The adapted weight becomes:

$$W_{new} = W_{frozen} + \frac{\alpha}{r} \cdot B \cdot A$$

**Critical design choice**: $B$ is initialized to **zeros**.
At the start of training, $\Delta W = 0 \cdot A = 0$, so the model
behaves exactly like the pre-trained base. Training gradually learns the
low-rank update. This means LoRA is a **safe** starting point — you
can only improve from the base model's behavior.

The scaling factor $\alpha/r$ controls the magnitude of the update
relative to the frozen weights. Think of it as a "learning rate multiplier"
for the adapter path.

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
print("PyTorch version:", torch.__version__)
```

## Building Block 1: Shapes & Parameter Savings

For a weight matrix $W \in \mathbb{R}^{out \times in}$:
- **Full fine-tuning**: $out \times in$ trainable parameters
- **LoRA**: $A$ has $r \times in$ params, $B$ has $out \times r$ params → total $r \times (in + out)$

Compression ratio = $\frac{out \times in}{r \times (in + out)}$

For a typical LLM linear layer with `in=out=4096` and `r=64`:

```python
# BB1: Parameter count comparison
d_in, d_out, r = 4096, 4096, 64

full_params = d_in * d_out
lora_params = r * d_in + r * d_out   # A is (r, d_in), B is (d_out, r)

print(f"Full fine-tuning:  {full_params:>12,} params  ({full_params/1e6:.1f}M)")
print(f"LoRA (r={r}):       {lora_params:>12,} params  ({lora_params/1e6:.3f}M)")
print(f"Compression ratio: {full_params / lora_params:.1f}x")
print(f"LoRA is {lora_params/full_params*100:.2f}% of full params")

# Verify shapes
A = torch.randn(r, d_in)
B = torch.randn(d_out, r)
delta_W = B @ A
print(f"\nA shape: {A.shape}")          # (64, 4096)
print(f"B shape: {B.shape}")          # (4096, 64)
print(f"delta_W shape: {delta_W.shape}")  # (4096, 4096) - same as W!
assert delta_W.shape == (d_out, d_in), "delta_W must match W shape"
print("✓ delta_W has the same shape as the original weight matrix")
```

## Building Block 2: Zero Initialization of B

$B$ is initialized to **all zeros**. This means at initialization:

$$\Delta W = B \cdot A = \mathbf{0} \cdot A = \mathbf{0}$$

The adapter starts as a **no-op**. The model's output is identical to the
frozen base model before any training happens. This is a crucial property —
it means inserting LoRA adapters doesn't break the pre-trained model.

```python
# BB2: Zero init means adapter starts as identity (no-op)
d_in, d_out, r = 4096, 4096, 64

# LoRA init: A gets random (Kaiming), B gets zeros
A = torch.randn(r, d_in) * (1 / d_in**0.5)  # Kaiming-like
B = torch.zeros(d_out, r)                     # ZEROS!

delta_W = B @ A
print(f"B is all zeros: {(B == 0).all().item()}")
print(f"delta_W max absolute value: {delta_W.abs().max().item()}")
print(f"delta_W is all zeros: {(delta_W == 0).all().item()}")

# Now pass some data through
x = torch.randn(2, d_in)  # batch of 2
lora_out = (x @ A.T) @ B.T
print(f"\nInput shape: {x.shape}")
print(f"LoRA output shape: {lora_out.shape}")
print(f"LoRA output is all zeros: {(lora_out == 0).all().item()}")
print("✓ At init, LoRA contributes nothing — base model is unchanged")
```

## Building Block 3: The Forward Pass

During inference, given input $x$:

$$y = x W_{frozen}^T + x A^T B^T \cdot \frac{\alpha}{r}$$

Or equivalently: `base_output + lora_output * scaling`

The LoRA path creates a **rank-r bottleneck**:
1. $x A^T$: project from $in$ → $r$ dimensions (compress)
2. $(x A^T) B^T$: project from $r$ → $out$ dimensions (expand)

The information must squeeze through that $r$-dimensional bottleneck.
With $r=64$ and $d=4096$, that's **64x compression** of the update signal.

```python
# BB3: Forward pass with the rank-r bottleneck
d_in, d_out, r = 512, 512, 8  # smaller for visualization
alpha = 16
scaling = alpha / r

# Simulated frozen base layer
W_frozen = torch.randn(d_out, d_in) * 0.02

# LoRA matrices (pretend some training has happened, so B != 0)
A = torch.randn(r, d_in) * 0.01
B = torch.randn(d_out, r) * 0.01  # non-zero after training

# Input
x = torch.randn(4, d_in)  # batch of 4

# Base path: standard linear
base_out = x @ W_frozen.T
print(f"Base output shape: {base_out.shape}")

# LoRA path: rank-r bottleneck
compressed = x @ A.T          # (4, 512) @ (512, 8) -> (4, 8)  ← bottleneck!
expanded = compressed @ B.T   # (4, 8) @ (8, 512) -> (4, 512)
lora_out = expanded * scaling

print(f"After compression (x @ A.T): {compressed.shape}  ← rank-{r} bottleneck")
print(f"After expansion (... @ B.T): {expanded.shape}")

# Combined output
y = base_out + lora_out
print(f"\nFinal output shape: {y.shape}")
print(f"Base output norm:  {base_out.norm():.4f}")
print(f"LoRA output norm:  {lora_out.norm():.4f}")
print(f"LoRA contribution: {lora_out.norm() / base_out.norm() * 100:.2f}% of base norm")
```

## Building Block 4: The Scaling Factor α/r

The scaling factor `alpha / r` serves a specific purpose:

- **α (alpha)**: a hyperparameter you set once (like 128) and typically don't change
- **r (rank)**: the bottleneck dimension you might experiment with (16, 32, 64...)

**Why not just tune the learning rate?** Because α/r **decouples rank selection
from learning dynamics**. When you increase r, the raw output magnitude of B@A
tends to grow. Dividing by r compensates, so you can change r without retuning
the learning rate. α is the "effective strength" of the adapter.

**Your setting**: `r=64, alpha=128` → scaling = `128/64 = 2.0`

This means the LoRA update is amplified 2x relative to the raw B@A product.
A scaling > 1 means you're telling the optimizer "the adapter path matters —
let it have a strong effect.":

```python
# BB4: How scaling affects the adapter contribution
d_in, d_out = 512, 512
x = torch.randn(4, d_in)

# Same A and B, different scaling
A = torch.randn(8, d_in) * 0.01
B = torch.randn(d_out, 8) * 0.01
raw_lora = (x @ A.T) @ B.T

configs = [
    ("r=8,  α=8",   8 / 8),    # scaling = 1.0
    ("r=8,  α=16",  16 / 8),   # scaling = 2.0
    ("r=64, α=64",  64 / 64),  # scaling = 1.0 (but higher rank)
    ("r=64, α=128", 128 / 64), # scaling = 2.0 ← your EvoCUA config
]

print("Config           | Scaling | Output Norm")
print("-" * 48)
for name, s in configs:
    out = raw_lora * s
    print(f"{name:17s} | {s:7.2f} | {out.norm():.4f}")

print("\n→ With r=64, α=128: scaling = 2.0")
print("  This amplifies the adapter signal, giving it more influence")
print("  over the frozen base weights.")
```

## Building Block 5: Your EvoCUA Numbers

Let's compute the actual parameter counts for EvoCUA-8B (Qwen2-VL architecture)
to see why attention-only vs attention+MLP is such a different story.

EvoCUA-8B has 32 transformer layers. Each layer has:
- **Attention**: q_proj, k_proj, v_proj, o_proj — each is (4096, 4096) for a 8B-class model
- **MLP**: gate_proj (11008, 4096), up_proj (11008, 4096), down_proj (4096, 11008)

With r=64, each LoRA adapter adds `r × (in + out)` parameters.

```python
# BB5: Real parameter counts from your EvoCUA-8B work
r = 64
n_layers = 32

# --- Attention projections (q, k, v, o) ---
# Simplified: each is roughly 4096 x 4096 for 8B models
# (actual Qwen2-VL may have GQA, but this captures the scale)
attn_d_in, attn_d_out = 4096, 4096
lora_per_attn_proj = r * (attn_d_in + attn_d_out)
n_attn_projs = 4  # q, k, v, o
attn_lora_total = lora_per_attn_proj * n_attn_projs * n_layers

print("=== Attention-Only LoRA (your working config) ===")
print(f"Per projection: r × (in + out) = {r} × ({attn_d_in} + {attn_d_out}) = {lora_per_attn_proj:,}")
print(f"4 projections × {n_layers} layers = {n_attn_projs * n_layers} adapters")
print(f"Total: {attn_lora_total:,} = {attn_lora_total/1e6:.1f}M trainable params")

# --- MLP projections (gate, up, down) ---
mlp_configs = [
    ("gate_proj", 4096, 11008),
    ("up_proj",   4096, 11008),
    ("down_proj", 11008, 4096),
]
mlp_lora_total = 0
print(f"\n=== MLP LoRA (what broke spatial grounding) ===")
for name, d_in, d_out in mlp_configs:
    params = r * (d_in + d_out)
    mlp_lora_total += params * n_layers
    print(f"  {name}: r × ({d_in} + {d_out}) = {params:,} per layer")
mlp_lora_total_display = mlp_lora_total

print(f"3 projections × {n_layers} layers: {mlp_lora_total:,} = {mlp_lora_total/1e6:.1f}M params")

# --- Combined ---
combined = attn_lora_total + mlp_lora_total
print(f"\n=== Comparison ===")
print(f"Attention-only:    {attn_lora_total/1e6:>6.1f}M params  ✓ preserved spatial grounding")
print(f"MLP-only:          {mlp_lora_total/1e6:>6.1f}M params")
print(f"Attention + MLP:   {combined/1e6:>6.1f}M params  ✗ destroyed spatial grounding")
print(f"\nAttention+MLP is {combined/attn_lora_total:.1f}x more params than attention-only")
print(f"\nLesson: More trainable params ≠ better. The MLP weights encoded")
print(f"spatial reasoning from 82K GUI trajectories. LoRA on those layers")
print(f"overwrote that knowledge, breaking WHERE the model clicks.")
```

## Complete Implementation: LoRALinear

Now let's put it all together into a proper `nn.Module`.
This is the complete LoRA adapter that wraps any existing `nn.Linear`:

```python
class LoRALinear(nn.Module):
    """A Linear layer with a frozen base weight and a trainable low-rank adapter.

    Forward: y = x @ W_frozen.T + (x @ A.T @ B.T) * (alpha / r)

    Args:
        base_linear: existing nn.Linear to wrap (its weight gets frozen)
        r: rank of the low-rank adapter
        alpha: scaling hyperparameter (controls adapter strength)
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        d_in = base_linear.in_features
        d_out = base_linear.out_features

        # Freeze the base weight (and bias if present)
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad_(False)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(False)

        # LoRA matrices
        # A: (r, d_in) — Kaiming uniform init
        self.A = nn.Parameter(torch.empty(r, d_in))
        nn.init.kaiming_uniform_(self.A)

        # B: (d_out, r) — ZERO init (critical: adapter starts as no-op)
        self.B = nn.Parameter(torch.zeros(d_out, r))

        # Scaling factor
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (frozen, no gradient)
        base_out = self.base_linear(x)

        # LoRA path (trainable)
        # x: (..., d_in)
        # x @ A.T: (..., r)       ← compress to rank r
        # ... @ B.T: (..., d_out)  ← expand back
        lora_out = (x @ self.A.T) @ self.B.T * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        d_in = self.base_linear.in_features
        d_out = self.base_linear.out_features
        r = self.A.shape[0]
        total = r * (d_in + d_out)
        return f"in={d_in}, out={d_out}, r={r}, α={self.scaling * r:.0f}, lora_params={total:,}"

print("✓ LoRALinear class defined")
print(LoRALinear.__doc__)
```

## Your Turn

Implement `LoRALinear` from scratch without looking at the cell above.
Key things to remember:
- A is (r, d_in), initialized with Kaiming
- B is (d_out, r), initialized to ZEROS
- scaling = alpha / r
- Forward: base_out + (x @ A.T) @ B.T * scaling
- Freeze the base linear's parameters

```python
# YOUR IMPLEMENTATION HERE
class MyLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        pass  # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # TODO
```

## Tests

These tests verify two critical properties:
1. **Shape correctness**: output shape matches what a regular linear would produce
2. **No-op at init**: because B=0, the LoRA layer produces EXACTLY the same output as the base linear

```python
# --- Test Suite ---
torch.manual_seed(0)

# Create a base linear layer
base = nn.Linear(256, 128)

# Wrap it with LoRA
lora_layer = LoRALinear(base, r=16, alpha=32.0)

# Test input
x = torch.randn(4, 256)

# Test 1: Shape
y = lora_layer(x)
assert y.shape == (4, 128), f"Expected (4, 128), got {y.shape}"
print("✓ Test 1 PASSED: output shape is correct:", y.shape)

# Test 2: Output matches base at init (B=0 → no-op)
base_out = base(x)
assert torch.allclose(y, base_out, atol=1e-6), \
    f"At init, LoRA output should match base! Max diff: {(y - base_out).abs().max()}"
print("✓ Test 2 PASSED: at init (B=0), LoRA output == base output")
print(f"  Max difference: {(y - base_out).abs().max().item():.2e}")

# Test 3: Base weights are frozen
assert not base.weight.requires_grad, "Base weight should be frozen!"
print("✓ Test 3 PASSED: base weights are frozen")

# Test 4: LoRA params are trainable
assert lora_layer.A.requires_grad, "A should be trainable!"
assert lora_layer.B.requires_grad, "B should be trainable!"
print("✓ Test 4 PASSED: A and B are trainable")

# Test 5: Parameter count
n_trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
expected = 16 * 256 + 128 * 16  # r*d_in + d_out*r
assert n_trainable == expected, f"Expected {expected} trainable params, got {n_trainable}"
print(f"✓ Test 5 PASSED: {n_trainable:,} trainable params (expected {expected:,})")
print(f"  vs {256*128:,} for full fine-tuning ({n_trainable/(256*128)*100:.1f}%)")

# Test 6: After "training" (manually set B != 0), output should change
lora_layer.B.data.normal_(0, 0.01)
y_after = lora_layer(x)
assert not torch.allclose(y_after, base_out, atol=1e-6), \
    "After B is non-zero, output should differ from base!"
print("✓ Test 6 PASSED: non-zero B produces different output (adapter is active)")

print("\n🎉 All tests passed! Your LoRA implementation is correct.")
```

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 5: LoRA (Low-Rank Adaptation)
Implement a LoRA wrapper around a frozen nn.Linear.
Core idea: output = base(x) + (x @ A^T) @ B^T * scaling
A is initialized small random, B is initialized ZERO.
At init, LoRA contributes nothing (B=0), so output == base output.
Steps:
1. Store base_linear and freeze it (requires_grad=False)
2. Create A: nn.Parameter of shape (r, in_features), init small random
3. Create B: nn.Parameter of shape (out_features, r), init ZERO
4. Compute scaling = alpha / r
5. Forward: base_out + (x @ A^T) @ B^T * scaling

```python
import torch
import torch.nn as nn
```

```python
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
```

```python
# TEST 1: shape check
base = nn.Linear(64, 128)
lora = MyLoRALinear(base, r=16, alpha=32)
x = torch.randn(2, 5, 64)
out = lora(x)
assert out.shape == (2, 5, 128), f'Wrong shape: {out.shape}'
print('PASS: shape check')
```

```python
# TEST 2: at init, output matches base (B=0 so LoRA adds nothing)
base2 = nn.Linear(64, 128)
lora2 = MyLoRALinear(base2, r=8, alpha=16)
x2 = torch.randn(3, 10, 64)
with torch.no_grad():
    base_out = base2(x2)
    lora_out = lora2(x2)
assert torch.allclose(base_out, lora_out, atol=1e-6), 'At init, LoRA output should match base output (B is zero)'
print('PASS: init equivalence')
```

```python
# TEST 3: base weights are frozen
for p in lora2.base_linear.parameters():
    assert not p.requires_grad, 'Base linear params should be frozen'
print('PASS: base frozen')
```

```python
# TEST 4: only LoRA params are trainable
trainable = [n for n, p in lora2.named_parameters() if p.requires_grad]
assert len(trainable) == 2, f'Expected 2 trainable params (A, B), got {len(trainable)}: {trainable}'
print(f'PASS: trainable params = {trainable}')
```
