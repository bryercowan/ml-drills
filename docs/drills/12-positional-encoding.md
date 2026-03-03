# Drill 12 — Sinusoidal Positional Encoding

## Why This Matters

Attention is permutation-invariant — it treats tokens as a **set**, not a sequence. Without positional encoding, "the cat sat on the mat" and "mat the on sat cat the" produce identical attention patterns.

Sinusoidal positional encoding gives each position a unique fingerprint that the model can use to reason about order and relative distance. This is the original scheme from "Attention Is All You Need" and still shows up in interviews constantly.

## Intuition

Each position gets a vector of `d_model` dimensions. We fill it with **sin and cos waves at different frequencies**:
- Low-frequency waves (slow oscillation) → capture coarse position info
- High-frequency waves (fast oscillation) → capture fine-grained position info

Think of it like a clock: the hour hand, minute hand, and second hand each oscillate at different rates. Together they uniquely identify any time. Same idea here — stacking waves at different frequencies uniquely identifies any position.

```python
import torch
import matplotlib.pyplot as plt
import math
```

## Building Block 1: Sin at Different Frequencies

Let's see what `sin(pos / frequency)` looks like for different frequency scales.

```python
positions = torch.arange(50).float()

fig, axes = plt.subplots(1, 3, figsize=(14, 3))
for idx, freq in enumerate([1.0, 10.0, 100.0]):
    axes[idx].plot(positions.numpy(), torch.sin(positions / freq).numpy())
    axes[idx].set_title(f'sin(pos / {freq:.0f})')
    axes[idx].set_xlabel('Position')
plt.tight_layout()
plt.show()

# Notice: higher frequency divisor = slower oscillation = coarser position info
```

## Building Block 2: The div_term Formula

The paper uses exponentially spaced frequencies: `10000^(2i / d_model)`

This means dimension 0 oscillates fast (period ≈ 2π) and the last dimension oscillates very slowly (period ≈ 2π × 10000). The log-space trick keeps computation stable:

```
div_term = exp(2i * (-log(10000) / d_model))  =  1 / 10000^(2i/d_model)
```

```python
d_model = 16

i = torch.arange(0, d_model, 2).float()  # [0, 2, 4, ..., d_model-2]

# These two are equivalent:
div_term_direct = 10000.0 ** (i / d_model)
div_term_log    = torch.exp(i * (-math.log(10000.0) / d_model))

print("Direct (denominator): ", div_term_direct[:4])
print("Log-space (multiplier):", div_term_log[:4])
print("Product (should be 1):", (div_term_direct * div_term_log)[:4])
```

## Building Block 3: Even Dims → Sin, Odd Dims → Cos

For each position `pos` and dimension `i`:
- `PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))`
- `PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))`

This interleaving gives each position a unique fingerprint and has a nice property: relative positions can be represented as linear transformations of the encoding.

```python
max_len, d_model = 10, 8

pe = torch.zeros(max_len, d_model)
pos = torch.arange(max_len).unsqueeze(1).float()        # (max_len, 1)
div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

pe[:, 0::2] = torch.sin(pos * div)   # even columns
pe[:, 1::2] = torch.cos(pos * div)   # odd columns

print("PE shape:", pe.shape)
print("Position 0:", pe[0].tolist())
print("Position 1:", pe[1].tolist())
```

## Complete Implementation

```python
def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """
    Returns positional encoding matrix of shape (max_len, d_model).
    Values are in [-1, 1] (sin/cos range).
    """
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()                              # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

# Quick visualization
pe = sinusoidal_pe(50, 64)
plt.figure(figsize=(10, 4))
plt.imshow(pe.numpy(), aspect='auto', cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Sinusoidal Positional Encoding (50 positions, 64 dims)')
plt.colorbar()
plt.show()
```

## Your Turn

Implement `sinusoidal_pe` from scratch without looking above.

```python
def sinusoidal_pe_scratch(max_len: int, d_model: int) -> torch.Tensor:
    # YOUR CODE HERE
    pass
```

## Tests

```python
pe = sinusoidal_pe(100, 64)

# Shape check
assert pe.shape == (100, 64), f"Expected (100, 64), got {pe.shape}"

# All values in [-1, 1]
assert pe.min() >= -1.0 and pe.max() <= 1.0, "Values must be in [-1, 1]"

# Position 0, even dims should be sin(0) = 0
assert torch.allclose(pe[0, 0::2], torch.zeros(32), atol=1e-6), "sin(0) should be 0"

# Position 0, odd dims should be cos(0) = 1
assert torch.allclose(pe[0, 1::2], torch.ones(32), atol=1e-6), "cos(0) should be 1"

# Each position should be unique
for i in range(99):
    assert not torch.allclose(pe[i], pe[i+1], atol=1e-4), f"Positions {i} and {i+1} are not unique"

print("All tests passed!")
```

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 12: Sinusoidal Positional Encoding
Implement the fixed positional encoding from "Attention Is All You Need".
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

```python
import torch
import math
```

```python
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
```

## Tests

```python
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
```
