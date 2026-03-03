# Drill 7 — The Pre-Norm Transformer Decoder Block

## Why This Matters

EvoCUA-8B (built on the Qwen3-VL backbone) is literally a **stack of these blocks**.
Every decoder-only LLM — GPT-2, LLaMA, Mistral, Qwen — is the same idea:

    Token Embeddings + Positional Info
          ↓
    TransformerBlock × N   ← THIS is what we're building today
          ↓
    LayerNorm → Linear → logits

Understand this one block and you understand the **entire architecture**.
The rest is just stacking and plumbing.

## Intuition: The Pre-Norm Transformer Block

A pre-norm decoder block has **two sublayers**, each wrapped with a **LayerNorm before** and a **residual connection around**:

```
x = x + MultiHeadAttention(LayerNorm(x))   # sublayer 1: self-attention
x = x + FFN(LayerNorm(x))                  # sublayer 2: feed-forward
```

The FFN (feed-forward network) is a simple **expand → activate → contract** pattern:

```
Linear(D, 4D) → GELU → Linear(4D, D)
```

That's it. Every modern LLM block follows this template.
Let's build each piece, then assemble the full block.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
print(f"PyTorch {torch.__version__}")
```

## Building Block 1: Residual Connections

The residual connection is deceptively simple: `output = x + f(x)`.

Why does this matter so much? Think about the **gradient** during backprop:

```
output = x + f(x)
dL/dx  = dL/doutput * (1 + df/dx)
                       ↑
                    identity path!
```

No matter how bad `df/dx` gets (vanishing, exploding), the gradient **always has the +1 identity path** to flow through. This is the "gradient highway" that lets us train networks with 100+ layers.

Without residuals, gradients must pass through every layer's transformation — they shrink exponentially and deep networks can't learn.

```python
# Residual connection in action
D = 64
x = torch.randn(2, 10, D)  # (batch, seq_len, d_model)

# Some arbitrary transformation (stand-in for attention or FFN)
f = nn.Linear(D, D)

# WITHOUT residual — information can be destroyed
out_no_res = f(x)

# WITH residual — original signal is always preserved
out_res = x + f(x)

# The residual output is "x plus a learned correction"
correction = f(x)
print(f"Input norm:      {x.norm():.2f}")
print(f"Correction norm: {correction.norm():.2f}")
print(f"Output norm:     {out_res.norm():.2f}")
print(f"\nThe correction is ADDED to the input, not replacing it.")
print(f"This means each layer only needs to learn a small delta.")
```

## Building Block 2: Pre-Norm vs Post-Norm

There are two ways to combine LayerNorm with residual connections:

**Post-norm** (original "Attention Is All You Need"):
```
x = LayerNorm(x + f(x))   # norm AFTER the residual add
```

**Pre-norm** (modern standard — GPT-2, LLaMA, Qwen, etc.):
```
x = x + f(LayerNorm(x))   # norm BEFORE the sublayer
```

Why did the field switch to pre-norm?

1. **More stable gradients** — the residual path is completely clean (no norm in the way)
2. **Easier to train** — often works without learning rate warmup
3. **Better for deep models** — gradients flow unchanged through the residual stream

The tradeoff: Pre-norm typically needs a **final LayerNorm** after the last block (since the residual stream is never normalized otherwise).

```python
# Side-by-side comparison
D = 64
x = torch.randn(2, 10, D)
f = nn.Linear(D, D)
norm = nn.LayerNorm(D)

# Post-norm: LayerNorm wraps the residual sum
post_norm_out = norm(x + f(x))

# Pre-norm: LayerNorm is INSIDE the sublayer, residual is CLEAN
pre_norm_out = x + f(norm(x))

print("Post-norm: x = LayerNorm(x + f(x))")
print(f"  Output norm: {post_norm_out.norm():.2f}")
print(f"  Output mean: {post_norm_out.mean():.4f}")
print()
print("Pre-norm: x = x + f(LayerNorm(x))")
print(f"  Output norm: {pre_norm_out.norm():.2f}")
print(f"  Output mean: {pre_norm_out.mean():.4f}")
print()
print("Key difference: in pre-norm, the gradient through the")
print("residual path is EXACTLY the identity — no LayerNorm to")
print("pass through. This makes deep networks much easier to train.")
```

## Building Block 3: The Feed-Forward Network (FFN)

The FFN in every transformer block follows a simple pattern:

```
Linear(D → 4D)  →  GELU  →  Linear(4D → D)
   expand         activate      contract
```

**Why expand by 4x?** The attention sublayer mixes information *across positions*. But each position's representation needs to be individually transformed too. The 4x expansion gives each position a bigger "workspace" to compute features before projecting back down.

**Why GELU?** It's a smooth approximation of ReLU that allows small negative values through. Most modern LLMs use GELU or its variants (SiLU/Swish). The smoothness helps optimization.

```python
# The FFN: expand → activate → contract
D = 64

ffn = nn.Sequential(
    nn.Linear(D, 4 * D),   # expand: D → 4D
    nn.GELU(),              # smooth activation
    nn.Linear(4 * D, D),   # contract: 4D → D
)

x = torch.randn(2, 10, D)
out = ffn(x)

print(f"Input shape:  {x.shape}")          # (2, 10, 64)
print(f"Output shape: {out.shape}")         # (2, 10, 64) — same!
print(f"\nInside the FFN:")
print(f"  After expand:   (2, 10, {4*D})")
print(f"  After GELU:     (2, 10, {4*D})")
print(f"  After contract: (2, 10, {D})")
print(f"\nFFN parameters: {sum(p.numel() for p in ffn.parameters()):,}")
print(f"  = D*4D + 4D + 4D*D + D = 2*D*4D + 5D = {2*D*4*D + 5*D:,}")

# GELU vs ReLU visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x_vals = torch.linspace(-4, 4, 200)
relu_vals = F.relu(x_vals)
gelu_vals = F.gelu(x_vals)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(x_vals.numpy(), relu_vals.numpy(), label='ReLU', linewidth=2)
ax.plot(x_vals.numpy(), gelu_vals.numpy(), label='GELU', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Input')
ax.set_ylabel('Output')
ax.set_title('GELU vs ReLU — GELU allows small negative values through')
ax.legend()
plt.tight_layout()
plt.savefig('gelu_vs_relu.png', dpi=100)
plt.show()
print("GELU is smooth everywhere — no sharp corner at 0 like ReLU.")
print("This smoothness helps gradient-based optimization.")
```

## Building Block 4: The Complete Pre-Norm Decoder Block

Now we assemble everything into the full block:

```
┌─────────────────────────────────────┐
│         TransformerBlock            │
│                                     │
│  ┌──────────┐                       │
│  │ LayerNorm │──→ MultiHeadAttn ──┐ │
│  └──────────┘    (causal mask)    │ │
│       ↑                           │ │
│       x ─────────────────────── + ←┘ │  ← residual connection 1
│       │                           │ │
│       ↓                           ↓ │
│  ┌──────────┐                       │
│  │ LayerNorm │──→ FFN ───────────┐ │
│  └──────────┘                    │ │
│       ↑                          │ │
│       x ────────────────────── + ←┘ │  ← residual connection 2
│                                     │
└─────────────────────────────────────┘
```

Two sublayers, each with pre-norm + residual. That's it.

```python
# Step-by-step forward pass through a pre-norm block
D = 64
n_heads = 4
seq_len = 10
batch = 2

# Components
norm1 = nn.LayerNorm(D)
attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
norm2 = nn.LayerNorm(D)
ffn = nn.Sequential(
    nn.Linear(D, 4 * D),
    nn.GELU(),
    nn.Linear(4 * D, D),
)

# Input
x = torch.randn(batch, seq_len, D)

# Causal mask: prevent attending to future positions
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

# ── Sublayer 1: Self-Attention with Pre-Norm + Residual ──
x_norm = norm1(x)                                        # pre-norm
attn_out, attn_weights = attn(                           # self-attention
    x_norm, x_norm, x_norm,
    attn_mask=causal_mask,
    is_causal=False  # we provide mask explicitly
)
x = x + attn_out                                         # residual

print(f"After sublayer 1 (attention): {x.shape}")

# ── Sublayer 2: FFN with Pre-Norm + Residual ──
x_norm = norm2(x)                                        # pre-norm
ffn_out = ffn(x_norm)                                    # feed-forward
x = x + ffn_out                                          # residual

print(f"After sublayer 2 (FFN):       {x.shape}")
print(f"\nShape is preserved throughout: (batch, seq_len, D) = ({batch}, {seq_len}, {D})")
```

## The Complete TransformerBlock Class

Let's wrap everything into a clean, reusable `nn.Module`:

```python
class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block.
    
    Architecture:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model  # default: 4x expansion
        
        # Sublayer 1: Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Sublayer 2: Feed-Forward Network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        # Sublayer 1: Pre-Norm → Attention → Residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
        )
        x = x + self.dropout(attn_out)
        
        # Sublayer 2: Pre-Norm → FFN → Residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x

# Test it
D, H = 64, 4
block = TransformerBlock(d_model=D, n_heads=H)

x = torch.randn(2, 10, D)
causal_mask = torch.triu(torch.ones(10, 10), diagonal=1).bool()

out = block(x, causal_mask)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"\nBlock parameters: {sum(p.numel() for p in block.parameters()):,}")
print(f"  Norm1:     {2 * D:,}    (scale + bias)")
print(f"  Attention: {4 * D * D + 4 * D:,}  (Q, K, V, Out projections)")
print(f"  Norm2:     {2 * D:,}    (scale + bias)")
print(f"  FFN:       {D * 4*D + 4*D + 4*D * D + D:,}  (expand + contract)")
```

## BONUS: Stacking Blocks into a GPT-Like Model

A full decoder-only language model is just:
1. Token embedding + positional embedding
2. N transformer blocks (stacked)
3. Final LayerNorm + linear head → logits

Let's build a mini version:

```python
class MiniGPT(nn.Module):
    """A minimal GPT-like decoder-only transformer.
    
    Architecture:
        Embedding → [TransformerBlock × N] → LayerNorm → Linear → logits
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm + output head
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: share embedding and output weights
        self.head.weight = self.token_emb.weight
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        
        # Embeddings
        tok = self.token_emb(token_ids)                      # (B, T, D)
        pos = self.pos_emb(torch.arange(T, device=token_ids.device))  # (T, D)
        x = self.drop(tok + pos)                             # (B, T, D)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=token_ids.device), diagonal=1
        ).bool()
        
        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final norm + project to vocab
        x = self.final_norm(x)       # pre-norm arch needs this final norm!
        logits = self.head(x)        # (B, T, vocab_size)
        
        return logits

# Build a small model
model = MiniGPT(
    vocab_size=1000,
    d_model=128,
    n_heads=4,
    n_layers=4,
    max_seq_len=256,
)

# Forward pass with random tokens
tokens = torch.randint(0, 1000, (2, 32))  # batch=2, seq_len=32
logits = model(tokens)

total_params = sum(p.numel() for p in model.parameters())
print(f"Input tokens:  {tokens.shape}")
print(f"Output logits: {logits.shape}")
print(f"\nModel structure:")
print(f"  Vocab size:    1,000")
print(f"  D_model:       128")
print(f"  N_heads:       4")
print(f"  N_layers:      4")
print(f"  Total params:  {total_params:,}")
print(f"\nFor comparison, EvoCUA-8B has ~8 BILLION parameters")
print(f"Same architecture, just bigger numbers!")

# Verify output is valid logit distribution
probs = F.softmax(logits[0, -1], dim=-1)
print(f"\nLast token probabilities sum to: {probs.sum():.4f}")
print(f"Top-5 predicted token IDs: {probs.topk(5).indices.tolist()}")

# Let's trace the shapes through the whole model
print("=== Shape Trace Through MiniGPT ===")
print(f"{'Layer':<35} {'Shape':<25}")
print("=" * 60)
print(f"{'Input token IDs':<35} {str(tokens.shape):<25}")
print(f"{'Token embedding':<35} {'(2, 32, 128)':<25}")
print(f"{'+ Position embedding':<35} {'(2, 32, 128)':<25}")
print()
for i in range(4):
    print(f"{'  Block ' + str(i) + ': LayerNorm → Attn':<35} {'(2, 32, 128)':<25}")
    print(f"{'  Block ' + str(i) + ': LayerNorm → FFN':<35} {'(2, 32, 128)':<25}")
print()
print(f"{'Final LayerNorm':<35} {'(2, 32, 128)':<25}")
print(f"{'Linear head → logits':<35} {'(2, 32, 1000)':<25}")
print()
print("Notice: the shape is (B, T, D) ALL THE WAY THROUGH.")
print("Only the final linear head changes D → vocab_size.")
```

## Your Turn

Try these exercises:

1. **Change the FFN expansion ratio** — modify `d_ff` to use 2x or 8x expansion instead of 4x. How does parameter count change?
2. **Add a SwiGLU FFN** — LLaMA/Qwen use SwiGLU instead of GELU. Replace `Linear → GELU → Linear` with `(Linear × SiLU(Linear)) → Linear` (a gated variant).
3. **Stack more layers** — increase `n_layers` to 12 or 24. How does parameter count scale?
4. **Remove residual connections** — try training without them. Watch what happens to the gradients.

```python
# Your experiments here!
```

## Tests

Run these to verify your TransformerBlock is correct:

```python
# ── Test Suite ──
def test_transformer_block():
    D, H, T, B = 64, 4, 16, 2
    block = TransformerBlock(d_model=D, n_heads=H, dropout=0.0)
    block.eval()  # disable dropout for deterministic testing
    
    x = torch.randn(B, T, D)
    causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    
    # Test 1: Output shape
    out = block(x, causal_mask)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    print("✓ Test 1 passed: output shape is correct")
    
    # Test 2: Output differs from input (block does something)
    assert not torch.allclose(x, out), "Output should differ from input"
    print("✓ Test 2 passed: block transforms the input")
    
    # Test 3: Residual connection works (output in neighborhood of input)
    # With random init, correction should be smaller than a completely random tensor
    diff = (out - x).norm() / x.norm()
    assert diff < 10.0, f"Residual diff too large: {diff}"
    print(f"✓ Test 3 passed: residual connection active (relative diff = {diff:.2f})")
    
    # Test 4: Deterministic in eval mode
    out2 = block(x, causal_mask)
    assert torch.allclose(out, out2), "Should be deterministic in eval mode"
    print("✓ Test 4 passed: deterministic in eval mode")
    
    # Test 5: Works without mask (for non-causal uses)
    out_no_mask = block(x)
    assert out_no_mask.shape == (B, T, D), "Should work without mask"
    print("✓ Test 5 passed: works without causal mask")
    
    # Test 6: Pre-norm structure check — norm is applied BEFORE sublayers
    # Verify norm1 and norm2 exist and are LayerNorm
    assert isinstance(block.norm1, nn.LayerNorm), "norm1 should be LayerNorm"
    assert isinstance(block.norm2, nn.LayerNorm), "norm2 should be LayerNorm"
    print("✓ Test 6 passed: pre-norm structure verified")
    
    print("\n✅ All tests passed!")

test_transformer_block()

# Test MiniGPT too
def test_mini_gpt():
    model = MiniGPT(vocab_size=100, d_model=32, n_heads=2, n_layers=2)
    model.eval()
    tokens = torch.randint(0, 100, (1, 8))
    logits = model(tokens)
    
    assert logits.shape == (1, 8, 100), f"Expected (1, 8, 100), got {logits.shape}"
    print("✓ MiniGPT test passed: correct output shape")
    
    # Weight tying check
    assert model.head.weight is model.token_emb.weight, "Weights should be tied"
    print("✓ MiniGPT test passed: weight tying verified")
    
    print("\n✅ All MiniGPT tests passed!")

test_mini_gpt()
```

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 7: Transformer Block (Pre-Norm)
Implement a single pre-norm transformer block.

Architecture:
  x = x + attn(norm1(x))    # self-attention with residual
  x = x + ffn(norm2(x))     # feed-forward with residual

Components:
  - norm1, norm2: nn.LayerNorm(d_model)
  - attn: nn.MultiheadAttention(d_model, num_heads, batch_first=True)
  - ffn: Sequential(Linear(D, 4D), GELU, Linear(4D, D))

Forward should apply a causal mask so positions can only attend to past.
You may use nn.LayerNorm, nn.MultiheadAttention, nn.Linear, nn.GELU.

```python
import torch
import torch.nn as nn
```

```python
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
```

```python
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
```
