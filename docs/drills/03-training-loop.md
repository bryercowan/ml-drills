# Drill 03 -- The PyTorch Training Loop

## Why This Matters

You've launched dozens of training runs -- SFT, GRPO, DPO, CNN policy training.
You've tweaked hyperparameters, watched loss curves, debugged NaN gradients.

But if someone handed you an empty `.py` file and said "write a training loop
from scratch, no frameworks" -- could you do it cold?

This drill makes sure the answer is **yes**. We'll build the loop piece by piece,
understand *why* the order matters, and then write the whole thing from memory.

## The 6-Step Cycle

Every PyTorch training step follows this exact order. **ORDER MATTERS.**

```
optimizer.zero_grad()          # 1. Clear old gradients
output = model(x)              # 2. Forward pass
loss = criterion(output, y)    # 3. Compute loss
loss.backward()                # 4. Backward pass (compute gradients)
clip_grad_norm_(...)           # 5. Clip gradients (optional but recommended)
optimizer.step()               # 6. Update weights
```

**Why this order?**
- `zero_grad` before `backward`: gradients ACCUMULATE by default. If you forget
  to zero, you're adding this batch's gradients on top of the last batch's.
- `forward` before `loss`: obvious -- you need predictions to compare against labels.
- `backward` before `clip`: you need gradients to exist before you can clip them.
- `clip` before `step`: clip the raw gradients, THEN use them to update weights.
- `step` last: applies the (clipped) gradients to update model parameters.

Swap any two of these and something breaks. Let's prove it.

## Building Block 1: Model + Data Setup

We'll use a simple MLP on fake classification data. Nothing fancy --
this is about the loop, not the architecture.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

torch.manual_seed(42)

# Fake dataset: 256 samples, 20 features, 5 classes
X = torch.randn(256, 20)
y = torch.randint(0, 5, (256,))

# Simple MLP
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(32, 5),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y values: {y.unique().tolist()}")
```

## Building Block 2: Forward Pass + Loss

The forward pass pushes data through the model. Cross-entropy loss combines
`log_softmax` + `nll_loss` in one step -- it expects **raw logits**, not probabilities.

A random 5-class classifier has expected loss = -ln(1/5) ~ 1.609.
If your initial loss is much higher, something is wrong with your data or model.

```python
# Step 2: Forward pass
output = model(X)
print(f"Output shape: {output.shape}")      # [256, 5] -- one logit per class
print(f"Sample logits: {output[0].detach()}")  # raw scores, NOT probabilities

# Step 3: Compute loss
loss = criterion(output, y)
print(f"\nLoss: {loss.item():.4f}")
print(f"Expected random loss: {-torch.log(torch.tensor(1/5)).item():.4f}")
print(f"Ratio to random: {loss.item() / (-torch.log(torch.tensor(1/5)).item()):.2f}x")
```

## Building Block 3: Backward Pass

`loss.backward()` computes dL/dw for every parameter with `requires_grad=True`.

**The accumulation gotcha**: PyTorch ADDS new gradients to existing ones.
This is useful for gradient accumulation across mini-batches, but deadly
if you forget to zero them out. Watch:

```python
# Check gradients BEFORE backward
first_layer = model[0]  # nn.Linear(20, 64)
print(f"Grad before backward: {first_layer.weight.grad}")

# Step 4: Backward pass
loss.backward()

grad_after_first = first_layer.weight.grad.clone()
print(f"Grad after 1st backward: norm = {first_layer.weight.grad.norm():.4f}")

# WARNING -- THE GOTCHA: do another forward+backward WITHOUT zeroing
output2 = model(X)
loss2 = criterion(output2, y)
loss2.backward()

grad_after_second = first_layer.weight.grad.clone()
print(f"Grad after 2nd backward: norm = {first_layer.weight.grad.norm():.4f}")
print(f"\nAre they equal? {torch.equal(grad_after_first, grad_after_second)}")
print(f"Second ~ 2x first? Ratio: {grad_after_second.norm() / grad_after_first.norm():.2f}")
print("\n--> Gradients ACCUMULATED. That's why zero_grad() is step 1.")
```

## Building Block 4: Optimizer Step + Zero Grad

`optimizer.step()` reads `.grad` from every parameter and updates the weights.
`optimizer.zero_grad()` sets all `.grad` back to zero (or None with set_to_none=True).

```python
# Snapshot weights before step
w_before = first_layer.weight.data.clone()

# Step 6: Update weights
optimizer.step()

w_after = first_layer.weight.data.clone()
weight_change = (w_after - w_before).norm()
print(f"Weight change (L2 norm): {weight_change:.6f}")
print(f"Not zero -> step did something: {weight_change > 0}")

# Step 1 (for next iteration): Zero gradients
optimizer.zero_grad()
print(f"\nGrad after zero_grad: {first_layer.weight.grad}")
# With default zero_grad(), grads are zeros
# With zero_grad(set_to_none=True), grads become None (slightly faster)

optimizer.zero_grad(set_to_none=True)
print(f"Grad after zero_grad(set_to_none=True): {first_layer.weight.grad}")
```

## Building Block 5: Gradient Clipping

Remember those GRPO runs with gradient norms around ~1500? That's why training
was unstable. `clip_grad_norm_` rescales all gradients so their combined norm
doesn't exceed a threshold (typically 1.0).

This goes AFTER `backward()` and BEFORE `step()` -- you need gradients to exist,
and you want to clip them before they're used to update weights.

```python
# Fresh forward/backward to get clean gradients
optimizer.zero_grad()
output = model(X)
loss = criterion(output, y)
loss.backward()

# Check gradient norm BEFORE clipping
total_norm_before = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
print(f"Gradient norm before clip: {total_norm_before:.4f}")

# Step 5: Clip gradients
max_norm = 1.0
total_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)

# Verify: recompute norm manually after clipping
manual_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
print(f"Gradient norm after clip:  {manual_norm:.4f}")
print(f"Max norm threshold:        {max_norm}")
print(f"Clipped to max_norm?       {manual_norm <= max_norm + 1e-6}")

optimizer.step()
```

## Building Block 6: Learning Rate Scheduler

The optimizer has a fixed learning rate. A scheduler adjusts it over time.
`CosineAnnealingLR` smoothly decays LR from initial value to near-zero,
following a cosine curve.

**Key**: `scheduler.step()` is called once per EPOCH (not per batch) for most
schedulers. It goes AFTER `optimizer.step()`.

```python
# Reset optimizer for clean demo
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

lrs = []
for epoch in range(20):
    # (training step would go here)
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

print("Epoch | Learning Rate")
print("-" * 25)
for i, lr in enumerate(lrs):
    bar = "#" * int(lr / max(lrs) * 30)
    print(f"  {i:2d}  | {lr:.6f}  {bar}")

print(f"\nStart LR: {lrs[0]:.6f}")
print(f"End LR:   {lrs[-1]:.6f}")
print(f"Ratio:    {lrs[0]/lrs[-1]:.0f}x decay")
```

## Building Block 7: Eval Mode + torch.no_grad()

Two separate things that both matter at inference time:

1. **`model.eval()`** -- changes BEHAVIOR of Dropout and BatchNorm:
   - Dropout: stops dropping (uses all neurons)
   - BatchNorm: uses running stats instead of batch stats

2. **`torch.no_grad()`** -- stops TRACKING gradients (saves memory + compute).
   Doesn't change model behavior, just tells autograd "don't record this."

You need BOTH for proper evaluation. And don't forget `model.train()` after!

```python
# Demonstrate eval vs train mode
model.train()
out_train1 = model(X[:4])
out_train2 = model(X[:4])
print("TRAIN MODE (dropout active):")
print(f"  Same input, same output? {torch.equal(out_train1, out_train2)}")

model.eval()
out_eval1 = model(X[:4])
out_eval2 = model(X[:4])
print("\nEVAL MODE (dropout disabled):")
print(f"  Same input, same output? {torch.equal(out_eval1, out_eval2)}")

# torch.no_grad() context manager
print("\nMemory/grad tracking:")
model.train()  # back to train mode

x_small = X[:4]
out_with_grad = model(x_small)
print(f"  With grad tracking:    requires_grad={out_with_grad.requires_grad}")

with torch.no_grad():
    out_no_grad = model(x_small)
    print(f"  Without grad tracking: requires_grad={out_no_grad.requires_grad}")

# Always switch back!
model.train()
print("\n--> Always pair model.eval() with model.train() when you're done evaluating.")
```

## The Complete Training Loop

All 6 steps together, in order, with evaluation. This is the function you
should be able to write from memory.

```python
def train(model, X, y, epochs=30, lr=1e-3, max_grad_norm=1.0, print_every=5):
    "Complete training loop with all components."
    # Setup
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # === THE 6-STEP CYCLE ===
        optimizer.zero_grad()                        # 1. Zero gradients
        output = model(X)                            # 2. Forward pass
        loss = criterion(output, y)                  # 3. Compute loss
        loss.backward()                              # 4. Backward pass
        grad_norm = clip_grad_norm_(                 # 5. Clip gradients
            model.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()                             # 6. Update weights
        scheduler.step()                             # (bonus) Update LR

        # === EVALUATION ===
        if epoch % print_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(dim=1)
                acc = (preds == y).float().mean()
            model.train()

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc.item():.3f} | Grad Norm: {grad_norm:.4f} | "
                  f"LR: {lr_now:.6f}")

    return model


# Rebuild model fresh
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(32, 5),
)

# Regenerate data
X = torch.randn(256, 20)
y = torch.randint(0, 5, (256,))

print("Training...")
print("=" * 70)
model = train(model, X, y, epochs=50, lr=1e-2, print_every=10)
print("=" * 70)
print("Done! Loss should be decreasing, accuracy should be increasing.")
```

## Your Turn

Write the training loop from scratch. No peeking above.

```python
def my_training_loop(model, X, y, epochs=30):
    """
    Write a complete training loop from scratch.

    Requirements:
    - Use Adam optimizer with lr=1e-3
    - Use CrossEntropyLoss
    - Include all 6 steps in the correct order:
      zero_grad -> forward -> loss -> backward -> clip -> step
    - Clip gradients to max_norm=1.0
    - Add a CosineAnnealingLR scheduler
    - Print loss and accuracy every 5 epochs
    - Use model.eval() + torch.no_grad() for accuracy computation
    - Remember to call model.train() after eval

    Test: loss should decrease from ~1.6 to <0.5 over 50 epochs.
    """
    pass  # YOUR CODE HERE


# Test it:
# torch.manual_seed(0)
# test_model = nn.Sequential(
#     nn.Linear(20, 64), nn.ReLU(),
#     nn.Linear(64, 5),
# )
# test_X = torch.randn(128, 20)
# test_y = torch.randint(0, 5, (128,))
# my_training_loop(test_model, test_X, test_y, epochs=50)
```

## Quick Check

No formal test -- just run the training and verify:
- Loss starts near 1.6 (random baseline for 5 classes)
- Loss decreases over epochs
- Accuracy increases over epochs
- No NaN values anywhere

If all of that checks out, you've got the training loop down cold.

---

## 🏋️ Practice — Implement It Yourself

Copy the stubs below into your editor. Implement without looking at the code above. Run the test cell to verify.

# Drill 3: Training Loop
Implement a complete training loop from scratch.
Create a small MLP, optimizer, scheduler, loss fn.
Train on synthetic data and verify loss decreases.
Steps:
1. Create nn.Sequential MLP (in -> hidden -> ReLU -> hidden -> out)
2. Create AdamW optimizer
3. Create CosineAnnealingLR scheduler
4. Create CrossEntropyLoss
5. Inner loop: zero_grad -> forward -> loss -> backward -> clip_grad_norm_ -> optimizer step
6. Scheduler step per epoch
7. Eval with torch.no_grad()

```python
import torch
import torch.nn as nn
```

```python
def my_train(input_dim=32, hidden_dim=64, num_classes=10,
             num_samples=500, num_epochs=20, lr=1e-3, batch_size=64):
    """Train a small MLP on synthetic classification data. Return list of per-epoch losses."""

    # --- synthetic data ---
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    # TODO step 1: create nn.Sequential model
    #   Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)
    model = None

    # TODO step 2: create AdamW optimizer with given lr
    optimizer = None

    # TODO step 3: create CosineAnnealingLR scheduler with T_max=num_epochs
    scheduler = None

    # TODO step 4: create CrossEntropyLoss
    criterion = None

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            # TODO step 5a: zero gradients
            # TODO step 5b: forward pass
            # TODO step 5c: compute loss
            # TODO step 5d: backward
            # TODO step 5e: clip gradients (max_norm=1.0)
            # TODO step 5f: optimizer step
            raise NotImplementedError

            running_loss += loss.item()
            num_batches += 1

        # TODO step 6: scheduler step
        avg_loss = running_loss / num_batches
        epoch_losses.append(avg_loss)

    # TODO step 7: eval pass with torch.no_grad
    # Compute final accuracy on full dataset
    model.eval()
    with torch.no_grad():
        # TODO: forward pass on X, get predictions, compute accuracy
        accuracy = None  # fraction correct
    pass

    return epoch_losses, accuracy
```

```python
# TEST
losses, acc = my_train()
assert losses[-1] < losses[0], f'Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}'
print(f'Loss: {losses[0]:.4f} -> {losses[-1]:.4f}')
print(f'Accuracy: {acc:.2%}')
print('PASS')
```
