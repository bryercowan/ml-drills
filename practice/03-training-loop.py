# %% [markdown]
# # Drill 3: Training Loop
# Implement a complete training loop from scratch.
# Create a small MLP, optimizer, scheduler, loss fn.
# Train on synthetic data and verify loss decreases.
# Steps:
# 1. Create nn.Sequential MLP (in -> hidden -> ReLU -> hidden -> out)
# 2. Create AdamW optimizer
# 3. Create CosineAnnealingLR scheduler
# 4. Create CrossEntropyLoss
# 5. Inner loop: zero_grad -> forward -> loss -> backward -> clip_grad_norm_ -> optimizer step
# 6. Scheduler step per epoch
# 7. Eval with torch.no_grad()

# %%
import torch
import torch.nn as nn

# %%
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

# %%
# TEST
losses, acc = my_train()
assert losses[-1] < losses[0], f'Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}'
print(f'Loss: {losses[0]:.4f} -> {losses[-1]:.4f}')
print(f'Accuracy: {acc:.2%}')
print('PASS')
