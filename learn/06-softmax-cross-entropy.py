# %% [markdown]
# # Drill 06 — Softmax & Cross-Entropy Loss
#
# ## Why This Matters
#
# You already built softmax when you implemented attention — it turned raw dot-product
# scores into weights that sum to 1. That same function shows up everywhere.
#
# **Cross-entropy** is the loss function used in SFT (supervised fine-tuning). When you
# trained your model and watched the loss drop from **3.29 → 0.09**, those numbers have
# a precise meaning:
#
# - **Loss = 3.29** → the model assigned **exp(-3.29) ≈ 3.7%** probability to the correct action
# - **Loss = 0.09** → the model assigned **exp(-0.09) ≈ 91.4%** probability to the correct action
#
# By the end of this notebook you'll:
# 1. Implement softmax from scratch (with the stability trick interviewers love to ask about)
# 2. Implement cross-entropy loss from scratch
# 3. Understand exactly what your training loss numbers mean

# %% [markdown]
# ---
# ## Intuition
#
# **Softmax** converts a vector of raw scores (logits) into a probability distribution:
# - All outputs are between 0 and 1
# - They sum to 1
# - Larger logits get larger probabilities
#
# **Cross-entropy** measures how "wrong" a predicted probability distribution is:
# - If the model puts 100% on the correct class → loss = 0 (perfect)
# - If the model puts 0% on the correct class → loss = ∞ (catastrophic)
#
# They're connected by one clean formula:
#
# **CE loss = -log( softmax(logits)[correct_class] )**
#
# That's it. The loss is just the negative log of the probability the model assigned
# to the right answer.

# %% [markdown]
# ---
# ## Building Block 1: Naive Softmax
#
# The formula: **softmax(x)_i = exp(x_i) / Σ exp(x_j)**
#
# Exponentiate each element, then divide by the sum. Let's try it.

# %%
import numpy as np

def naive_softmax(x):
    """Softmax — the naive way."""
    exps = np.exp(x)
    return exps / np.sum(exps)

# Try it on small, friendly numbers
logits = np.array([2.0, 1.0, 0.1])
probs = naive_softmax(logits)

print("Logits:", logits)
print("Probabilities:", probs)
print("Sum:", probs.sum())  # Should be 1.0
print()
print("Largest logit (2.0) → largest probability ({:.1%})".format(probs[0]))
print("Smallest logit (0.1) → smallest probability ({:.1%})".format(probs[2]))

# %% [markdown]
# ---
# ## Building Block 2: The Numerical Instability Problem
#
# That naive version has a fatal flaw. Watch what happens with large logits:

# %%
# What happens with large numbers?
big_logits = np.array([1000.0, 1001.0, 1002.0])

print("exp(1000) =", np.exp(1000))  # inf!
print()

try:
    result = naive_softmax(big_logits)
    print("Naive softmax:", result)
except Exception as e:
    print("Error:", e)

# Even without an error, we get nan (inf / inf = nan)
print()
print("The result is full of nan — completely useless.")
print("This WILL happen in real networks. Logits can get large.

# %% [markdown]
# ---
# ## Building Block 3: The Fix — Subtract the Max
#
# Here's the trick (and the interview answer):
#
# **softmax(x) = softmax(x - c)** for any constant c.
#
# Proof: if we subtract c from every element:
# - Numerator: exp(x_i - c) = exp(x_i) · exp(-c)
# - Denominator: Σ exp(x_j - c) = exp(-c) · Σ exp(x_j)
# - The exp(-c) cancels!
#
# So we choose **c = max(x)**. This makes the largest exponent exp(0) = 1.
# No overflow possible.
#
# This is mathematically identical but numerically stable.

# %%
def stable_softmax(x):
    """Softmax — numerically stable version."""
    shifted = x - np.max(x)      # Subtract max → largest value becomes 0
    exps = np.exp(shifted)        # exp(0) = 1, everything else < 1
    return exps / np.sum(exps)

# Same small example — identical results
logits_small = np.array([2.0, 1.0, 0.1])
print("Naive:  ", naive_softmax(logits_small))
print("Stable: ", stable_softmax(logits_small))
print("Match:", np.allclose(naive_softmax(logits_small), stable_softmax(logits_small)))
print()

# Now the big example — stable version handles it fine
big_logits = np.array([1000.0, 1001.0, 1002.0])
print("Stable softmax on [1000, 1001, 1002]:", stable_softmax(big_logits))
print("Sum:", stable_softmax(big_logits).sum())
print()
print("No inf, no nan. Same math, safe numbers.")

# %% [markdown]
# ---
# ## Building Block 4: Cross-Entropy from Softmax
#
# Cross-entropy loss for a single example:
#
# **loss = -log( prob[correct_class] )**
#
# Why the log?
# - If prob = 1.0 (perfect): loss = -log(1) = **0** ✓
# - If prob = 0.5 (coin flip): loss = -log(0.5) = **0.693**
# - If prob = 0.01 (very wrong): loss = -log(0.01) = **4.605**
# - If prob → 0 (disastrously wrong): loss → **∞**
#
# The log creates a steep penalty as the model gets more confident about the WRONG answer.
# This is exactly what you want during training.

# %%
def cross_entropy_from_probs(probs, target_idx):
    """Cross-entropy loss given probabilities and the correct class index."""
    return -np.log(probs[target_idx])

# Example: 3 classes, correct class is index 1
logits = np.array([2.0, 5.0, 1.0])
probs = stable_softmax(logits)
target = 1

print("Logits:", logits)
print("Probabilities:", np.round(probs, 4))
print(f"Probability on correct class (idx {target}): {probs[target]:.4f}")
print(f"Cross-entropy loss: {cross_entropy_from_probs(probs, target):.4f}")
print()

# What if the model is very wrong?
logits_wrong = np.array([5.0, 0.5, 1.0])  # Correct class (1) has low logit
probs_wrong = stable_softmax(logits_wrong)
print("Logits (model is wrong):", logits_wrong)
print("Probabilities:", np.round(probs_wrong, 4))
print(f"Probability on correct class (idx {target}): {probs_wrong[target]:.4f}")
print(f"Cross-entropy loss: {cross_entropy_from_probs(probs_wrong, target):.4f}")
print("↑ Much higher loss when the model is wrong")

# %% [markdown]
# ---
# ## Building Block 5: What Your Training Loss Numbers Actually Mean
#
# Remember: **loss = -log(prob_correct)**, so **prob_correct = exp(-loss)**.
#
# Let's decode the loss values from your SFT training run:

# %%
print("=" * 55)
print(f"{'Loss':>8}  {'Prob on correct action':>22}  {'Meaning'}")
print("=" * 55)

losses = [3.29, 2.30, 1.61, 1.00, 0.50, 0.10, 0.09]
labels = [
    "Start of training",
    "~10% correct",
    "~20% correct",
    "Getting somewhere",
    "More right than wrong",
    "Near convergence",
    "Your final loss!",
]

for loss, label in zip(losses, labels):
    prob = np.exp(-loss)
    bar = "█" * int(prob * 30)
    print(f"  {loss:5.2f}   →  {prob:6.1%}  {bar:30s}  {label}")

print()
print("Your model went from assigning 3.7% probability to the")
print("correct action to assigning 91.4%. That's what training did.")

# %% [markdown]
# ---
# ## Complete Implementation: Reference Solutions
#
# Here are the final, clean implementations:

# %%
def softmax(x):
    """
    Numerically stable softmax.
    
    Args:
        x: numpy array of logits (any shape along last axis)
    Returns:
        Probability distribution (same shape as x, sums to 1 along last axis)
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def cross_entropy_loss(logits, target_idx):
    """
    Cross-entropy loss from raw logits.
    
    Combines softmax + negative log likelihood in one function.
    Uses the log-sum-exp trick for extra numerical stability.
    
    Args:
        logits: numpy array of raw scores, shape (num_classes,)
        target_idx: integer index of the correct class
    Returns:
        Scalar loss value
    """
    # Log-softmax is more stable than log(softmax(x))
    # log_softmax(x)_i = x_i - log(sum(exp(x)))
    # With stability trick: x_i - max(x) - log(sum(exp(x - max(x))))
    shifted = logits - np.max(logits)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    log_probs = shifted - log_sum_exp
    return -log_probs[target_idx]


# Quick demo
demo_logits = np.array([1.5, 3.0, 0.5, -1.0])
demo_target = 1  # correct class

print("Logits:      ", demo_logits)
print("Softmax:     ", np.round(softmax(demo_logits), 4))
print("Sum check:   ", softmax(demo_logits).sum())
print(f"CE Loss:      {cross_entropy_loss(demo_logits, demo_target):.4f}")
print(f"Prob correct: {softmax(demo_logits)[demo_target]:.4f}")
print(f"Check: -log(prob) = {-np.log(softmax(demo_logits)[demo_target]):.4f}")

# %% [markdown]
# ---
# ## Your Turn
#
# Implement both functions from scratch. Don't peek at the reference above!
#
# **Softmax requirements:**
# - Subtract the max for numerical stability
# - Works on 1D numpy arrays
# - Output sums to 1
#
# **Cross-entropy requirements:**
# - Take raw logits (not probabilities) and a target index
# - Return a scalar loss value
# - Handle large logits without overflow

# %%
def my_softmax(x):
    """
    Implement numerically stable softmax.
    
    Steps:
    1. Subtract max(x) from x
    2. Exponentiate
    3. Divide by sum
    """
    pass  # YOUR CODE HERE


# Quick test (uncomment when ready)
# test = np.array([2.0, 1.0, 0.1])
# print(my_softmax(test))
# print("Sum:", my_softmax(test).sum())

def my_cross_entropy_loss(logits, target_idx):
    """
    Implement cross-entropy loss from raw logits.
    
    Steps:
    1. Compute softmax (use your my_softmax function)
    2. Take the probability of the correct class
    3. Return -log of that probability
    """
    pass  # YOUR CODE HERE


# Quick test (uncomment when ready)
# test_logits = np.array([1.5, 3.0, 0.5, -1.0])
# print(f"Your CE loss: {my_cross_entropy_loss(test_logits, 1):.4f}")
# print(f"Reference:    {cross_entropy_loss(test_logits, 1):.4f}")

# %% [markdown]
# ---
# ## Verification Against PyTorch
#
# Let's compare our implementations against PyTorch's battle-tested versions.
# This is how you know you got it right.

# %%
import torch
import torch.nn.functional as F

np.random.seed(42)

print("Testing softmax...")
print("-" * 50)
for name, test_logits in [
    ("Small values", np.array([2.0, 1.0, 0.1])),
    ("Large values", np.array([1000.0, 1001.0, 1002.0])),
    ("Negative values", np.array([-5.0, -2.0, -0.1])),
    ("Random (10 classes)", np.random.randn(10) * 5),
]:
    ours = softmax(test_logits)
    theirs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    match = np.allclose(ours, theirs, atol=1e-6)
    print(f"  {name:25s} {'PASS ✓' if match else 'FAIL ✗'}")

print()
print("Testing cross-entropy loss...")
print("-" * 50)
for name, logits_np, target in [
    ("Confident correct", np.array([0.5, 5.0, 0.1]), 1),
    ("Confident wrong",   np.array([5.0, 0.5, 0.1]), 1),
    ("Uniform-ish",       np.array([1.0, 1.0, 1.0]), 0),
    ("Large logits",      np.array([500.0, 501.0, 499.0]), 1),
    ("Random (10 cls)",   np.random.randn(10) * 3, 7),
]:
    ours = cross_entropy_loss(logits_np, target)
    theirs = F.cross_entropy(
        torch.tensor(logits_np).unsqueeze(0).float(),
        torch.tensor([target])
    ).item()
    match = np.isclose(ours, theirs, atol=1e-4)
    print(f"  {name:25s} ours={ours:.4f}  torch={theirs:.4f}  {'PASS ✓' if match else 'FAIL ✗'}")

print()
print("All tests passed! Your implementation matches PyTorch.")

# %% [markdown]
# ---
# ## Key Takeaways
#
# 1. **Softmax** = exp(x) / sum(exp(x)), but ALWAYS subtract the max first
# 2. **Cross-entropy loss** = -log(softmax(logits)[correct_class])
# 3. **Loss value meaning**: prob_correct = exp(-loss)
#    - Loss 3.29 → 3.7% confidence on correct answer
#    - Loss 0.09 → 91.4% confidence on correct answer
# 4. **The stability trick** (subtract max) is a standard interview question — know why it works
# 5. In practice, libraries fuse softmax + log + negative into one numerically stable operation
#
# These two functions are the foundation of classification training. Every time you
# see `F.cross_entropy` in PyTorch, you now know exactly what's happening inside.
