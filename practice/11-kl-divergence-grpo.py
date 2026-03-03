# %% [markdown]
# # Drill 11: KL Divergence & GRPO Loss
# Two implementations:
# 1. KL divergence between two probability distributions
# 2. GRPO-style policy gradient loss with KL penalty

# %%
import torch
import torch.nn.functional as F

# %% [markdown]
# ## Part 1: KL Divergence
# KL(P || Q) = sum(P * log(P / Q))
# Input: P, Q — probability distributions of shape (B, V)
# Output: scalar (mean over batch)

# %%
def my_kl_divergence(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Compute KL(P || Q) = sum over vocab of P * log(P/Q), averaged over batch.

    Args:
        P: (B, V) probability distribution
        Q: (B, V) probability distribution

    Returns:
        scalar tensor
    """
    # TODO: compute KL divergence
    # Hint: be careful with numerical stability (log of small numbers)
    raise NotImplementedError

# %% [markdown]
# ## Part 2: GRPO Loss
# Group Relative Policy Optimization loss:
# 1. Normalize rewards: advantages = (R - mean(R)) / std(R)
# 2. Policy gradient loss: pg_loss = -(advantages * log_probs).mean()
# 3. KL penalty: kl = (log_probs - ref_log_probs).mean()
# 4. Return: pg_loss + beta * kl

# %%
def my_grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute GRPO loss = policy gradient loss + beta * KL penalty.

    Args:
        log_probs: (B,) log probabilities from current policy
        ref_log_probs: (B,) log probabilities from reference policy
        rewards: (B,) reward scores for each sample
        beta: KL penalty coefficient

    Returns:
        scalar loss tensor
    """
    # TODO: compute advantages (normalize rewards)
    # TODO: compute policy gradient loss
    # TODO: compute KL penalty between current and reference policy
    # TODO: return combined loss
    raise NotImplementedError

# %% [markdown]
# ## Tests

# %%
# TEST: KL Divergence
torch.manual_seed(42)

# KL(P, P) should be 0
P = F.softmax(torch.randn(4, 10), dim=-1)
kl_same = my_kl_divergence(P, P)
assert torch.allclose(kl_same, torch.tensor(0.0), atol=1e-6), f"KL(P,P) should be 0, got {kl_same.item()}"

# KL should be >= 0 (Gibbs' inequality)
Q = F.softmax(torch.randn(4, 10), dim=-1)
kl_val = my_kl_divergence(P, Q)
assert kl_val >= 0, f"KL should be non-negative, got {kl_val.item()}"

# KL should be asymmetric: KL(P||Q) != KL(Q||P) in general
kl_pq = my_kl_divergence(P, Q)
kl_qp = my_kl_divergence(Q, P)
assert not torch.allclose(kl_pq, kl_qp, atol=1e-6), "KL should be asymmetric"

print(f"KL(P,P) = {kl_same.item():.6f}")
print(f"KL(P,Q) = {kl_pq.item():.6f}")
print(f"KL(Q,P) = {kl_qp.item():.6f}")
print("All KL divergence tests passed!")

# %%
# TEST: GRPO Loss
torch.manual_seed(42)

log_probs = torch.randn(8)
ref_log_probs = log_probs.clone()  # same policy
rewards = torch.randn(8)

# Should run without error
loss = my_grpo_loss(log_probs, ref_log_probs, rewards, beta=0.1)
assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"

# When policies are same, KL penalty should be 0, loss = pg_loss only
loss_no_kl = my_grpo_loss(log_probs, ref_log_probs, rewards, beta=0.0)
loss_with_kl = my_grpo_loss(log_probs, ref_log_probs, rewards, beta=1.0)
# With identical policies, both should be equal (KL=0)
assert torch.allclose(loss_no_kl, loss_with_kl, atol=1e-5), "With same policy, beta shouldn't matter"

# With different policies, beta should matter
ref_log_probs_diff = torch.randn(8)
loss_low_beta = my_grpo_loss(log_probs, ref_log_probs_diff, rewards, beta=0.01)
loss_high_beta = my_grpo_loss(log_probs, ref_log_probs_diff, rewards, beta=10.0)
assert not torch.allclose(loss_low_beta, loss_high_beta), "Different beta should give different loss"

print(f"GRPO loss (same policy): {loss.item():.6f}")
print(f"GRPO loss (low beta):    {loss_low_beta.item():.6f}")
print(f"GRPO loss (high beta):   {loss_high_beta.item():.6f}")
print("All GRPO tests passed!")
