# %% [markdown]
# # Drill 11 — KL Divergence & GRPO Loss
#
# ## Why This Matters (The Hard Way)
#
# You tried 7 GRPO configurations. Every single one hit a KL divergence wall:
# - KL shot to **14-23 after just 2 gradient updates**
# - Gradient norms hit **~1500** even at lr=1e-7
# - The base policy only completed step 1 about **20% of the time**
#
# That's not a hyperparameter problem. That's a **signal problem**.
#
# This notebook will give you the math to explain *exactly* why it failed — the kind
# of understanding that lets you diagnose RL training failures in an interview and
# sound like you actually know what's going on, not just that you tried a bunch of configs.
#
# **By the end you'll be able to:**
# 1. Compute KL divergence by hand and explain what the number means
# 2. Explain why GRPO normalizes advantages within a group
# 3. Show mathematically why low reward variance + KL penalty = catastrophic drift
# 4. Implement both `kl_divergence()` and `grpo_loss()` from scratch

# %%
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ---
# ## Part 1: KL Divergence — The Core Intuition
#
# **KL(P || Q)** = the expected number of **extra bits** you need if you use
# distribution Q to encode samples that actually come from distribution P.
#
# $$KL(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$
#
# Three properties to memorize:
# 1. **Always >= 0** (Gibbs' inequality)
# 2. **= 0 if and only if P = Q** (perfect encoding)
# 3. **NOT symmetric**: KL(P||Q) ≠ KL(Q||P) in general
#
# **In GRPO context**: KL(π_θ || π_ref) measures how far our training policy has
# drifted from the reference (SFT) policy. When this hits 14-23, the policy has
# moved so far that it's basically a different model speaking a different language.

# %% [markdown]
# ### Building Block 1: Computing KL Divergence
#
# Let's compute KL(P||Q) for two simple distributions over 3 outcomes.
#
# **Key insight**: Each term P(x) * log(P(x)/Q(x)) tells you how much "surprise cost"
# outcome x contributes. When P puts mass where Q doesn't, KL explodes.

# %%
# BB1: KL divergence from scratch
# P = true distribution, Q = approximate distribution

P = np.array([0.5, 0.3, 0.2])   # true distribution over 3 outcomes
Q = np.array([0.4, 0.4, 0.2])   # our approximation

# Step by step:
ratio = P / Q
log_ratio = np.log(ratio)           # natural log (nats, not bits)
per_term = P * log_ratio             # contribution of each outcome

print("P:         ", P)
print("Q:         ", Q)
print("P/Q:       ", ratio)
print("log(P/Q):  ", log_ratio)
print("P*log(P/Q):", per_term)
print()
print(f"KL(P||Q) = {per_term.sum():.6f} nats")
print(f"         = {per_term.sum() / np.log(2):.6f} bits")
print()
print("Interpretation: you need ~0.02 extra bits per sample if you")
print("use Q instead of P for encoding. These distributions are CLOSE.")

# %% [markdown]
# ### Building Block 2: KL is NOT Symmetric
#
# This is a classic interview gotcha. Let's see it concretely.
#
# **Intuition for why it matters**:
# - KL(P||Q) penalizes Q for putting LOW probability where P puts HIGH probability
# - KL(Q||P) penalizes P for putting LOW probability where Q puts HIGH probability
# - These are fundamentally different failure modes

# %%
# BB2: Asymmetry of KL divergence

P = np.array([0.9, 0.05, 0.05])  # peaked distribution
Q = np.array([0.33, 0.34, 0.33]) # nearly uniform

def kl_div(P, Q):
    """KL(P || Q) in nats. Assumes P, Q are valid probability distributions."""
    # Mask zeros in P to avoid 0 * log(0) = nan
    mask = P > 0
    return np.sum(P[mask] * np.log(P[mask] / Q[mask]))

kl_pq = kl_div(P, Q)
kl_qp = kl_div(Q, P)

print(f"P (peaked):  {P}")
print(f"Q (uniform): {Q}")
print()
print(f"KL(P || Q) = {kl_pq:.4f} nats  <-- 'Q is bad at encoding P'")
print(f"KL(Q || P) = {kl_qp:.4f} nats  <-- 'P is bad at encoding Q'")
print(f"Ratio:       {kl_pq / kl_qp:.2f}x")
print()
print("KL(P||Q) >> KL(Q||P) because Q spreads mass everywhere,")
print("but P concentrates mass. Using uniform Q to encode peaked P")
print("wastes a LOT of bits. The reverse is less wasteful.")
print()
print("In GRPO: KL(pi_theta || pi_ref) means we penalize the POLICY")
print("for putting probability where the REFERENCE does not.")
print("This is the 'mode-seeking' direction of KL.")

# %% [markdown]
# ### Building Block 3: Policy Gradient Intuition
#
# Before GRPO, let's recall the basic policy gradient idea.
#
# The core equation (REINFORCE):
#
# $$\nabla J(\theta) = \mathbb{E}\left[ \sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot R \right]$$
#
# In English: **increase the log-probability of actions that led to high reward**.
#
# The problem: reward R has high variance → noisy gradients → unstable training.

# %%
# BB3: Policy gradient intuition with a tiny example
# Imagine a policy over 4 possible responses to a prompt

responses = ["Response A", "Response B", "Response C", "Response D"]
log_probs = np.array([-2.3, -1.6, -3.1, -0.8])  # log pi(response | prompt)
rewards   = np.array([0.1,   0.9,  0.05,  0.3])  # reward for each

# Vanilla policy gradient: grad = reward * grad(log_prob)
# The gradient pushes harder on high-reward responses
vanilla_signal = rewards  # weight for each response's gradient

print("Response | log_prob | reward | gradient weight")
print("-" * 50)
for i in range(4):
    print(f"   {responses[i]:12s} | {log_probs[i]:6.2f} | {rewards[i]:5.2f}  | {vanilla_signal[i]:5.2f}")

print()
print("Problem: ALL rewards are positive, so we increase probability of")
print("ALL responses. Even the terrible Response C gets a small push up.")
print("This is why we need a BASELINE to center the signal.")

# %% [markdown]
# ### Building Block 4: Group-Relative Advantage (The 'GR' in GRPO)
#
# GRPO's key innovation: instead of a learned value baseline, use the **group statistics**.
#
# For a group of G responses to the same prompt:
#
# $$A_i = \frac{R_i - \text{mean}(R_{1:G})}{\text{std}(R_{1:G})}$$
#
# This is just z-score normalization. Responses better than the group average
# get positive advantage; worse ones get negative. The gradient pushes probability
# **toward** above-average responses and **away from** below-average ones.

# %%
# BB4: Group-relative advantage with G=8 responses

rewards_group = np.array([0.1, 0.9, 0.05, 0.3, 0.8, 0.15, 0.6, 0.2])
G = len(rewards_group)

mean_r = rewards_group.mean()
std_r  = rewards_group.std()

advantages = (rewards_group - mean_r) / std_r

print(f"Group of {G} rewards: {rewards_group}")
print(f"Mean: {mean_r:.4f},  Std: {std_r:.4f}")
print()
print("Response | Reward | Advantage | Direction")
print("-" * 55)
for i in range(G):
    direction = "INCREASE prob" if advantages[i] > 0 else "DECREASE prob"
    print(f"   {i+1}      | {rewards_group[i]:5.2f}  | {advantages[i]:+6.3f}   | {direction}")

print()
print("Key: advantages are CENTERED (sum ≈ 0) and SCALED (std ≈ 1)")
print(f"Sum of advantages: {advantages.sum():.6f} (≈ 0)")
print(f"Std of advantages: {advantages.std():.4f} (≈ 1)")

# %% [markdown]
# ### Building Block 5: The Full GRPO Loss
#
# The GRPO loss for a single prompt with G sampled responses:
#
# $$\mathcal{L}_{GRPO} = -\frac{1}{G}\sum_{i=1}^{G} \left[ \min\left(\rho_i A_i,\; \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i \right) - \beta \cdot KL_i \right]$$
#
# Where:
# - $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}$ is the importance ratio (like PPO)
# - $A_i$ is the group-relative advantage
# - $\epsilon$ is the clip range (typically 0.2)
# - $\beta$ is the KL penalty coefficient
# - $KL_i$ is the per-sample KL from reference policy
#
# The loss has **two competing forces**:
# 1. **Policy gradient term** (ρ·A): push toward high-advantage responses
# 2. **KL penalty** (β·KL): pull back toward the reference policy
#
# When β is too small → KL explodes (your problem: KL hit 14-23)
# When β is too large → no learning at all

# %%
# BB5: GRPO loss computation step by step

# Setup: 4 responses to one prompt
G = 4
rewards = np.array([0.1, 0.8, 0.05, 0.6])

# Log probabilities under current policy, old policy, and reference
log_pi_theta = np.array([-3.2, -1.5, -4.1, -2.0])  # current policy
log_pi_old   = np.array([-3.0, -1.6, -4.0, -2.1])  # old policy (from sampling)
log_pi_ref   = np.array([-2.8, -1.8, -3.5, -2.3])  # reference (SFT) policy

# --- Step 1: Group-relative advantages ---
mean_r = rewards.mean()
std_r  = rewards.std()
advantages = (rewards - mean_r) / (std_r + 1e-8)
print("Step 1: Advantages")
print(f"  rewards:    {rewards}")
print(f"  advantages: {advantages}")
print()

# --- Step 2: Importance ratios ---
log_ratio = log_pi_theta - log_pi_old
rho = np.exp(log_ratio)
print("Step 2: Importance ratios (rho = pi_theta / pi_old)")
print(f"  log_ratio: {log_ratio}")
print(f"  rho:       {rho}")
print()

# --- Step 3: Clipped surrogate (PPO-style) ---
epsilon = 0.2
clipped_rho = np.clip(rho, 1 - epsilon, 1 + epsilon)
surr1 = rho * advantages
surr2 = clipped_rho * advantages
pg_term = np.minimum(surr1, surr2)
print("Step 3: Clipped policy gradient")
print(f"  unclipped: {surr1}")
print(f"  clipped:   {surr2}")
print(f"  pg_term:   {pg_term}")
print()

# --- Step 4: Per-sample KL from reference ---
# KL at token level: pi_theta / pi_ref in log space, approximated
# Using the simple estimator: exp(log_pi_ref - log_pi_theta) - 1 - (log_pi_ref - log_pi_theta)
# or just log(pi_theta/pi_ref) for the per-sample version
log_r_ref = log_pi_theta - log_pi_ref
kl_per_sample = np.exp(log_r_ref) - 1 - log_r_ref  # Schulman's KL estimator (always >= 0)
print("Step 4: Per-sample KL from reference")
print(f"  log(pi_theta/pi_ref): {log_r_ref}")
print(f"  KL per sample:        {kl_per_sample}")
print()

# --- Step 5: Combined loss ---
beta = 0.1  # KL penalty coefficient
loss_per_sample = -(pg_term - beta * kl_per_sample)
loss = loss_per_sample.mean()
print("Step 5: Combined GRPO loss")
print(f"  pg_term:          {pg_term}")
print(f"  beta * KL:        {beta * kl_per_sample}")
print(f"  loss per sample:  {loss_per_sample}")
print(f"  TOTAL LOSS:       {loss:.6f}")

# %% [markdown]
# ### Building Block 6: Why YOUR GRPO Failed
#
# Here's the critical insight. Your base policy completed step 1 only ~20% of the time.
# That means most of the G sampled responses probably all failed in the same way,
# getting similar (low) rewards.
#
# **When all rewards are similar → std(R) ≈ 0 → advantages explode or become meaningless**
#
# Let's simulate this.

# %%
# BB6: The failure mode — low reward variance

print("=" * 60)
print("SCENARIO 1: Healthy reward distribution")
print("=" * 60)
rewards_healthy = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.9, 0.0, 0.1])
mean_h = rewards_healthy.mean()
std_h  = rewards_healthy.std()
adv_h  = (rewards_healthy - mean_h) / (std_h + 1e-8)
print(f"Rewards: {rewards_healthy}")
print(f"Mean: {mean_h:.3f}, Std: {std_h:.3f}")
print(f"Advantages: {adv_h}")
print(f"Max |advantage|: {np.abs(adv_h).max():.3f}")
print()

print("=" * 60)
print("SCENARIO 2: Your situation — nearly all failures")
print("=" * 60)
rewards_bad = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
mean_b = rewards_bad.mean()
std_b  = rewards_bad.std()
adv_b  = (rewards_bad - mean_b) / (std_b + 1e-8)
print(f"Rewards: {rewards_bad}")
print(f"Mean: {mean_b:.4f}, Std: {std_b:.4f}")
print(f"Advantages: {adv_b}")
print(f"Max |advantage|: {np.abs(adv_b).max():.3f}")
print()

print("=" * 60)
print("SCENARIO 3: Complete failure — ALL zero rewards")
print("=" * 60)
rewards_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
mean_z = rewards_zero.mean()
std_z  = rewards_zero.std()
# With epsilon protection:
adv_z  = (rewards_zero - mean_z) / (std_z + 1e-8)
print(f"Rewards: {rewards_zero}")
print(f"Mean: {mean_z:.4f}, Std: {std_z:.6f}")
print(f"Advantages: {adv_z}")
print(f"Max |advantage|: {np.abs(adv_z).max():.3f}")
print()

print("KEY INSIGHT:")
print("-" * 60)
print("In Scenario 2, that tiny 0.05 reward gets advantage = +3.7")
print("The policy MASSIVELY upweights that one response, regardless")
print("of whether it was actually good or just slightly less terrible.")
print()
print("This creates HUGE gradient norms (you saw ~1500!), which push")
print("the policy far from the reference in one step → KL explodes.")
print()
print("In Scenario 3 with pure zeros, advantages = 0, so the ONLY")
print("gradient signal comes from the KL penalty pulling back to ref.")
print("But the damage from earlier huge updates is already done.")

# %% [markdown]
# ### The Death Spiral
#
# Here's the full failure cascade:
#
# 1. **Base policy is weak** → 80% of responses fail → low reward variance
# 2. **Low variance** → one slightly-better response gets enormous advantage
# 3. **Enormous advantage** → massive gradient (norm ~1500)
# 4. **Massive gradient** → policy jumps far from reference in one step
# 5. **Far from reference** → KL penalty now dominates the loss
# 6. **KL dominates** → gradient points back toward reference, NOT toward good responses
# 7. **Oscillation** → policy bounces between "collapse toward ref" and "massive jump away"
# 8. **KL reads 14-23** → training is effectively dead
#
# **The fix**: the base model needed to be stronger BEFORE GRPO. GRPO refines a policy
# that's already decent — it can't teach a model a fundamentally new capability.
# Think of GRPO as fine-tuning the last mile, not the first mile.

# %% [markdown]
# ---
# ## Part 2: Complete Implementations
#
# Now let's write clean implementations that you could explain line-by-line in an interview.

# %%
def kl_divergence(P, Q):
    """
    Compute KL(P || Q) for discrete distributions P and Q.
    
    KL(P||Q) = sum_x P(x) * log(P(x) / Q(x))
    
    Args:
        P: array of probabilities (true distribution)
        Q: array of probabilities (approximate distribution)
    
    Returns:
        KL divergence in nats (use / log(2) for bits)
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    
    assert np.allclose(P.sum(), 1.0), f"P must sum to 1, got {P.sum()}"
    assert np.allclose(Q.sum(), 1.0), f"Q must sum to 1, got {Q.sum()}"
    assert (P >= 0).all() and (Q >= 0).all(), "Probabilities must be non-negative"
    
    # Where P > 0 but Q = 0, KL = infinity
    if np.any((P > 0) & (Q == 0)):
        return float('inf')
    
    # Only sum where P > 0 (0 * log(0/q) = 0 by convention)
    mask = P > 0
    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))


# Verify properties
P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.3, 0.4, 0.3])

print("kl_divergence tests:")
print(f"  KL(P||Q) = {kl_divergence(P, Q):.6f} nats  (should be > 0)")
print(f"  KL(Q||P) = {kl_divergence(Q, P):.6f} nats  (different from above!)")
print(f"  KL(P||P) = {kl_divergence(P, P):.6f} nats  (should be 0)")
print(f"  Non-negative: {kl_divergence(P, Q) >= 0}  ✓")
print(f"  Asymmetric:   {abs(kl_divergence(P, Q) - kl_divergence(Q, P)) > 1e-10}  ✓")

def grpo_loss(log_pi_theta, log_pi_old, log_pi_ref, rewards,
              epsilon=0.2, beta=0.1):
    """
    Compute GRPO loss for a single prompt with G sampled responses.
    
    L = -mean_i [ min(rho_i * A_i, clip(rho_i) * A_i) - beta * KL_i ]
    
    Args:
        log_pi_theta: (G,) log probs under current policy
        log_pi_old:   (G,) log probs under sampling policy
        log_pi_ref:   (G,) log probs under reference (SFT) policy
        rewards:      (G,) rewards for each response
        epsilon:      clip range for importance ratio
        beta:         KL penalty coefficient
    
    Returns:
        dict with loss, advantages, kl, pg_term for inspection
    """
    G = len(rewards)
    
    # --- Group-relative advantages ---
    mean_r = rewards.mean()
    std_r  = rewards.std()
    advantages = (rewards - mean_r) / (std_r + 1e-8)
    
    # --- Importance ratio (PPO-style) ---
    log_ratio = log_pi_theta - log_pi_old
    rho = np.exp(log_ratio)
    
    # --- Clipped surrogate objective ---
    clipped_rho = np.clip(rho, 1 - epsilon, 1 + epsilon)
    surr1 = rho * advantages
    surr2 = clipped_rho * advantages
    pg_term = np.minimum(surr1, surr2)
    
    # --- KL penalty from reference (Schulman's unbiased estimator) ---
    # KL(pi_theta || pi_ref) ≈ exp(log_ratio_ref) - 1 - log_ratio_ref
    log_ratio_ref = log_pi_theta - log_pi_ref
    kl_penalty = np.exp(log_ratio_ref) - 1 - log_ratio_ref  # always >= 0
    
    # --- Combined loss (negative because we minimize) ---
    per_sample = pg_term - beta * kl_penalty
    loss = -per_sample.mean()
    
    return {
        'loss': float(loss),
        'advantages': advantages,
        'rho': rho,
        'pg_term': pg_term,
        'kl_penalty': kl_penalty,
        'mean_kl': float(kl_penalty.mean()),
    }


# Test with a healthy scenario
result = grpo_loss(
    log_pi_theta = np.array([-3.2, -1.5, -4.1, -2.0]),
    log_pi_old   = np.array([-3.0, -1.6, -4.0, -2.1]),
    log_pi_ref   = np.array([-2.8, -1.8, -3.5, -2.3]),
    rewards       = np.array([0.1,  0.8,  0.05, 0.6]),
    epsilon=0.2,
    beta=0.1,
)

print("GRPO Loss (healthy scenario):")
print(f"  Loss:       {result['loss']:.6f}")
print(f"  Mean KL:    {result['mean_kl']:.6f}")
print(f"  Advantages: {result['advantages']}")
print(f"  Rho:        {result['rho']}")
print()

# Test with YOUR failure scenario
result_bad = grpo_loss(
    log_pi_theta = np.array([-5.0, -4.8, -4.9, -4.7, -4.85, -4.95, -4.6, -3.2]),
    log_pi_old   = np.array([-3.5, -3.5, -3.5, -3.5, -3.5,  -3.5,  -3.5, -3.5]),
    log_pi_ref   = np.array([-3.5, -3.5, -3.5, -3.5, -3.5,  -3.5,  -3.5, -3.5]),
    rewards       = np.array([0.0,  0.0,  0.0,  0.0,  0.0,   0.0,   0.0,  0.05]),
    epsilon=0.2,
    beta=0.04,
)

print("GRPO Loss (YOUR failure scenario):")
print(f"  Loss:       {result_bad['loss']:.6f}")
print(f"  Mean KL:    {result_bad['mean_kl']:.6f}")
print(f"  Advantages: {result_bad['advantages']}")
print(f"  Max |adv|:  {np.abs(result_bad['advantages']).max():.3f}")
print()
print("See that max advantage? ONE response dominates the entire gradient.")
print("Now imagine this with a neural net and millions of parameters...")

# %% [markdown]
# ---
# ## Interview Cheat Sheet
#
# **"What is KL divergence?"**
# > KL(P||Q) measures the extra bits needed to encode samples from P using distribution Q.
# > It's always non-negative, zero iff P=Q, and asymmetric.
#
# **"What is GRPO?"**
# > Group Relative Policy Optimization. Like PPO but replaces the learned value function
# > with group-relative normalization: advantage = (reward - group_mean) / group_std.
# > Simpler to implement, no critic network needed.
#
# **"Why does GRPO include a KL penalty?"**
# > Without it, the policy can drift arbitrarily far from the reference. The KL penalty
# > acts as an elastic band pulling the policy back. The coefficient β controls the trade-off
# > between reward maximization and staying close to the reference.
#
# **"What goes wrong when the base model is too weak for GRPO?"**
# > When most sampled responses get similar (bad) rewards, the reward variance within each
# > group is tiny. This makes advantages huge for any slightly-better response, causing
# > massive gradient norms, which push the policy far from the reference in one step.
# > Then KL penalty dominates, creating a death spiral of oscillation.
# > GRPO is for refinement, not teaching fundamentally new capabilities.
#
# **"What's the difference between KL(π_θ||π_ref) and KL(π_ref||π_θ)?"**
# > KL(π_θ||π_ref) is "mode-seeking" — penalizes the policy for putting mass where the
# > reference doesn't. This is what GRPO uses. KL(π_ref||π_θ) is "mode-covering" —
# > penalizes the policy for NOT covering all modes of the reference.

# %% [markdown]
# ---
# ## Your Turn
#
# Implement from scratch without looking above.

# %%
def your_kl_divergence(P, Q):
    """Compute KL(P || Q). Return value in nats."""
    pass  # YOUR CODE HERE


def your_grpo_loss(log_pi_theta, log_pi_old, log_pi_ref, rewards,
                   epsilon=0.2, beta=0.1):
    """
    Compute GRPO loss.
    Return dict with at least 'loss' and 'mean_kl' keys.
    """
    pass  # YOUR CODE HERE

# %% [markdown]
# ### Tests

# %%
# ---- Test kl_divergence ----
P1 = np.array([0.5, 0.3, 0.2])
Q1 = np.array([0.4, 0.4, 0.2])

kl_val = your_kl_divergence(P1, Q1)
assert kl_val is not None, "your_kl_divergence returned None"
assert abs(kl_val - 0.02659) < 0.001, f"KL(P1||Q1) should be ~0.02659, got {kl_val}"

# Verify properties
assert abs(your_kl_divergence(P1, P1)) < 1e-10, "KL(P,P) should be 0"
assert your_kl_divergence(P1, Q1) >= -1e-10, "KL should be non-negative"
assert abs(your_kl_divergence(P1, Q1) - your_kl_divergence(Q1, P1)) > 1e-6, "KL should be asymmetric"

print("kl_divergence: ALL TESTS PASSED ✓")

# ---- Test grpo_loss ----
result_test = your_grpo_loss(
    log_pi_theta = np.array([-3.2, -1.5, -4.1, -2.0]),
    log_pi_old   = np.array([-3.0, -1.6, -4.0, -2.1]),
    log_pi_ref   = np.array([-2.8, -1.8, -3.5, -2.3]),
    rewards       = np.array([0.1,  0.8,  0.05, 0.6]),
    epsilon=0.2,
    beta=0.1,
)

assert result_test is not None, "your_grpo_loss returned None"
assert 'loss' in result_test, "Result must contain 'loss' key"
assert 'mean_kl' in result_test, "Result must contain 'mean_kl' key"
assert result_test['mean_kl'] >= -1e-10, "KL should be non-negative"

# Compare with reference
ref_result = grpo_loss(
    log_pi_theta = np.array([-3.2, -1.5, -4.1, -2.0]),
    log_pi_old   = np.array([-3.0, -1.6, -4.0, -2.1]),
    log_pi_ref   = np.array([-2.8, -1.8, -3.5, -2.3]),
    rewards       = np.array([0.1,  0.8,  0.05, 0.6]),
)
assert abs(result_test['loss'] - ref_result['loss']) < 0.01, \
    f"Loss mismatch: got {result_test['loss']:.6f}, expected {ref_result['loss']:.6f}"

print("grpo_loss: ALL TESTS PASSED ✓")
print()
print("You can now explain KL divergence and GRPO loss in an interview. 🎯")
