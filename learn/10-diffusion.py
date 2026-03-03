# %% [markdown]
# # Drill 10 — DDPM Diffusion: Forward Noising & Denoising
#
# ## Why This Matters
#
# Your CNN policy **is** a diffusion policy (Chi et al., 2023 — *Diffusion Policy*).
# It doesn't denoise images — it denoises **action vectors**.
#
# Here's what happens every time your policy runs:
#
# 1. Start with pure noise: a random 10-dim vector (8 action-type logits + 2 coordinates)
# 2. The CNN looks at the current observation + that noisy vector
# 3. It predicts the noise component
# 4. Subtract (a scaled version of) that noise → slightly cleaner action vector
# 5. Repeat for a few steps → clean action prediction
#
# **Training**: 100 timesteps, linear beta schedule
# **Inference**: 10 DDIM steps (skipping timesteps for speed)
#
# This notebook builds every piece from scratch so you understand exactly what
# `add_noise()` and `denoise_step()` do inside your policy.

# %% [markdown]
# ## The Core Idea
#
# ### Forward process (adding noise)
#
# Given a clean sample $x_0$, we can create a noisy version at ANY timestep $t$:
#
# $$x_t = \sqrt{\bar\alpha_t}\; x_0 \;+\; \sqrt{1 - \bar\alpha_t}\; \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
#
# **Key insight**: $\bar\alpha_t$ is a cumulative product, so you can jump straight
# to timestep $t$ — no need to noise step-by-step. This is the reparameterization trick.
#
# - At $t=0$: $\bar\alpha \approx 1$ → almost all signal, almost no noise
# - At $t=T$: $\bar\alpha \approx 0$ → almost pure noise
#
# ### Backward process (removing noise)
#
# A neural network $\epsilon_\theta(x_t, t)$ learns to predict which noise was added.
# Then we reverse one step:
#
# $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
#
# ### Training loss
#
# Dead simple — just MSE between the true noise and predicted noise:
#
# $$\mathcal{L} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$
#
# That's it. Sample a random $t$, noise $x_0$ to get $x_t$, predict the noise, compute MSE.

# %%
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
T = 100  # number of diffusion timesteps (same as your policy)
dim = 10  # action vector dimension (8 action types + 2 coords)

# A fake "clean" action vector — our x_0
# Think of this as: [action_type_logits..., x_coord, y_coord]
x_0 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7])
print(f"x_0 (clean action): {x_0}")
print(f"  action type logits: {x_0[:8]}")
print(f"  coordinates:        {x_0[8:]}")

# %% [markdown]
# ## Building Block 1: Beta Schedule
#
# $\beta_t$ controls how much noise is added at each step.
# Linear schedule from $\beta_1 = 10^{-4}$ (tiny noise) to $\beta_T = 0.02$ (more noise).
#
# Small betas → gradual noising → easier for the network to learn the reverse.

# %%
# Linear beta schedule — same as the original DDPM paper
betas = torch.linspace(1e-4, 0.02, T)

print(f"beta_1  = {betas[0]:.6f}  (barely any noise)")
print(f"beta_T  = {betas[-1]:.6f}  (more noise)")
print(f"shape:    {betas.shape}")

plt.figure(figsize=(8, 3))
plt.plot(betas.numpy(), linewidth=2)
plt.xlabel("Timestep t")
plt.ylabel("β_t")
plt.title("Linear Beta Schedule")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Building Block 2: Alpha and Alpha-Bar
#
# - $\alpha_t = 1 - \beta_t$ (how much signal is *kept* at step $t$)
# - $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$ (cumulative — how much signal survives from step 0 to $t$)
#
# $\bar\alpha_t$ is the magic number. It tells you the signal-to-noise ratio at any timestep.

# %%
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

print(f"alpha_bar at t=0:   {alpha_bars[0]:.4f}  (almost all signal)")
print(f"alpha_bar at t=49:  {alpha_bars[49]:.4f}  (about half)")
print(f"alpha_bar at t=99:  {alpha_bars[99]:.4f}  (almost no signal)")

fig, axes = plt.subplots(1, 2, figsize=(12, 3))

axes[0].plot(alphas.numpy(), linewidth=2, label="α_t")
axes[0].set_xlabel("Timestep t")
axes[0].set_ylabel("α_t")
axes[0].set_title("Alpha (per-step signal retention)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(alpha_bars.numpy(), linewidth=2, color="tab:orange", label="ᾱ_t")
axes[1].set_xlabel("Timestep t")
axes[1].set_ylabel("ᾱ_t")
axes[1].set_title("Alpha-Bar (cumulative signal retention)")
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nInterpretation:")
print(f"  sqrt(alpha_bar_0)  = {alpha_bars[0].sqrt():.4f} → signal coefficient at t=0")
print(f"  sqrt(alpha_bar_99) = {alpha_bars[99].sqrt():.4f} → signal coefficient at t=99")
print(f"  sqrt(1 - alpha_bar_99) = {(1-alpha_bars[99]).sqrt():.4f} → noise coefficient at t=99")

# %% [markdown]
# ## Building Block 3: Forward Noising (The Reparameterization Trick)
#
# $$x_t = \underbrace{\sqrt{\bar\alpha_t}}_{\text{signal scale}} \cdot x_0 + \underbrace{\sqrt{1 - \bar\alpha_t}}_{\text{noise scale}} \cdot \epsilon$$
#
# This is **the** equation. You can jump to any timestep directly.
# No need to noise step by step — that's the beauty of the math.

# %%
def forward_noise(x_0, t, alpha_bars):
    """Add noise to x_0 to get x_t. Returns (x_t, epsilon)."""
    abar = alpha_bars[t]
    epsilon = torch.randn_like(x_0)
    x_t = abar.sqrt() * x_0 + (1 - abar).sqrt() * epsilon
    return x_t, epsilon

# Show x_0 getting progressively noisier
torch.manual_seed(0)
timesteps_to_show = [0, 10, 25, 50, 75, 99]

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for idx, t in enumerate(timesteps_to_show):
    ax = axes[idx // 3][idx % 3]
    x_t, eps = forward_noise(x_0, t, alpha_bars)
    
    colors = ['tab:blue'] * 8 + ['tab:red'] * 2
    ax.bar(range(dim), x_t.numpy(), color=colors, alpha=0.7)
    ax.set_title(f"t={t}  (ᾱ={alpha_bars[t]:.3f})")
    ax.set_ylim(-3, 3)
    ax.set_xticks(range(dim))
    ax.set_xticklabels([f"a{i}" for i in range(8)] + ["x", "y"], fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Forward Process: Clean Action → Pure Noise", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("Notice:")
print("  t=0:  The action vector is almost unchanged (ᾱ ≈ 1)")
print("  t=99: It's basically random noise (ᾱ ≈ 0)")
print("  The one-hot action type and coordinates get buried in noise")

# %% [markdown]
# ## Building Block 4: The Training Objective
#
# During training, the network sees $(x_t, t)$ and must predict $\epsilon$ — the noise that was added.
#
# $$\mathcal{L} = \text{MSE}(\epsilon, \; \epsilon_\theta(x_t, t))$$
#
# That's the entire loss function. Let's simulate one training step with a dummy "network".

# %%
# Simulate one training step
torch.manual_seed(7)

# 1. Sample a random timestep
t = torch.randint(0, T, (1,)).item()

# 2. Noise the clean action
x_t, epsilon_true = forward_noise(x_0, t, alpha_bars)

# 3. Pretend our network predicts something (random for now)
epsilon_pred = torch.randn_like(x_0)  # a real network would do better!

# 4. Compute MSE loss
loss = torch.nn.functional.mse_loss(epsilon_pred, epsilon_true)

print(f"Sampled timestep: t = {t}")
print(f"True noise (first 4):      {epsilon_true[:4].tolist()}")
print(f"Predicted noise (first 4): {epsilon_pred[:4].tolist()}")
print(f"MSE loss: {loss.item():.4f}")
print()
print("A random prediction gives loss ≈ 2.0 (two unit gaussians).")
print("A perfect prediction gives loss = 0.0.")
print("Your CNN policy has been trained to get this close to 0.")

# %% [markdown]
# ## Building Block 5: One Denoising Step (The Reverse Process)
#
# Given $x_t$ and a noise prediction $\epsilon_\theta$, compute $x_{t-1}$:
#
# $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \cdot \epsilon_\theta \right) + \sigma_t \cdot z$$
#
# Where:
# - First term removes the predicted noise (scaled appropriately)
# - $\sigma_t = \sqrt{\beta_t}$ adds a small amount of fresh noise (stochasticity)
# - $z \sim \mathcal{N}(0, I)$, but $z = 0$ at the final step ($t = 0$)

# %%
def denoise_step(x_t, t, epsilon_pred, betas, alphas, alpha_bars):
    """One DDPM reverse step: x_t → x_{t-1}."""
    alpha_t = alphas[t]
    abar_t = alpha_bars[t]
    beta_t = betas[t]
    
    # The mean of x_{t-1}
    coeff = beta_t / (1 - abar_t).sqrt()
    mean = (1 / alpha_t.sqrt()) * (x_t - coeff * epsilon_pred)
    
    if t == 0:
        return mean  # no noise at final step
    
    # Add stochastic noise
    sigma = beta_t.sqrt()
    z = torch.randn_like(x_t)
    return mean + sigma * z

# Demo: if we had a PERFECT noise predictor, one step would help
torch.manual_seed(42)
t = 50
x_t, epsilon_true = forward_noise(x_0, t, alpha_bars)

# Use the TRUE epsilon (perfect oracle predictor)
x_prev = denoise_step(x_t, t, epsilon_true, betas, alphas, alpha_bars)

print(f"x_0 (original):     {x_0[:5].tolist()}")
print(f"x_t (noised, t={t}): {x_t[:5].tolist()}")
print(f"x_{{t-1}} (denoised):  {x_prev[:5].tolist()}")
print()
print("One step moves x_t closer to x_0, but doesn't fully recover it.")
print("You need to chain many steps: x_T → x_{T-1} → ... → x_0")

# %% [markdown]
# ## Building Block 6: DDIM vs DDPM
#
# **DDPM** (what we just built): denoise one step at a time. 100 timesteps → 100 network calls. Slow.
#
# **DDIM** (Song et al., 2020): a deterministic variant that can **skip timesteps**.
#
# Instead of going $x_{99} \to x_{98} \to \ldots \to x_0$ (100 steps), DDIM picks a subset:
#
# $$x_{99} \to x_{89} \to x_{79} \to \ldots \to x_9 \to x_0 \quad \text{(10 steps!)}$$
#
# The formula is slightly different (no added noise, uses a different interpolation), but
# the idea is the same: predict noise, subtract it, jump to a much earlier timestep.
#
# **Your policy uses DDIM with 10 inference steps** — that's why inference is fast enough
# for real-time robot control. Training still uses all 100 timesteps for the loss.

# %%
# DDIM step (simplified, deterministic)
def ddim_step(x_t, t_now, t_next, epsilon_pred, alpha_bars):
    """DDIM step: jump from timestep t_now to t_next (t_next < t_now)."""
    abar_now = alpha_bars[t_now]
    abar_next = alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0)
    
    # Predict x_0 from x_t and epsilon
    x0_pred = (x_t - (1 - abar_now).sqrt() * epsilon_pred) / abar_now.sqrt()
    
    # Interpolate to get x_{t_next}
    x_next = abar_next.sqrt() * x0_pred + (1 - abar_next).sqrt() * epsilon_pred
    return x_next

# Demo: 10-step DDIM schedule (same as your policy)
ddim_timesteps = list(range(99, -1, -10))  # [99, 89, 79, ..., 9]
print(f"DDIM timesteps ({len(ddim_timesteps)} steps): {ddim_timesteps}")
print(f"vs DDPM: {list(range(99, -1, -1))} (100 steps)")
print()
print("Same network, 10x fewer calls. That's why your policy is fast.")

# %% [markdown]
# ## Putting It All Together: DDPMSchedule Class
#
# This is essentially what lives inside your diffusion policy.
# Two key methods:
# - `add_noise(x_0, t, noise)` — used during **training** to create noisy samples
# - `denoise_step(x_t, t, noise_pred)` — used during **inference** to clean up

# %%
class DDPMSchedule:
    """Minimal DDPM noise schedule for 1D action vectors."""
    
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x_0, t, noise=None):
        """Forward process: x_0 → x_t.
        
        Args:
            x_0: clean sample [dim] or [batch, dim]
            t: int timestep (0 to T-1)
            noise: optional pre-sampled noise (same shape as x_0)
        
        Returns:
            x_t: noisy sample
            noise: the noise that was added (for computing training loss)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        abar = self.alpha_bars[t]
        x_t = abar.sqrt() * x_0 + (1 - abar).sqrt() * noise
        return x_t, noise
    
    def denoise_step(self, x_t, t, noise_pred):
        """Reverse process: x_t → x_{t-1} (DDPM).
        
        Args:
            x_t: noisy sample at timestep t
            t: current timestep
            noise_pred: network's prediction of the noise in x_t
        
        Returns:
            x_{t-1}: slightly less noisy sample
        """
        alpha_t = self.alphas[t]
        abar_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        
        coeff = beta_t / (1 - abar_t).sqrt()
        mean = (1 / alpha_t.sqrt()) * (x_t - coeff * noise_pred)
        
        if t == 0:
            return mean
        
        sigma = beta_t.sqrt()
        z = torch.randn_like(x_t)
        return mean + sigma * z
    
    def full_denoise(self, x_T, noise_predictor):
        """Run full DDPM reverse chain: x_T → x_0.
        
        Args:
            x_T: pure noise sample
            noise_predictor: callable(x_t, t) → predicted noise
        
        Returns:
            x_0: denoised sample
        """
        x = x_T.clone()
        for t in range(self.T - 1, -1, -1):
            noise_pred = noise_predictor(x, t)
            x = self.denoise_step(x, t, noise_pred)
        return x

schedule = DDPMSchedule(T=100)
print(f"DDPMSchedule created with T={schedule.T}")
print(f"  betas range: [{schedule.betas[0]:.4f}, {schedule.betas[-1]:.4f}]")
print(f"  alpha_bar range: [{schedule.alpha_bars[-1]:.4f}, {schedule.alpha_bars[0]:.4f}]")

# %% [markdown]
# ### Demo: Full Round-Trip with an Oracle Predictor
#
# To verify our math works, we'll use a "cheating" noise predictor that knows the
# true noise. In your real policy, the CNN fills this role.

# %%
torch.manual_seed(123)

# Our clean action vector
x_0 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7])

# Add noise to get x_T (full noise)
x_t, true_noise = schedule.add_noise(x_0, t=99)
print(f"x_0:  {x_0.numpy().round(3)}")
print(f"x_99: {x_t.numpy().round(3)}")
print(f"x_99 looks like random noise — the action is completely buried.\n")

# Denoise with oracle predictor (cheating — knows the true noise)
# In practice: the CNN would predict different noise at each step
# For this demo, we use the original noise as an approximation
x_recovered = x_t.clone()
for t in range(99, -1, -1):
    # A perfect oracle for step 99's noise (approximate for other steps)
    x_recovered = schedule.denoise_step(x_recovered, t, true_noise)

print(f"x_0 recovered: {x_recovered.numpy().round(3)}")
print(f"x_0 original:  {x_0.numpy().round(3)}")
print(f"\nNote: Recovery isn't perfect because we used the same noise estimate")
print(f"for all timesteps. A trained network adapts its prediction per step.")

# %% [markdown]
# ## Your Turn
#
# Exercises:
# 1. Create a DDPMSchedule with `T=200` and a cosine-shaped alpha_bar (google "cosine noise schedule"). Plot it against the linear one.
# 2. Write a `ddim_denoise()` method that takes a list of timesteps (e.g., `[99, 89, ..., 9]`) and denoises using DDIM steps. Compare speed vs full DDPM.
# 3. Batch it: modify `add_noise` to handle a batch of different timesteps (one per sample). Hint: `t` becomes a tensor and you need to index `alpha_bars[t]` with broadcasting.

# %%
# Your code here

# %% [markdown]
# ## Tests
#
# Run these to verify your DDPMSchedule implementation is correct.

# %%
# ── Tests ──────────────────────────────────────────────────────────────────
sched = DDPMSchedule(T=100)

# Test 1: schedule shapes
assert sched.betas.shape == (100,), f"betas shape: {sched.betas.shape}"
assert sched.alphas.shape == (100,), f"alphas shape: {sched.alphas.shape}"
assert sched.alpha_bars.shape == (100,), f"alpha_bars shape: {sched.alpha_bars.shape}"

# Test 2: alpha_bar is decreasing
assert (sched.alpha_bars[1:] < sched.alpha_bars[:-1]).all(), "alpha_bar should be monotonically decreasing"

# Test 3: alpha_bar stays in (0, 1)
assert (sched.alpha_bars > 0).all() and (sched.alpha_bars < 1).all(), "alpha_bar must be in (0,1)"

# Test 4: add_noise at t=0 barely changes x_0
torch.manual_seed(0)
x_test = torch.ones(10)
x_noised, noise = sched.add_noise(x_test, t=0)
error_t0 = (x_noised - x_test).abs().max().item()
assert error_t0 < 0.05, f"At t=0, noise should be tiny, got max error {error_t0:.4f}"

# Test 5: add_noise at t=99 should make it look like noise
torch.manual_seed(0)
x_noised_99, _ = sched.add_noise(x_test, t=99)
# signal is sqrt(alpha_bar_99) ≈ 0.08, so x_test contribution is small
signal_strength = sched.alpha_bars[99].sqrt().item()
assert signal_strength < 0.15, f"Signal at t=99 should be small, got {signal_strength:.4f}"

# Test 6: add_noise with provided noise is deterministic
noise_fixed = torch.ones(10) * 0.5
x_a, _ = sched.add_noise(x_test, t=50, noise=noise_fixed)
x_b, _ = sched.add_noise(x_test, t=50, noise=noise_fixed)
assert torch.allclose(x_a, x_b), "add_noise with same noise should be deterministic"

# Test 7: denoise_step at t=0 returns mean (no stochastic noise)
torch.manual_seed(0)
x_t = torch.randn(10)
noise_pred = torch.randn(10)
result1 = sched.denoise_step(x_t, 0, noise_pred)
result2 = sched.denoise_step(x_t, 0, noise_pred)
assert torch.allclose(result1, result2), "denoise at t=0 should be deterministic"

# Test 8: alpha relationship
assert torch.allclose(sched.alphas, 1.0 - sched.betas), "alphas = 1 - betas"

print("All tests passed! ✓")
print()
print("Your DDPMSchedule correctly implements:")
print("  ✓ Linear beta schedule")
print("  ✓ Cumulative alpha-bar computation")
print("  ✓ Forward noising (reparameterization trick)")
print("  ✓ Single denoising step (reverse process)")
