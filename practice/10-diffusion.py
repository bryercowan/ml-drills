# %% [markdown]
# # Drill 10: DDPM Noise Schedule
# Implement the core noise schedule and forward/reverse operations for DDPM.
#
# __init__:
#   - betas: linear schedule from beta_start to beta_end, shape (T,)
#   - alphas: 1 - betas
#   - alpha_bars: cumulative product of alphas (gets smaller over time)
#
# add_noise(x_0, t):
#   Given clean data x_0 and timestep t, produce noisy x_t:
#     x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
#   Return both x_t and the noise used.
#
# denoise_step(x_t, t, predicted_noise):
#   Reverse one step from t to t-1 using the DDPM formula:
#     x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * pred_noise) + sigma_t * z
#   where sigma_t = sqrt(beta_t) and z is random noise (zero for t=0).

# %%
import torch

# %%
class MyDDPMSchedule:
    def __init__(self, num_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        # TODO: create linear beta schedule from beta_start to beta_end, shape (T,)
        # TODO: compute alphas = 1 - betas
        # TODO: compute alpha_bars = cumulative product of alphas
        pass

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Forward diffusion: add noise to clean data.
        x_0: (B, ...) clean data
        t: (B,) integer timesteps
        Returns: (x_t, noise) both same shape as x_0
        """
        # TODO: gather alpha_bar values for each sample's timestep
        #       reshape for broadcasting: (B, 1, 1, ...) matching x_0 dims
        # TODO: sample noise ~ N(0, 1) same shape as x_0
        # TODO: compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # TODO: return (x_t, noise)
        pass

    def denoise_step(self, x_t: torch.Tensor, t: int, predicted_noise: torch.Tensor) -> torch.Tensor:
        """
        Reverse one step: go from x_t to x_{t-1}.
        x_t: (B, ...) noisy data at timestep t
        t: integer timestep (same for whole batch)
        predicted_noise: (B, ...) model's noise prediction
        Returns: x_{t-1}
        """
        # TODO: get alpha_t, beta_t, alpha_bar_t for this timestep
        # TODO: compute mean: (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * predicted_noise)
        # TODO: if t > 0, add sigma_t * z where sigma_t = sqrt(beta_t), z ~ N(0,1)
        # TODO: return x_{t-1}
        pass

# %%
# TEST
schedule = MyDDPMSchedule(num_timesteps=100)

# Check schedule properties
assert schedule.betas.shape == (100,), f"betas shape: {schedule.betas.shape}"
assert schedule.alphas.shape == (100,), f"alphas shape: {schedule.alphas.shape}"
assert schedule.alpha_bars.shape == (100,), f"alpha_bars shape: {schedule.alpha_bars.shape}"

# alpha_bars should be monotonically decreasing
diffs = schedule.alpha_bars[1:] - schedule.alpha_bars[:-1]
assert (diffs < 0).all(), "alpha_bars should be monotonically decreasing"

# alpha_bars should start near 1 and end near 0
assert schedule.alpha_bars[0] > 0.99, f"First alpha_bar should be ~1, got {schedule.alpha_bars[0]:.4f}"
assert schedule.alpha_bars[-1] < 0.5, f"Last alpha_bar should be small, got {schedule.alpha_bars[-1]:.4f}"

# Test add_noise
x_0 = torch.randn(4, 3, 8, 8)  # batch of tiny "images"
t_low = torch.zeros(4, dtype=torch.long)    # t=0 (barely noisy)
t_high = torch.full((4,), 99, dtype=torch.long)  # t=99 (very noisy)

x_t_low, noise_low = schedule.add_noise(x_0, t_low)
x_t_high, noise_high = schedule.add_noise(x_0, t_high)

assert x_t_low.shape == x_0.shape, f"add_noise shape mismatch: {x_t_low.shape}"
assert x_t_high.shape == x_0.shape, f"add_noise shape mismatch: {x_t_high.shape}"

# At t=0, x_t should be very close to x_0
low_diff = (x_t_low - x_0).abs().mean().item()
high_diff = (x_t_high - x_0).abs().mean().item()
assert low_diff < high_diff, f"Low-t should be closer to x_0: {low_diff:.4f} vs {high_diff:.4f}"

# Test denoise_step
x_t = torch.randn(4, 3, 8, 8)
pred_noise = torch.randn(4, 3, 8, 8)
x_prev = schedule.denoise_step(x_t, 50, pred_noise)
assert x_prev.shape == x_t.shape, f"denoise_step shape mismatch: {x_prev.shape}"

print(f"alpha_bars: first={schedule.alpha_bars[0]:.4f}, last={schedule.alpha_bars[-1]:.4f}")
print(f"Low-t noise diff: {low_diff:.4f}, High-t noise diff: {high_diff:.4f}")
print("diffusion schedule: ALL TESTS PASSED")
