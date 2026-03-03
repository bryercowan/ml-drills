# Concepts They'll Expect You to Explain

For each concept: what it IS in formal terms, and what YOU'D say
connecting it to your work.

### Attention Mechanism
FORMAL: Given queries Q, keys K, values V:
  Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
The dot product QK^T measures similarity between queries and keys.
Softmax normalizes to a probability distribution. The output is a
weighted sum of values, where weights = how relevant each key is
to each query.

YOUR VERSION: "In my VLM, Q/K/V projections are the layers I LoRA-tune.
They determine what the model attends to in the screenshot given the
current subgoal. In my CNN policy, cross-attention is how action tokens
attend to spatial image features — Q comes from the action being predicted,
K and V come from the ConvNeXt spatial map. This is what gives the model
coordinate grounding."

GOTCHA QUESTION: "Why divide by sqrt(d_k)?"
"Without it, the dot products grow large with dimension, pushing softmax
into regions with tiny gradients. The scaling keeps gradients healthy.
This is the same reason you see training instability with large models —
attention scores can saturate."

### Backpropagation
FORMAL: The chain rule applied recursively through a computation graph.
For any parameter w: dL/dw = (dL/da)(da/dw) where a is an intermediate
activation. Stack this through every layer.

YOUR VERSION: "When I freeze the vision encoder, I'm cutting off gradient
flow to those parameters — they don't update because dL/dw is never
computed for them. When I freeze CLIP, same thing. When I discovered
that LoRA on MLP layers destroyed spatial grounding, the issue was that
gradients through the MLP were overwriting Meituan's learned
visual-to-coordinate mapping with our smaller dataset."

### LoRA (Low-Rank Adaptation)
FORMAL: Instead of updating a full weight matrix W (d x d), decompose
the update as W + BA where B is (d x r) and A is (r x d), with r << d.
This restricts updates to a rank-r subspace. Mathematically related to
SVD — you're approximating the weight change with its top-r singular
vectors.

YOUR VERSION: "I use r=64, alpha=128 on the attention projections.
174M trainable params out of 8.9B total — 1.95%. The key discovery
was that WHICH layers you apply LoRA to matters as much as the rank.
Attention-only (61M params, 0.69%) outperformed attention+MLP (174M,
1.95%) because the MLP layers store spatial grounding that shouldn't
be disturbed with only 3,912 examples."

GOTCHA: "What does alpha do?"
"Alpha scales the LoRA output: delta_W = (alpha/r) * BA. Higher alpha
relative to r amplifies the adapter's contribution. It's effectively
a learning rate multiplier for the low-rank update."

### Diffusion Models
FORMAL: Learn to reverse a gradual noising process. Forward process adds
Gaussian noise over T steps: q(x_t|x_{t-1}) = N(sqrt(1-beta_t)*x_{t-1}, beta_t*I).
Training: predict the noise epsilon from a noised sample x_t.
Loss: ||epsilon - epsilon_theta(x_t, t)||^2.
Inference: start from pure noise, iteratively denoise using learned
predictions. DDIM sampling uses fewer steps (10 vs 100).

YOUR VERSION: "My CNN policy IS a diffusion policy — same architecture as
Chi et al. 2023. Instead of denoising images, I denoise action vectors.
The 10-dim action vector (8 action type dims + 2 coordinate dims) starts
as Gaussian noise and gets iteratively refined. I train with 100 diffusion
timesteps but sample with 10 DDIM steps at inference for speed. The
advantage over direct regression: diffusion handles multi-modal distributions.
If there are two valid buttons to click, direct regression predicts the
average (between the buttons, hitting nothing). Diffusion can represent
both modes and sample from one."

### KL Divergence
FORMAL: KL(P||Q) = E_P[log(P/Q)] = sum over x of P(x) * log(P(x)/Q(x)).
Measures how different distribution Q is from distribution P. Always >= 0.
Zero iff P = Q. NOT symmetric: KL(P||Q) != KL(Q||P).

YOUR VERSION: "In GRPO, KL(pi_theta || pi_ref) penalizes the policy for
diverging from the reference (SFT checkpoint). My KL hit 14-23 after just
2 updates — the policy was catastrophically forgetting the SFT behavior.
Even at lr=1e-7 with KL coeff 0.5, the raw gradient norms were ~1500.
The fundamental problem wasn't the KL penalty — it was that the reward
signal was too sparse to give the optimizer a good direction, so it just
drifted."

### Cross-Entropy Loss
FORMAL: H(p,q) = -sum of p(x) * log(q(x)). When p is the true label
(one-hot) and q is the model's predicted distribution, this simplifies to
-log(q(correct_class)). Equivalent to negative log-likelihood. This IS
your SFT training loss.

YOUR VERSION: "My SFT loss went from 3.29 to 0.09 over training. That
means the model went from assigning ~3.6% probability to the correct action
(e^-3.29 ≈ 0.037) to assigning ~91% probability (e^-0.09 ≈ 0.914).
Loss is computed only on the action tokens, not the image or prompt tokens."

### Policy Gradient / REINFORCE
FORMAL: The gradient of expected reward:
  nabla J = E[nabla log pi(a|s) * (R - b)]
where b is a baseline to reduce variance. Update the policy by increasing
the probability of actions that got above-average reward.

YOUR VERSION: "GRPO is a variant of this. Instead of a learned baseline,
it uses the group mean reward — sample G trajectories, rank by reward,
update toward the better ones. The problem in my case: with ~20% step-1
completion rate, almost all G trajectories got the same (bad) reward.
No variance in the group means the relative advantage is noise, so the
gradient is noise, so KL diverges."

### Residual Connections
FORMAL: f(x) = x + F(x) where F is the learned transformation. The
identity path (x) provides a gradient highway — during backprop,
dL/dx = dL/df * (1 + dF/dx). The "1" term means gradients always flow
through, preventing vanishing gradients in deep networks.

YOUR VERSION: "Both my VLM backbone (Qwen3-VL) and my CNN (ConvNeXt) use
residual connections throughout. My denoiser's ResidualBlock1D layers
each have skip connections. Without them, training a 50-layer ConvNeXt
would be impossible — gradients would vanish before reaching early layers."

### Batch Normalization / Layer Normalization
FORMAL: Normalize activations to zero mean and unit variance within a
batch (BatchNorm) or within a layer (LayerNorm). Stabilizes training by
preventing internal covariate shift — each layer sees consistently
scaled inputs regardless of what earlier layers are doing.

YOUR VERSION: "My transformer backbone uses LayerNorm (standard for
transformers — BatchNorm has issues with variable sequence lengths and
doesn't work well for autoregressive models). LayerNorm normalizes across
the feature dimension for each token independently."

### Tokenization
FORMAL: Converting raw input (text, images) into discrete tokens the model
processes. For text: BPE (Byte Pair Encoding) — start with individual bytes,
iteratively merge the most frequent adjacent pairs. For images (ViT): split
into fixed-size patches (e.g., 16x16 pixels), project each patch to a vector.

YOUR VERSION: "My compact action format is designed around the existing BPE
vocabulary. 'click(500,300)' uses tokens that already exist in the LLM's
vocabulary — parentheses, commas, numbers. No special tokens needed. This
is why it works better than custom XML formats with tokens the model has
rarely seen. For vision, EvoCUA's Qwen3-VL backbone uses a ViT that splits
the screenshot into patches and produces visual tokens."

### Transfer Learning / Fine-Tuning
FORMAL: Taking a model pretrained on a large general dataset and adapting
it to a specific task. The pretrained weights provide good initialization —
features learned on general data (edges, textures, object parts for vision;
syntax, semantics for language) transfer to new tasks.

YOUR VERSION: "My entire approach is transfer learning. EvoCUA-8B was
pretrained by Meituan on 82K GUI trajectories. I transfer that knowledge
to my specific benchmark via LoRA SFT on Claude's demonstrations. The
frozen vision encoder IS the transferred knowledge — it already understands
GUIs. I only adapt the attention layers to change what the model focuses on."

### Overfitting vs Underfitting
FORMAL: Overfitting = model memorizes training data, fails on new data
(training loss low, eval loss high). Underfitting = model hasn't learned
enough (both losses high). The gap between train and eval loss indicates
overfitting.

YOUR VERSION: "My SFT v5 had eval loss 0.087 at epoch 3.8 with training
loss around 0.09 — almost no gap, meaning minimal overfitting despite only
857 examples. This is because LoRA with frozen vision encoder is highly
regularized — the model can only change within a low-rank subspace of the
attention weights. The frozen backbone prevents catastrophic forgetting of
general GUI knowledge."

---

# Concepts to Quickly Review (Flash Card Style)

If you can explain each of these in 2 sentences, you're ready.

TRANSFORMER: Stack of attention + feedforward blocks with residual
connections and layer norm. Attention lets every token look at every
other token; feedforward processes each token independently.

SELF-ATTENTION vs CROSS-ATTENTION: Self = Q, K, V all come from the
same sequence. Cross = Q from one sequence, K/V from another (like
action tokens attending to image features in your CNN).

MULTI-HEAD ATTENTION: Run attention in parallel across h heads with
d_k = d_model/h. Each head can learn different patterns (one head
attends to nearby tokens, another to similar tokens, etc.).

POSITIONAL ENCODING: Transformers have no built-in notion of order.
Positional encodings (sinusoidal or learned) add position information
so the model knows token 3 comes before token 7.

VISION TRANSFORMER (ViT): Split image into 16x16 patches, embed each
patch as a token, run through a standard transformer. Your EvoCUA
backbone uses this for processing screenshots.

CONVNEXT: A modernized ConvNet that matches ViT performance. Uses
depthwise convolutions, inverted bottlenecks, GELU, LayerNorm. Your
CNN policy uses ConvNeXt-Tiny because it preserves spatial structure
better than ViT's [CLS] token for coordinate regression.

CLIP: Contrastive model trained on 400M image-text pairs. Learns to
align image and text in a shared embedding space. You use only the
text encoder as a frozen subgoal encoder.

FiLM CONDITIONING: output = scale * input + shift, where scale and
shift are predicted from a conditioning signal. You use this for
global conditioning ("what to do") while cross-attention handles
spatial conditioning ("where to do it").

CONTRASTIVE LEARNING: Train by pulling positive pairs together and
pushing negative pairs apart. CLIP does this with image-text pairs.
The loss is symmetric cross-entropy over a similarity matrix.

SFT: Show the model (input, correct_output) pairs. Minimize
cross-entropy on the output tokens. This is imitation learning /
behavioral cloning in robotics terms.

RLHF PIPELINE: SFT → reward model training → PPO optimization.
InstructGPT pipeline. DPO skips the reward model by directly
optimizing preferences. GRPO skips both by using group-relative
ranking.

REWARD SHAPING: Design RL rewards to be informative without being
exploitable. Your DOM-based rewards (+0.3 popup dismissed, +0.5
code entry, +1.0 step complete) give phase-level signals. Too sparse =
never learns. Too dense = games the reward.

GRADIENT ACCUMULATION: Run N forward/backward passes, sum gradients,
then update. Simulates batch_size * N without using more memory.
Your A100 config: micro-batch 4, grad_accum 4, effective batch 16.

GRADIENT CHECKPOINTING: Don't store all intermediate activations.
Recompute them during backward pass. Trades ~30% more compute for
~40% less GPU memory. You use this on CUDA but not MPS.

COSINE LEARNING RATE SCHEDULE: LR starts high, decays following a
cosine curve to near-zero. Warmup phase at the start prevents large
early updates from destabilizing training.

MIXED PRECISION (bf16/fp16): Use 16-bit floats for forward/backward
(speed + memory), 32-bit for weight updates (precision). bfloat16 on
CUDA (more dynamic range), fp16 on MPS.

INFERENCE vs TRAINING MEMORY: Training uses ~3-4x more memory than
inference — need to store activations for backprop, optimizer states
(AdamW stores 2 extra copies of parameters), and gradients.

EPOCHS vs STEPS: Epoch = one full pass through all data. Step = one
optimizer update. Your 857-example dataset with batch 4 and
grad_accum 4 = 53 steps per epoch.

TEMPERATURE: Scaling factor on logits before softmax. T>1 = softer
distribution (more random). T<1 = sharper (more deterministic).
T=0 = argmax (greedy). Used in sampling from your VLM policy.

TOP-K / TOP-P SAMPLING: Restrict sampling to the top-k tokens or
tokens comprising top-p probability mass. Prevents low-probability
garbage tokens from being selected.

BEAM SEARCH: Keep top-k sequences at each step, expand all, prune
back to top-k. More compute per step but finds higher-probability
sequences than greedy decoding.

KNOWLEDGE DISTILLATION: Train student model to match teacher model's
outputs. Original Hinton formulation uses soft labels (teacher's
probability distribution, not just argmax). Your distillation uses
hard labels (Claude's chosen action) which is behavioral cloning.

BEHAVIORAL CLONING: Imitation learning by supervised learning on
expert demonstrations. Your SFT on Claude's recordings IS behavioral
cloning. Weakness: doesn't learn to recover from mistakes because
the expert never makes them. This is why RL is supposed to help.

WEIGHT DECAY: Regularization that pulls weights toward zero.
Penalizes large weights. In AdamW, decoupled from the adaptive
learning rate (this is the "W" — the distinction from vanilla Adam).

DROPOUT: Randomly zero out neurons during training with probability p.
Forces the network to not rely on any single neuron. Acts as
ensemble regularization.
