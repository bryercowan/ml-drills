# Your Project in Interview Language

When they ask "tell me about your work," you need a crisp narrative.
Practice saying these OUT LOUD until they feel natural.

### The 30-Second Pitch
"I built a vision-only GUI automation agent — a digital humanoid for legacy
software. It watches the screen as pixels and outputs mouse/keyboard actions,
no DOM or API access. The architecture is a two-tier system: a Claude Sonnet
planner for strategic reasoning, and a local 8B VLA model distilled from
Claude's demonstrations for fast execution at 0.5 seconds per action with
zero API cost. I also built a parallel CNN policy track — a 34M parameter
diffusion policy inspired by robotics, running at 100ms per action. The core
insight is that the browser is the simulation environment — same as MuJoCo
for robots, but with no sim-to-real gap."

### The 2-Minute Version (add these details)
- "I recorded Claude Sonnet operating GUIs — about 9,000 demonstration
  actions across 30 different UI challenge types. Then I built a four-stage
  filtering pipeline to clean the data: removing stuck steps, scroll
  oscillations, reset loops, and no-effect actions. That took 9,186 raw
  records down to 3,912 clean training examples."

- "For the VLM policy, I fine-tuned EvoCUA-8B using LoRA on the attention
  layers only — q, k, v, and o projections. I discovered that including
  the MLP layers destroyed the model's spatial grounding because Meituan
  had trained those on 82K GUI trajectories. The MLP stores the
  visual-to-coordinate mapping; by only training attention, I changed WHAT
  the model attends to while preserving WHERE it clicks."

- "For the CNN policy, I used a ConvNeXt vision encoder with spatial feature
  preservation — not global average pooling, which destroys coordinate
  information — plus a frozen CLIP text encoder for subgoal conditioning via
  cross-attention, feeding into a diffusion denoising head. This is directly
  inspired by Chi et al.'s Diffusion Policy for robotics."

- "I attempted both GRPO and DPO for RL improvement. Seven GRPO configs all
  hit monotonically increasing KL divergence — the base policy was too weak
  to generate enough positive signal within groups. Six DPO attempts failed
  because SSIM-based preference labels were too noisy. The core lesson:
  RL amplifies existing capability, it doesn't create it. You need a strong
  SFT baseline first."

### When They Probe Deeper — Ready Answers

**"Why vision-only instead of DOM/accessibility APIs?"**
"DOM selectors are brittle — a UI redesign breaks every automation. Desktop
apps have no DOM at all. Remote desktops are just pixels. Vision-only means
one architecture works on any GUI: web, desktop, mobile, terminal, remote
desktop. The DOM is used only during training as a reward signal, like physics
state in robot simulation. At inference, the model sees only pixels."

**"Why distillation instead of training from scratch?"**
"Claude Sonnet achieved 6/30 benchmark steps as planner+policy — proving the
task is solvable. But at $5/run and 1s/action, it can't scale. The local model
doesn't need to independently reason — it needs to replicate Claude's behavior.
Distillation is the standard robotics approach: collect expert demonstrations,
then imitation learning. We use Claude as the expert demonstrator."

**"Why did GRPO fail?"**
"GRPO needs variance within each group — some trajectories succeed, some fail.
With the model completing step 1 only ~20% of the time, almost every trajectory
in a group got negative reward. There was no positive signal to amplify. The
algorithm could only suppress behaviors, which destroyed the SFT knowledge.
DeepSeek started GRPO from a model that already solved ~30% of problems. Our
base was nowhere near that threshold. The gradient norms were ~1500 even at
lr=1e-7 — the loss landscape is extremely steep around a weak policy."

**"Why did DPO fail?"**
"Two reasons. First, our preference labels were based on SSIM — if the
screenshot changed, the action was 'chosen.' But clicking a wrong button
that triggers a visual toast counts as 'chosen' even though it's a bad action.
The Bradley-Terry preference model assumes clean labels. Second, with
sum log probabilities over long VLM sequences, the gradient magnitudes were
either enormous (causing divergence) or near-zero with length normalization
(causing no learning). We tried 6 configurations including DPO+SFT auxiliary
loss — the SFT component worked but the DPO margin went negative."

**"What's the compact action format and why?"**
"Five tokens per action: click(500,300), type(ABC123), scroll(down), key(Enter),
done(). Coordinates are 0-999 relative, resolution-independent. Standard formats
like UI-TARS use ~40-45 tokens per action with verbose XML. Ours is 8x more
efficient. This matters for inference speed, RL trajectory collection, and
context window utilization. Inspired by Standard Intelligence's FDM-1."

**"Tell me about the CNN policy architecture."**
"Three components. The vision encoder is ConvNeXt-Tiny producing a (B, 768, 11, 20)
spatial feature map — 220 spatial tokens with learned 2D positional embeddings.
I preserve the spatial layout instead of average pooling because coordinate
regression needs to know WHERE things are, not just WHAT's there. The text
encoder is a frozen CLIP model that turns subgoal strings into 512-dim vectors,
projected down to 128-dim. The denoiser uses cross-attention — action tokens
attend to spatial image tokens, so each predicted action can 'look at' different
screen regions. Plus FiLM conditioning from the global context for 'what to do'
while cross-attention handles 'where to do it.' Total: 34M trainable params,
63M frozen CLIP params."

**"What's the motor intent tier?"**
"The CNN can't read text or reason about challenge types — it has no language
model. Motor intent bridges the planner's strategic analysis and the CNN's
pixel-level execution. It maps challenge_type to concrete mechanical instructions
via template lookup (~0ms) for 25 known patterns, with Haiku API fallback for
novel situations. It also acts as a circuit breaker: if 3+ actions have no
effect, it forces a subgoal change. This is the classic robotics separation —
deliberative planning at low frequency, reactive control at high frequency."
