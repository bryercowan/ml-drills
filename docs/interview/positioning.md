# Positioning

### What Makes You Unique
You have something most candidates don't: you built a complete research
system end-to-end, hit real walls, diagnosed them, and pivoted. Most
people with formal ML education have never trained a model on their own
data for a task they defined themselves.

### The Narrative Arc
"I started by trying to orchestrate existing models (failed — too slow,
too fragile). Pivoted to training from scratch (failed — not enough data,
wrong base model). Pivoted to distillation from a strong teacher (worked).
Then tried to improve with RL (GRPO failed — base too weak; DPO failed —
noisy labels). Each failure taught me something fundamental about how
these systems work. The project record is 6/30 steps with the distilled
model approaching that baseline."

This narrative shows: systematic approach, ability to diagnose failures,
willingness to pivot, understanding of the trade-offs between approaches.

### Target Companies and Why You Fit
- ANTHROPIC: They literally have Computer Use. You built an independent
  version. You understand their problem space deeply.
- PHYSICAL INTELLIGENCE (Pi): Your CNN policy is a diffusion policy for
  a digital environment. Same architecture as their robotics models.
- NOUS RESEARCH: They build the models you fine-tune. You understand
  the downstream use case.
- FIGURE AI / COVARIANT: Robotics companies where VLA experience
  directly applies.
- GOOGLE DEEPMIND ROBOTICS: RT-2 is a VLA. Your work IS this, in a
  digital domain.

### What to Say About Not Having a Formal ML Background
"I'm self-taught. I learned ML by building a complete VLA system —
distillation pipeline, RL training, diffusion-based policy, the whole
stack. My PROGRESS.md is a 2,000-line research journal documenting every
decision, dead end, and breakthrough. I don't have a degree in ML, but
I've shipped more ML systems than most people with one."

Don't apologize for being self-taught. Frame it as a strength: you learn
fast, you build things, you don't need to be taught — you figure it out.

---

# Appendix: Quick Numbers to Know

Your EvoCUA-8B: 8.9B params, ~16GB fp16, ~8GB int4
Your CNN policy: 34M params trainable, 63M frozen CLIP, ~100ms/action
Your LoRA: r=64 attention-only = 61M params (0.69% of base)
Training data: 3,912 clean examples from 9,186 raw recordings
Claude cost: $155 for full distillation dataset
Best result: 6/30 steps (Claude planner+policy)
A100-40GB: ~35 min for 5-epoch SFT, ~8 min per GRPO episode
3090 Ti-24GB: ~47 min for 50-epoch CNN training
Action format: ~5 tokens per action (vs 40-45 for competitors)
Benchmark: 30 steps, 29 challenge types, 3 versions

Attention: O(n^2 * d) compute, O(n^2) memory for sequence length n
LoRA rank 64: captures 64 dimensions of the weight update subspace
Diffusion: 100 training timesteps, 10 DDIM inference steps
KL divergence: 0 means identical distributions, >5 means severe drift
Cross-entropy loss 0.09: model assigns ~91% probability to correct action
SFT convergence: typically 3-5 epochs for distillation with <5K examples
