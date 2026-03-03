# Likely Interview Questions & Your Answers

### ML Systems Design
"Design a system that automates GUI interactions using vision."

Walk through YOUR architecture:
1. Frame it as a robotics problem — perception → planning → action
2. Two-tier split: expensive cloud model for strategic reasoning
   (runs infrequently), cheap local model for fast execution
   (every action)
3. Data pipeline: expert demonstrations → filtering → distillation
4. Training: SFT first to establish baseline, then RL to improve
5. Evaluation: real environment with DOM-based rewards
6. Deployment: local model only, planner frequency decreases as
   model improves

They'll probe: "How do you handle failures?" → motor intent circuit
breaker, subgoal re-planning. "How do you handle new applications?" →
record Claude on new app, filter, retrain. Same pipeline.
"What's the latency budget?" → 100ms CNN vs 500ms VLM vs 1s API.
"How do you evaluate?" → DOM-based step completion on live benchmark.

### Paper Discussion
They may ask you to discuss a paper. The 5 you MUST be able to discuss:

1. ATTENTION IS ALL YOU NEED (Vaswani et al. 2017)
   Key ideas: self-attention replaces recurrence, multi-head attention,
   positional encoding, encoder-decoder structure.
   Your angle: "My VLM backbone is a decoder-only transformer. The
   attention mechanism is what I LoRA-tune. Multi-head attention is
   why different heads can attend to different image regions."

2. LORA (Hu et al. 2021)
   Key ideas: low-rank decomposition of weight updates, no inference
   latency (merge at deployment), comparable performance to full
   fine-tuning at 0.1-1% of parameters.
   Your angle: "I ran extensive experiments with different LoRA configs.
   The key finding was target module selection matters more than rank —
   attention-only vs attention+MLP was the difference between 2 steps
   and 0 steps."

3. DIFFUSION POLICY (Chi et al. 2023)
   Key ideas: diffusion models for action prediction, action chunking,
   handles multi-modal action distributions, CNN backbone with spatial
   features.
   Your angle: "My CNN policy is directly inspired by this. Same
   architecture — ConvNeXt encoder, DDPM denoiser, cross-attention
   conditioning. I adapted it from robotics manipulation to GUI
   automation."

4. DEEPSEEK-R1 / GRPO (DeepSeek 2024)
   Key ideas: Group Relative Policy Optimization, no critic needed,
   relative advantage within groups, applied to reasoning tasks.
   Your angle: "I attempted 7 GRPO configurations. The failure mode
   taught me that GRPO needs a base policy competent enough to generate
   within-group variance. DeepSeek started from ~30% solve rate; my
   base was at ~20% for step 1 alone."

5. DPO (Rafailov et al. 2023)
   Key ideas: direct optimization of preferences without reward model,
   Bradley-Terry preference model, implicit reward, beta controls
   conservatism.
   Your angle: "I attempted 6 DPO configurations. The failure taught
   me that DPO is extremely sensitive to label noise — our SSIM-based
   labels had maybe 15-20% noise rate which corrupted the implicit
   reward model."

### Coding Questions
If they ask you to implement something ML-related:
- Self-attention from scratch (matrix multiply Q, K^T, scale, softmax,
  multiply by V) — practice this in PyTorch, it's 5 lines
- Simple training loop (forward, loss, backward, optimizer step)
- LoRA layer (linear layer that adds BA to the output)
- Softmax from scratch (exp(x - max(x)) / sum, with numerical stability)
- Cross-entropy loss from scratch

### "What would you do differently?"
This is a research taste question. Good answers:

"I'd invest more in data quality upfront. Our first 857 hand-curated
examples outperformed 3,912 auto-filtered ones because the LoRA target
was wrong — but the lesson is that data curation is worth 10x the
engineering effort of training tweaks."

"I'd try rejection sampling fine-tuning (RFT) before GRPO. RFT generates
many rollouts, keeps only the successful ones, and does more SFT. It's
less elegant but more robust when the base policy is weak. You build up
the success rate gradually until GRPO becomes viable."

"For the CNN policy, I'd explore replacing the diffusion head with direct
regression + mixture density network. Diffusion gives multi-modal
predictions but at the cost of 10 denoising steps. An MDN could handle
multi-modality in a single forward pass."

### "What's the hardest bug you found?"
"The LoRA target module discovery. SFT v7 and v8 both scored 0 steps
despite low training loss and 4.6x more data than v5 which scored 2 steps.
After eliminating data quality, learning rate, and training duration, I
compared the adapter configs and found v5 used attention-only LoRA while
v7/v8 included MLP layers. The MLP layers in EvoCUA store spatial grounding
from Meituan's 82K-trajectory pretraining. By including them in LoRA, we
overwrote that knowledge with our 3,912 examples — not enough data to
relearn grounding from scratch. The fix was trivial (remove 3 strings from
a config), but finding it required understanding the architectural role of
attention vs MLP in vision-language models."

Also strong: "Radix UI radio buttons. The benchmark used React's Radix UI
library which renders radio buttons as <button role='radio'> with
data-state='checked' instead of standard HTML <input type='radio'>. Our
DOM reward queries for input[type=radio] returned nothing — weeks of
broken modal detection in GRPO training traced back to this. The fix
was querying [role=radio][data-state=checked]."
