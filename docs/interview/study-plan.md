# Study Plan

## Week 1 Daily Prep Plan

### Day 1 (Today): Read This Guide
- Read this entire guide out loud. Saying things out loud is different from
  reading silently — it activates verbal recall.
- Mark anything that doesn't feel natural. Those are your gaps.

### Day 2: Practice the Pitch
- Record yourself giving the 30-second and 2-minute versions. Listen back.
- Do it again until it sounds like you, not like you're reading.
- Practice the "why vision-only" and "why distillation" answers out loud.

### Day 3: Code the Basics
- Implement self-attention from scratch in PyTorch (30 min)
- Implement a simple LoRA layer (15 min)
- Implement cross-entropy loss from scratch (15 min)
- Run these in a notebook so you can explain "I implemented these to make
  sure I understand the fundamentals"

### Day 4: Paper Familiarity
- Read Attention Is All You Need Section 3 (30 min)
- Read LoRA Abstract + Section 3 (20 min)
- Read Diffusion Policy Abstract + Section 3 (30 min)
- For each: write one sentence on how it connects to your CUA work

### Day 5: Systems Knowledge
- Do the first 2-3 lessons at workatafrontierlab.com (1-2 hours)
- Focus on: memory math (why models OOM), the training vs inference gap
- Be able to say: "EvoCUA-8B is ~16GB in fp16, LoRA adds ~340MB for
  the adapter. Training needs ~48GB with optimizer states + activations +
  gradients, which is why we use the A100-40GB with gradient checkpointing."

### Day 6: Your Failure Stories
- Practice telling the GRPO failure story (2 min version)
- Practice telling the DPO failure story (2 min version)
- Practice telling the LoRA target module discovery (2 min version)
- These are your BEST interview material. Failures that you diagnosed and
  learned from demonstrate research maturity.

### Day 7: Mock Interview
- Find someone (friend, ChatGPT voice mode, mirror) and do a mock:
  - "Tell me about your project" (2 min)
  - "How does attention work?" (1 min)
  - "Why did your RL training fail?" (2 min)
  - "Design a GUI automation system" (10 min)
  - "What would you do differently?" (2 min)

---

## Weeks 2-4 Deeper Prep (For In-Person)

### Week 2: Build From Scratch
- Karpathy Zero to Hero: do Micrograd (backprop engine) and Building GPT
  (transformer from scratch). Code along, don't just watch.
  https://karpathy.ai/zero-to-hero.html
- After finishing: you can now say "I've implemented backpropagation and
  the transformer architecture from scratch" in interviews.

### Week 3: RL + Systems Depth
- ARENA 3.0 Chapter 2: implement PPO from scratch
  https://github.com/callummcdougall/ARENA_3.0
- Finish workatafrontierlab.com remaining tracks
- Read DeepSeek-R1 paper Section 2 (GRPO pipeline)
- Read DPO paper Sections 3-4

### Week 4: Polish
- Clean up your CUA GitHub repo for public viewing
- Write a technical blog post: "Lessons from Building a Vision-Only
  GUI Agent: 30+ Experiments in Distillation, SFT, and RL"
- Practice your presentation: 10-minute research talk format
- Do another mock interview with harder probing questions
