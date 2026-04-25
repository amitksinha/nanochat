# My nanochat Run — Training a 1.38B GPT-2 Capability LLM for ~$100

This document covers how I ran [Karpathy's nanochat](https://github.com/karpathy/nanochat) end-to-end: pretraining a 1.38B parameter model on 8×H100 GPUs on RunPod, followed by SFT (supervised fine-tuning) to turn it into a chat assistant, and then exploring the architecture hands-on in a Jupyter notebook.

---

## Results

| Stage | Metric | Score |
|-------|--------|-------|
| Base model (5568 steps) | CORE | 0.2454 |
| Base model | Val BPB | 0.717 |
| SFT model (483 steps) | ARC-Easy | 64.2% |
| SFT model | ARC-Challenge | 49.7% |
| SFT model | MMLU | 36.4% |
| SFT model | SpellingBee | 99.6% |
| SFT model | ChatCORE | 36.4% |

**Total wall-clock training time: 3h 39m**  
**Total cost: ~$100 on RunPod (8×H100 SXM at ~$24/hr)**

---

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.10+
- A RunPod (or similar) account for GPU training

### Install dependencies

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# For GPU training (CUDA)
uv sync --extra gpu
source .venv/bin/activate

# For local inference / notebook on Mac
uv sync --extra cpu
source .venv/bin/activate

# For development (adds pytest, matplotlib, jupyter, etc.)
uv sync --extra cpu --group dev
```

---

## Cloud GPU Training (RunPod — 8×H100)

### 1. Provision the pod

- Go to [runpod.io](https://www.runpod.io) → Secure Cloud
- Select **8×H100 SXM** (~$24/hr)
- Container image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container disk: 50 GB** (ephemeral — wiped on pod restart)
- **Volume disk: 150 GB** (persistent — survives pod stop/restart, ~$1/day)
  - Mount volume at `/workspace`

> **Important:** Keep the volume at `/workspace` alive even when the pod is stopped. It costs only ~$1/day and is where all checkpoints are saved.

### 2. Set up the environment

SSH into the pod (or use the web terminal):

```bash
# Clone the repo into the volume so it persists
cd /workspace
git clone https://github.com/your-fork/nanochat.git
cd nanochat

# Install uv and Python deps
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync --extra gpu
source .venv/bin/activate

# Install hf_transfer for fast HuggingFace dataset downloads
uv pip install hf_transfer
```

### 3. Configure the run

Set the checkpoint directory to the persistent volume (important — the default goes to `~/.cache` on the container disk, which is wiped on restart):

```bash
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
export WANDB_RUN=my-run   # wandb run name, or use "dummy" to skip wandb
```

> **Note:** If using `speedrun.sh`, the script has its own `export NANOCHAT_BASE_DIR=...` line. Comment it out so your env var takes effect:
> ```bash
> sed -i 's/^export NANOCHAT_BASE_DIR=/#export NANOCHAT_BASE_DIR=/' runs/speedrun.sh
> ```

### 4. Run pretraining + SFT

Start a tmux session so the run survives if your SSH connection drops:

```bash
tmux new -s train
```

Then inside tmux:

```bash
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
export WANDB_RUN=my-run
cd /workspace/nanochat
bash runs/speedrun.sh
```

The speedrun script runs the full pipeline:
1. Tokenizer training (~2 min)
2. Tokenizer evaluation
3. **Base model pretraining** — d24, FP8, ~2h 55m
4. Base model evaluation (CORE score, BPB, samples)
5. **SFT fine-tuning** — ~40 min on SmolTalk + MMLU + GSM8K + SpellingBee
6. SFT evaluation
7. Generates `report.md`

### 5. Monitor training

GPU utilization:
```bash
watch -n 2 nvidia-smi
```

Training logs (tail the tmux session or watch stdout). Key metrics printed each `eval_every` steps:
- `val_bpb` — validation bits per byte (lower is better; GPT-2 ~1.0, our model reaches ~0.717)
- `train/mfu` — Model FLOP Utilization (we achieved ~34% in base, ~50% in SFT)
- `eta` — estimated time remaining

### 6. Disk space management

Each FP8 checkpoint is ~9.5 GB:
- Model weights: ~4 GB
- Optimizer state (8 shards × 685 MB): ~5.5 GB

With `--save-every=500` the script keeps the last few checkpoints. After training completes, delete optimizer states to free space:

```bash
rm /workspace/nanochat_cache/base_checkpoints/d24/optim_*.pt
```

### 7. Check checkpoints

```bash
ls -lh /workspace/nanochat_cache/base_checkpoints/d24/
ls -lh /workspace/nanochat_cache/chatsft_checkpoints/d24/
```

### 8. Copy results to your laptop

```bash
# On the RunPod pod:
git add -f report.md
git commit -m "add training report"
git push

# On your Mac:
git pull
```

Or use `scp`:
```bash
scp -P <PORT> root@<POD_IP>:/workspace/nanochat_cache/chatsft_checkpoints/d24/model_*.pt ~/Downloads/
```

### 9. Stop the pod

Once training is done, **stop** (not terminate) the pod from the RunPod console. The volume persists. You can restart a cheap CPU pod later for inference, or restart the H100 pod to continue training.

---

## Running Inference on Your Mac

Once you have a trained SFT checkpoint locally:

```bash
cd ~/Documents/nanochat
uv sync --extra cpu
source .venv/bin/activate

# Set the checkpoint directory
export NANOCHAT_BASE_DIR=~/Documents/nanochat_cache

# CLI chat
python -m scripts.chat_cli -p "What is the capital of France?"

# Web UI (ChatGPT-style interface)
python -m scripts.chat_web
# Visit http://localhost:8000
```

---

## Educational Notebook — Understanding the Transformer

The notebook `transformer_learning.ipynb` builds the GPT architecture from scratch in 20 steps, then trains a real character-level GPT on Tiny Shakespeare.

### Quick start

```bash
cd ~/Documents/nanochat
uv sync --extra cpu --group dev
source .venv/bin/activate
pip install matplotlib jupyter

jupyter notebook transformer_learning.ipynb
```

Run all cells (Kernel → Restart & Run All). Training in Step 17 takes **~5 minutes on an M4 Mac**.

### What the notebook covers

**Part 1 — Architecture walkthrough (Steps 1–13)**

| Step | Topic |
|------|-------|
| 1 | Tokenization — text → token IDs (BPE via tiktoken) |
| 2 | Token Embeddings — token IDs → vectors (`wte`) |
| 3 | Q, K, V Projections |
| 4 | Scaled Dot-Product Attention |
| 5 | Causal Masking |
| 6 | Multi-Head Attention |
| 7 | The MLP Block (expand 4×, relu², contract) |
| 8 | Residual Connections & RMSNorm |
| 9 | A Full Transformer Block |
| 10 | Stacking Layers |
| 11 | Next Token Prediction (LM Head) |
| 12 | The Full Picture — `TinyGPT` class |
| 13 | Running nanochat's actual production model |

**Part 2 — Train a GPT on Tiny Shakespeare (Steps 14–20)**

| Step | Topic |
|------|-------|
| 14 | Download Tiny Shakespeare + character-level tokenizer |
| 15 | Minibatch sampling (`get_batch`) |
| 16 | `GPTShakespeare` — full model with positional embeddings (`wte + wpe`) |
| 17 | Training loop — AdamW, cross-entropy loss, validation |
| 18 | Loss curves (matplotlib) |
| 19 | Text generation samples |
| 20 | Discussion — what affects model performance |

**Model hyperparameters (Part 2):**

| Param | Value |
|-------|-------|
| `n_layer` | 4 |
| `n_embd` | 128 |
| `n_head` | 4 |
| `block_size` | 128 |
| `batch_size` | 64 |
| `max_iters` | 5000 |
| Total params | ~820K |
| Training time | ~5 min (M4 MPS) |

---

## Key Architecture Concepts

### Why positional embeddings matter

Without `wpe`, attention is permutation-invariant — the model cannot tell whether "king" appears at position 0 or position 50. Learned positional embeddings (one vector per position 0…block_size-1) give the model this information.

```python
pos_emb = self.wpe(torch.arange(T))   # [T, n_embd]
x = self.wte(idx) + pos_emb           # [B, T, n_embd]
```

### nanochat vs our TinyGPT

| Feature | TinyGPT (notebook) | nanochat (production) |
|---------|-------------------|----------------------|
| Positional encoding | Learned absolute | Rotary embeddings (RoPE) |
| Attention stability | Basic softmax | QK-norm |
| Context window | Full | Sliding window (SSSL pattern) |
| Inference | Recompute all | KV cache |
| Precision | float32 | bfloat16 / FP8 |
| Optimizer | AdamW | Muon + AdamW |
| Multi-GPU | No | DDP via torchrun |
| Parameters | ~820K | 1.38B |

---

## Training Configuration (my run)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN \
    --save-every=500 \
    --resume-from-step=500
```

- **depth=24** → 1.38B parameters (24 transformer layers, width auto-calculated)
- **fp8** → FP8 precision via TransformerEngine (enables tensor-core throughput on H100)
- **target-param-data-ratio=8** → compute-optimal: train on 8× as many tokens as parameters (~5.8B tokens)
- **device-batch-size=16** → per-GPU batch; with 8 GPUs → effective batch of ~1M tokens/step

---

## File Structure

```
nanochat/
├── nanochat/
│   ├── gpt.py                  # The GPT nn.Module (production model)
│   ├── tokenizer.py            # BPE tokenizer (32K vocab)
│   ├── checkpoint_manager.py   # Save/load checkpoints
│   ├── engine.py               # KV-cache inference engine
│   ├── execution.py            # Python tool execution (for SpellingBee)
│   └── ...
├── scripts/
│   ├── base_train.py           # Pretraining script
│   ├── chat_sft.py             # SFT fine-tuning
│   ├── chat_eval.py            # Evaluation on tasks
│   ├── chat_cli.py             # Terminal chat interface
│   └── chat_web.py             # Web UI (http://localhost:8000)
├── tasks/
│   ├── gsm8k.py                # Grade school math (8K problems)
│   ├── mmlu.py                 # Multiple choice, broad topics
│   ├── spellingbee.py          # Letter-counting with Python verification
│   └── smoltalk.py             # SmolTalk conversation dataset
├── runs/
│   └── speedrun.sh             # Full pipeline script (tokenize → pretrain → SFT → eval)
├── transformer_learning.ipynb  # Educational notebook (this run)
├── report.md                   # Auto-generated training report
└── MY_RUN.md                   # This file
```

---

## Troubleshooting

**"No checkpoints found" / training starts from scratch**  
→ The `NANOCHAT_BASE_DIR` env var was not set before running `speedrun.sh`, or the script's own `export NANOCHAT_BASE_DIR=` line overrode it. Always comment out that line and set the env var manually.

**Disk full during training**  
→ Each FP8 checkpoint is ~9.5 GB. With `--save-every=500`, 12 checkpoints = ~115 GB. Use a 150 GB volume and delete optimizer shards after training: `rm .../optim_*.pt`

**`hf_transfer` not installed (SFT fails)**  
→ The venv only has `uv pip` available, not `pip` directly. Run: `uv pip install hf_transfer`

**SSH key not working after pod restart**  
→ RunPod containers are ephemeral. Add your SSH public key to RunPod Settings → SSH Keys so it propagates to new containers automatically.

**tmux "sessions should be nested with care" error**  
→ You're already inside a tmux session. Just run commands directly — no need to create a new tmux session inside an existing one.

**`git push` auth failure**  
→ GitHub no longer accepts passwords. Use a Personal Access Token (Settings → Developer settings → Tokens → classic, `repo` scope) as the password.

---

## How to Test — Notebook Model (Tiny Shakespeare GPT)

After training completes in Step 17 of the notebook, the `gpt` object is a fully trained character-level GPT. Use the `sample()` function defined in Step 19 to generate text.

### What to expect

At ~820K parameters with character-level tokenization and a 128-character context window, the model learns Shakespeare's **style and format** but not narrative coherence. Concretely:

- **Will produce:** correct spelling, Shakespearean vocabulary ("thee", "thou", "hath", "prithee"), speaker label formatting, plausible short phrases of 5–10 words
- **Won't produce:** coherent meaning across more than a sentence or two, accurate completion of famous quotes, plot or character continuity

A well-trained run (val loss ~1.5) should generate recognizable dialogue within the first 2–3 lines of any Shakespeare character prompt — that's the clearest sign the model learned the data distribution rather than memorizing it.

### Best prompts to use

These match the training data format closely and produce the best results:

```python
# Character name prompts — appear constantly in training data
sample("ROMEO:")
sample("KING:")
sample("HAMLET:")
sample("JULIET:")
sample("OTHELLO:")

# Stage direction style
sample("Enter ")
sample("ACT II. SCENE ")

# Common speech openers
sample("My lord, I ")
sample("I pray thee, ")
sample("What means this, ")

# Short famous fragments
sample("To be or not")
sample("All the world's a ")
```

### Prompts to avoid

```python
# Modern English — never seen in training data, will produce gibberish
sample("Hey, what's up")
sample("The weather today")

# Very long prompts — eats into the 128-char context window,
# leaving less room for the model to generate
sample("This is a very long prompt that uses up most of the context window and...")
```

### Adjusting generation quality

The `sample()` function accepts `temperature` and `top_k` parameters:

```python
# More creative / varied output (higher temperature)
sample("ROMEO:", temperature=1.0, top_k=50)

# More conservative / predictable output (lower temperature)
sample("ROMEO:", temperature=0.5, top_k=20)

# Greedy-ish (almost always picks the most likely next character)
sample("ROMEO:", temperature=0.2, top_k=5)
```

Lower temperature tends to produce more grammatically correct but repetitive text. Higher temperature produces more variety but occasionally nonsensical words. `temperature=0.8, top_k=40` (the default) is a good balance for this model size.

### Reading the loss curves

The loss curve saved to `loss_curves.png` tells you how well training went:

| Val loss at step 5000 | What it means |
|---|---|
| > 2.0 | Model barely learned — check device, batch size, learning rate |
| 1.6 – 2.0 | Reasonable for this size; text will look vaguely Shakespearean |
| 1.4 – 1.6 | Good convergence; clear Shakespeare style in output |
| < 1.4 | Excellent — try generating longer sequences |

A large gap between train and val loss (> 0.3) indicates overfitting — the model memorized the training set. At this model size on 1M characters that is unlikely after only 5000 steps.

---

## Changes from Original nanochat

This fork makes the following changes to [karpathy/nanochat](https://github.com/karpathy/nanochat). All changes are confined to three files and do not touch the model architecture or training code.

### 1. `runs/speedrun.sh` — checkpoint control

**Change 1: commented out `NANOCHAT_BASE_DIR`**

```diff
-export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
+#export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
```

*Why:* The original script unconditionally overwrites `NANOCHAT_BASE_DIR` to `~/.cache/nanochat` on the container disk (ephemeral). Commenting it out lets you set the variable externally before running the script — e.g. `export NANOCHAT_BASE_DIR=/workspace/nanochat_cache` — so checkpoints go to the persistent volume and survive pod restarts.

**Change 2: added `--save-every` and `--resume-from-step`**

```diff
-torchrun ... -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
+torchrun ... -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN --save-every=500 --resume-from-step=500
```

*Why:* The original script saves only the final checkpoint. With 5568 training steps on H100, a single checkpoint failure loses everything. `--save-every=500` writes a checkpoint every 500 steps. `--resume-from-step=500` allows resuming from step 500 if the run was interrupted (e.g. disk full) — the script replays all setup steps (tokenizer, dataset) but skips the first 500 training steps.

> **Note on disk space:** Each FP8 checkpoint is ~9.5 GB (4 GB model + 8 × 685 MB optimizer shards). With `--save-every=500` over 5568 steps that's up to ~105 GB. Delete optimizer shards after training completes: `rm .../optim_*.pt`

---

### 2. `transformer_learning.ipynb` — educational notebook (new file)

Added a 65-cell Jupyter notebook that teaches the GPT transformer architecture from scratch and then trains a small model end-to-end.

**Part 1 — Architecture walkthrough (Steps 0–13)**

Builds the transformer piece by piece using plain PyTorch, connecting each step to the corresponding code in `nanochat/gpt.py`:

| Step | Topic |
|------|-------|
| 0 | Setup and device detection (MPS on Apple Silicon) |
| 1 | Tokenization — BPE via `tiktoken` |
| 2 | Token embeddings (`wte`) |
| 3 | Q, K, V projections |
| 4 | Scaled dot-product attention |
| 5 | Causal masking |
| 6 | Multi-head attention |
| 7 | MLP block (expand 4×, relu², contract) |
| 8 | Residual connections and RMSNorm |
| 9 | Full transformer block |
| 10 | Stacking N layers |
| 11 | LM head — next token prediction |
| 12 | `TinyGPT` — complete minimal model |
| 13 | Instantiating nanochat's production `GPT` class |

**Part 2 — Train on Tiny Shakespeare (Steps 14–20)**

Trains a real character-level GPT from scratch, optimized to run in ~5 minutes on an M4 Mac:

| Step | Topic |
|------|-------|
| 14 | Download Tiny Shakespeare + character-level tokenizer (65 tokens) |
| 15 | `get_batch()` — random minibatch sampling |
| 16 | `GPTShakespeare` — full model with positional embeddings (`wte + wpe`) |
| 17 | Training loop — AdamW, cross-entropy loss, validation every 250 steps |
| 18 | Loss curves (matplotlib, saved to `loss_curves.png`) |
| 19 | Text generation — 3 Shakespeare prompts |
| 20 | Discussion — 6 factors that affect model performance |

Model hyperparameters: `n_layer=4`, `n_embd=128`, `n_head=4`, `block_size=128`, `batch_size=64`, `max_iters=5000` → **~820K parameters**, ~5 min on M4 MPS.

The key architectural addition vs `TinyGPT` in Part 1 is the positional embedding table `wpe`:
```python
self.wpe = nn.Embedding(block_size, n_embd)   # one vector per position
x = self.wte(idx) + self.wpe(pos)              # token + position
```

---

### 3. `nanogpt_runpod.md` — run documentation (new file)

Documents this specific training run end-to-end: RunPod setup, disk space management, commands used, benchmark results, and troubleshooting notes for every error encountered.

---

### What was NOT changed

- `nanochat/gpt.py` — model architecture untouched
- `nanochat/tokenizer.py` — tokenizer untouched
- `scripts/base_train.py`, `chat_sft.py`, `chat_eval.py` — training scripts untouched
- `tasks/` — SFT dataset definitions untouched
- `pyproject.toml` / `uv.lock` — dependencies untouched
