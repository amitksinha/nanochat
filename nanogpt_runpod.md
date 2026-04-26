# My nanochat Projects

This document covers two related but independent projects built on [Karpathy's nanochat](https://github.com/karpathy/nanochat):

1. **nanochat Full Pipeline** — train a 1.38B parameter LLM end-to-end on FineWeb data using 8×H100 GPUs on RunPod, then deploy it as a chat assistant locally on Mac.
2. **Transformer Learning Notebook** — a self-contained educational notebook that builds the GPT architecture from scratch and trains a small model on Tiny Shakespeare, designed to run in ~5 minutes on a Mac.

---

# Project 1: nanochat Full Pipeline

Train and deploy Karpathy's nanochat as designed — pretraining a 1.38B parameter model on the FineWeb/ClimbMix dataset, followed by SFT fine-tuning to produce a chat assistant.

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

> **Note:** The `speedrun.sh` script has its own `export NANOCHAT_BASE_DIR=...` line. Comment it out so your env var takes effect:
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

Key metrics printed each `eval_every` steps:
- `val_bpb` — validation bits per byte (lower is better; GPT-2 ~1.0, our model reaches ~0.717)
- `train/mfu` — Model FLOP Utilization (achieved ~34% in base, ~50% in SFT)
- `eta` — estimated time remaining

### 6. Disk space management

Each FP8 checkpoint is ~9.5 GB:
- Model weights: ~4 GB
- Optimizer state (8 shards × 685 MB): ~5.5 GB

After training completes, delete optimizer states to free ~5.5 GB:

```bash
rm /workspace/nanochat_cache/chatsft_checkpoints/d24/optim_*.pt
```

### 7. Stop the pod

Once training is done, **stop** (not terminate) the pod from the RunPod console. The volume persists. Restart a cheap single-GPU pod later for inference.

---

## Exposing the Web UI on RunPod — What Actually Works

The default instructions use port 8000, but several things get in the way on RunPod.

### What fails and why

| Approach | What happens |
|----------|-------------|
| `chat_web` default port 8000 | Not exposed — RunPod proxy returns 404 |
| `https://<pod-id>-8000.proxy.runpod.net` | 404 — port was never registered with the proxy |
| SSH tunnel `ssh -L 8000:localhost:8000 ... ssh.runpod.io` | RunPod's SSH proxy does not support port forwarding |
| Port 8888 | Already taken by Jupyter Lab |

### Required fix — expose port 7860 before starting the pod

1. Go to RunPod → Pods → **⋮** → **Edit Pod**
2. Find the **Expose HTTP ports** field — it shows `8888` by default
3. Change it to `8888,7860`
4. Click **Save**, then start the pod

### Starting the server

```bash
cd /workspace/nanochat
source .venv/bin/activate
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
python -m scripts.chat_web --source sft --port 7860
```

Wait for `Server ready at http://localhost:7860`, then open in your browser:

```
https://<pod-id>-7860.proxy.runpod.net
```

If the server is already running from a previous session:
```bash
pkill -f chat_web
python -m scripts.chat_web --source sft --port 7860
```

---

## Copying the Model to Your Local Mac

RunPod's SSH proxy does not support `scp` or `rsync`. The easiest method is Python's built-in HTTP server.

### Download model files

**1.** Stop the chat server (Ctrl+C), then serve the checkpoint directory:

```bash
cd /workspace/nanochat_cache/chatsft_checkpoints/d24/
python3 -m http.server 7860
```

**2.** Open `https://<pod-id>-7860.proxy.runpod.net` in your browser — you'll see a directory listing.

**3.** Download these two files:
- `model_000483.pt` (~4 GB) — model weights
- `meta_000483.json` (~1 KB) — model config

Skip the `optim_*.pt` files — those are only needed to resume training.

**4.** Download the tokenizer the same way:

```bash
cd /workspace/nanochat_cache
python3 -m http.server 7860
```

Navigate into `tokenizer/` in the browser and download `tokenizer.pkl`.

---

## Running the Model Locally on Mac

### Step 1 — Set up the directory structure

nanochat expects a specific layout under `NANOCHAT_BASE_DIR`. Create it and place the downloaded files:

```bash
mkdir -p ~/Documents/nanochat/trained_model/chatsft_checkpoints/d24
mkdir -p ~/Documents/nanochat/trained_model/tokenizer

mv ~/Downloads/model_000483.pt ~/Documents/nanochat/trained_model/chatsft_checkpoints/d24/
mv ~/Downloads/meta_000483.json ~/Documents/nanochat/trained_model/chatsft_checkpoints/d24/
mv ~/Downloads/tokenizer.pkl ~/Documents/nanochat/trained_model/tokenizer/
```

The final structure should be:

```
trained_model/
├── chatsft_checkpoints/
│   └── d24/
│       ├── model_000483.pt
│       └── meta_000483.json
└── tokenizer/
    └── tokenizer.pkl
```

### Step 2 — Install dependencies (first time only)

```bash
cd ~/Documents/nanochat
uv sync --extra cpu
source .venv/bin/activate
```

### Step 3 — Start the web UI

```bash
cd ~/Documents/nanochat
source .venv/bin/activate
export NANOCHAT_BASE_DIR=~/Documents/nanochat/trained_model
python -m scripts.chat_web --source sft --port 7860
```

Wait for `Server ready at http://localhost:7860`, then open **http://localhost:7860** in your browser.

> **Note:** On Mac MPS (Apple Silicon) the model loads automatically on the GPU (`Autodetected device type: mps`). Expect a few seconds per response — slower than H100 but quality is identical.

---

## Troubleshooting

**"No checkpoints found" / training starts from scratch**
→ `NANOCHAT_BASE_DIR` was not set, or `speedrun.sh` overrode it. Comment out the export line in the script and set the env var manually before running.

**Disk full during training**
→ Each FP8 checkpoint is ~9.5 GB. With `--save-every=500` over 5568 steps that's ~105 GB. Use a 150 GB volume and delete optimizer shards after training.

**`hf_transfer` not installed (SFT fails)**
→ Run: `uv pip install hf_transfer`

**SSH key not working after pod restart**
→ Add your SSH public key to RunPod Settings → SSH Keys so it propagates to new containers automatically.

**`git push` auth failure**
→ GitHub no longer accepts passwords. Use a Personal Access Token (`repo` scope) as the password.

**`address already in use` on port 7860**
→ Run `pkill -f chat_web` then restart the server.

**`FileNotFoundError: tokenizer.pkl`**
→ The tokenizer is not included with the model checkpoint. Download it separately from `/workspace/nanochat_cache/tokenizer/tokenizer.pkl` and place it at `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`.

---

# Project 2: Transformer Learning Notebook

A self-contained educational notebook (`transformer_learning.ipynb`) that builds the GPT transformer architecture from scratch in 20 steps, then trains a real character-level model on Tiny Shakespeare. Runs entirely on a Mac — no GPU cloud needed.

## Quick Start

```bash
cd ~/Documents/nanochat
uv sync --extra cpu --group dev
source .venv/bin/activate

jupyter notebook transformer_learning.ipynb
```

Run all cells (Kernel → Restart & Run All). Training in Step 17 takes **~5 minutes on an M4 Mac**.

---

## What the Notebook Covers

### Part 1 — Architecture Walkthrough (Steps 1–13)

Builds the transformer piece by piece using plain PyTorch, connecting each step to the corresponding code in `nanochat/gpt.py`:

| Step | Topic |
|------|-------|
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

### Part 2 — Train on Tiny Shakespeare (Steps 14–20)

Trains a real character-level GPT from scratch:

| Step | Topic |
|------|-------|
| 14 | Download Tiny Shakespeare + character-level tokenizer (65 tokens) |
| 15 | `get_batch()` — random minibatch sampling |
| 16 | `GPTShakespeare` — full model with positional embeddings (`wte + wpe`) |
| 17 | Training loop — AdamW, cross-entropy loss, validation every 250 steps |
| 18 | Loss curves (matplotlib) |
| 19 | Text generation samples |
| 20 | Discussion — 6 factors that affect model performance |

**Model hyperparameters:**

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

## How to Test the Shakespeare Model

After Step 17 completes, use the `sample()` function from Step 19 to generate text.

### What to expect

At ~820K parameters with character-level tokenization and a 128-character context window, the model learns Shakespeare's **style and format** but not narrative coherence. It will produce correct spelling, Shakespearean vocabulary, and speaker label formatting — but not coherent meaning across more than a sentence or two.

### Best prompts

```python
# Character name prompts — appear constantly in training data
sample("ROMEO:")
sample("KING:")
sample("HAMLET:")

# Stage direction style
sample("Enter ")
sample("ACT II. SCENE ")

# Common speech openers
sample("My lord, I ")
sample("I pray thee, ")

# Famous fragments
sample("To be or not")
sample("All the world's a ")
```

### Prompts to avoid

```python
# Modern English — never seen in training data
sample("Hey, what's up")
sample("The weather today")
```

### Adjusting generation quality

```python
sample("ROMEO:", temperature=1.0, top_k=50)   # more varied
sample("ROMEO:", temperature=0.5, top_k=20)   # more conservative
sample("ROMEO:", temperature=0.2, top_k=5)    # near-greedy
```

### Reading the loss curves

| Val loss at step 5000 | Meaning |
|---|---|
| > 2.0 | Model barely learned — check device, batch size, learning rate |
| 1.6 – 2.0 | Reasonable; text will look vaguely Shakespearean |
| 1.4 – 1.6 | Good convergence; clear Shakespeare style |
| < 1.4 | Excellent — try generating longer sequences |

---

# Reference

## Key Architecture Concepts

### Why positional embeddings matter

Without `wpe`, attention is permutation-invariant — the model cannot tell whether "king" appears at position 0 or position 50. Learned positional embeddings give the model this information:

```python
pos_emb = self.wpe(torch.arange(T))   # [T, n_embd]
x = self.wte(idx) + pos_emb           # [B, T, n_embd]
```

### nanochat vs TinyGPT (notebook)

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

## Training Configuration (this run)

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
│   └── ...
├── scripts/
│   ├── base_train.py           # Pretraining script
│   ├── chat_sft.py             # SFT fine-tuning
│   ├── chat_eval.py            # Evaluation on tasks
│   ├── chat_cli.py             # Terminal chat interface
│   └── chat_web.py             # Web UI
├── tasks/
│   ├── gsm8k.py                # Grade school math
│   ├── mmlu.py                 # Multiple choice, broad topics
│   ├── spellingbee.py          # Letter-counting with Python verification
│   └── smoltalk.py             # SmolTalk conversation dataset
├── runs/
│   └── speedrun.sh             # Full pipeline (tokenize → pretrain → SFT → eval)
├── transformer_learning.ipynb  # Project 2: educational notebook
├── report.md                   # Auto-generated training report
└── nanogpt_runpod.md           # This file
```

---

## Changes from Original nanochat

### 1. `runs/speedrun.sh` — checkpoint control

**Commented out `NANOCHAT_BASE_DIR`** so an external env var can override it, keeping checkpoints on the persistent volume:

```diff
-export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
+#export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
```

**Added `--save-every` and `--resume-from-step`** to checkpoint every 500 steps and allow resuming after interruption:

```diff
-torchrun ... -m scripts.base_train -- --depth=24 ... --run=$WANDB_RUN
+torchrun ... -m scripts.base_train -- --depth=24 ... --run=$WANDB_RUN --save-every=500 --resume-from-step=500
```

### 2. `transformer_learning.ipynb` — new file

Added a 65-cell Jupyter notebook (Project 2) covering the GPT architecture from scratch and training on Tiny Shakespeare.

### 3. `nanogpt_runpod.md` — new file

Documents this run end-to-end: RunPod setup, disk management, deployment gotchas, local Mac inference, and all errors encountered.

### What was NOT changed

Model architecture, tokenizer, training scripts, SFT tasks, and dependencies are all untouched from the original nanochat repo.
