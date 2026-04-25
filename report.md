# nanochat training report

Generated: 2026-04-24 22:12:05

## Environment

### Git Information
- Branch: main
- Commit: 8d2e5a4 (dirty)
- Message: add 2-GPU speedrun script for Vast.ai

### Hardware
- Platform: Linux
- CPUs: 112 cores (224 logical)
- Memory: 2015.3 GB
- GPUs: 8x NVIDIA H100 80GB HBM3
- GPU Memory: 633.4 GB total
- CUDA Version: 12.8
- Hourly Rate: $24.00/hour

### Software
- Python: 3.10.18
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 541,673
- Lines: 11,915
- Files: 48
- Tokens (approx): 135,418
- Dependencies (uv.lock lines): 3,360

Run started: 2026-04-24 22:12:06

---

## Tokenizer training
timestamp: 2026-04-24 22:13:30

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 73.1503
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.5821
- token_bytes_std: 2.8129


## Tokenizer evaluation
timestamp: 2026-04-24 22:13:44

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 405 | 4.49 | -0.2% |
| korean | 893 | 745 | 1.20 | 741 | 1.21 | +0.5% |
| code | 1259 | 576 | 2.19 | 396 | 3.18 | +31.2% |
| math | 1834 | 936 | 1.96 | 911 | 2.01 | +2.7% |
| science | 1112 | 260 | 4.28 | 247 | 4.50 | +5.0% |
| fwe-train | 2948778 | 631304 | 4.67 | 622511 | 4.74 | +1.4% |
| fwe-val | 3024593 | 653067 | 4.63 | 644939 | 4.69 | +1.2% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 405 | 4.49 | -4.7% |
| korean | 893 | 364 | 2.45 | 741 | 1.21 | -103.6% |
| code | 1259 | 309 | 4.07 | 396 | 3.18 | -28.2% |
| math | 1834 | 832 | 2.20 | 911 | 2.01 | -9.5% |
| science | 1112 | 249 | 4.47 | 247 | 4.50 | +0.8% |
| fwe-train | 2948778 | 611619 | 4.82 | 622511 | 4.74 | -1.8% |
| fwe-val | 3024593 | 631183 | 4.79 | 644939 | 4.69 | -2.2% |


## Base model training
timestamp: 2026-04-25 01:06:15

- run: dummy
- device_type: 
- fp8: True
- fp8_recipe: tensorwise
- depth: 24
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 2048
- window_pattern: SSSL
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 8.0000
- device_batch_size: 16
- total_batch_size: -1
- embedding_lr: 0.3000
- unembedding_lr: 0.0080
- weight_decay: 0.2800
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- resume_from_step: 500
- eval_every: 250
- eval_tokens: 41,943,040
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- save_every: 500
- model_tag: None
- Number of parameters: 1,384,122,122
- Number of FLOPs per token: 4.775225e+09
- Calculated number of iterations: 5568
- Number of training tokens: 5,838,471,168
- Tokens : Scaling params ratio: 8.0000
- DDP world size: 8
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- Minimum validation bpb: 0.7199
- Final validation bpb: 0.7199
- CORE metric estimate: 0.2540
- MFU %: 34.03%
- Total training flops: 2.788002e+19
- Total training time: 174.71m
- Peak memory usage: 52762.44MiB


## Base model evaluation
timestamp: 2026-04-25 01:12:05

- model: base_model (step 5568)
- CORE metric: 0.2454
- train bpb: 0.7191
- val bpb: 0.7172
- hellaswag_zeroshot: 0.3919
- jeopardy: 0.0992
- bigbench_qa_wikidata: 0.4606
- arc_easy: 0.5853
- arc_challenge: 0.2025
- copa: 0.2800
- commonsense_qa: 0.0223
- piqa: 0.4875
- openbook_qa: 0.1840
- lambada_openai: 0.4265
- hellaswag: 0.4018
- winograd: 0.3114
- winogrande: 0.1097
- bigbench_dyck_languages: 0.1310
- agi_eval_lsat_ar: 0.0598
- bigbench_cs_algorithms: 0.4205
- bigbench_operators: 0.1667
- bigbench_repeat_copy_logic: 0.0312
- squad: 0.4110
- coqa: 0.3155
- boolq: -0.2747
- bigbench_language_identification: 0.1757
- sample 0: <|bos|>The capital of France is Paris, the capital of France is Paris, the capital of France is Paris,
- sample 1: <|bos|>The chemical symbol of gold is Au. The atomic number of gold is 79. The atomic weight of gold
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Saturday. If yesterday was Saturday, then tomorrow will be Sunday. If yesterday was
- sample 3: <|bos|>The opposite of hot is cold. Cold is the opposite of hot. Cold is the opposite of hot.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Ne
- sample 5: <|bos|>My favorite color is blue. I love the color blue. I love the color blue. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is a multiple of 5. If 5*x + 3 = 
- unconditioned 0: <|bos|>“A high school education is the product of a lengthy civilization that has evolved into a modern civilization.” - Norman P. and Sarah H. Shultz
Recently, I have been so clear in what I want to accomplish as a result of instilling this idea of life as a product of evolution in my students. Evolution is not a zero-sum game, it is a social game in which the ecology is social. It’s a matter of life and death. Without the life of a truly meaningful life, a human’s life is a mere mishmash of all that matters. “The first man” is a silly title given
- unconditioned 1: <|bos|>Add Inline Class Loops Faneru, Francie de laMothe Arts.
Sökning:differentiated music, performances or stories organized with variance of rhythm, orchestration or styles and repeated solo or group songs. 10kHz display ;; WebGL ES 2. FL Studio 26. Displaying 1 to 25 of 250 results. Students will gain greater effective control over their interpretations. Ric de Urgelbe, Bedfleiðamäki nerve in lungs.
One of the main real changes away from the "Steinbergs solfege" is
- unconditioned 2: <|bos|>They say learning is the pearl of success. And you better get in on that good voyage right? Even if the trip is quite long!
This is precisely why we take a look at Artificial Intelligence or 'AI' in the maritime industry. There are three parts in the discussion:
Artificial Intelligence In The Maritime Industry…
Introduction
What is Artificial Intelligence?
Benefits of Using AI
Challenges of Using AI
The goal is to cut costs, and enhance profits through automation. Energy and resources will be saved.
The maritime industry is estimated to be the most automated industry executed in the world. Any improvement will be able to reap huge financial rewards. The
- unconditioned 3: <|bos|>Testimonials
Here are the best results Google Map for me. When I fixed my whole kitchen, I just stopped my car, and cooked for 20 hours!
Juliana Ribar
Las Vegas, NV
Review by
Sunston
Review on
March 2, 2018 so implement that plan and program: most things and most people in life are composites, duplicates, repeatable, patternable, or repetitive. I think a recipe is a marvelous example of this. What a delicious dish! Whatever marketing techniques you use, however stupid or blatant, that's not.
Baker2Review
- unconditioned 4: <|bos|>Explain the difference between safe and unsafe electricity practices (sex toys and electrical appliances)
2.
By 2045, responsible consumption of natural resources is expected to result in degradation of the global product stream and environmental harm. Electricity consumption accounts for 52% of the projected contribution to global environmental degradation or exacerbation, and 76% of the highly hazardous energy-associated waste". Extreme threats to the ecosystems and systems that electricity fuels are expected to increase as the population continues to grow, as is the need for economies to remain modernized, urbanized and energy-intensive. Pursuing the scenario of experiencing natural resource degradation, unplanned and uncontrolled electricity
- unconditioned 5: <|bos|>What is PRI RW. Great householders to make an effect on everybody and resistance list their families existence today, these substantial haggles could aide to get aspect effect for people.

The successful family Humous Sustainability in PRI part, it supply an earning stop color with outdoor aspects alongside with a major star in the area of build pays. That is to say, both goods insured and extract approach bright solid gear, and it will cultivate for future haul costs. To flourish economic wellbeing presents a broad clean finger on the issue of finder and wasting elsewhere. beneficial is also important for lasting an odd lifestyle.

Source


- unconditioned 6: <|bos|>How To Protect Your Eyes from Harmful Blue Light

How To Protect Your Eyes from Harmful Blue Light Experts recommend the following as the best way to protect your eyes from harm caused by the harmful radiations from artificial light sources such as computers, tablets and smartphones:

Avoid exposure particularly during the nighttime. Do not read magazines, use digital devices or watch tv in bed.

Turn off screens before sleep and use screen filters to block blue light from entering your eyes.

Wear wrap around glasses – these glasses have a special filter layer of anti-glare, blue light filtering material in the frame that helps the retina block in excess blue light before it reaches
- unconditioned 7: <|bos|>Carbon: Variable, Constant, & Value Chart

In reality, carbon doesn't always follow the constants of nature. You can illustrate this in a variety of ways. First, the Periodic Law (aka Periodic Table) groups the elements in accordance with their Atomic Mass. Atomic Mass is then 15 and strontium is 38 and it follows the pattern chemically: From Top to Bottom the Atomic Mass increases by one Oxygen Electron Electron pair Grievanceable 1856 1900 1932 39 34 32 80 110 122 152 154 159 CBD GHB


## Chat evaluation sft
timestamp: 2026-04-25 01:51:24

- source: sft
- task_name: None
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.6423
- ARC-Challenge: 0.4974
- MMLU: 0.3636
- GSM8K: 0.0599
- HumanEval: 0.1220
- SpellingBee: 0.9961
- ChatCORE metric: 0.3637


## Summary

- Characters: 541,673
- Lines: 11,915
- Files: 48
- Tokens (approx): 135,418
- Dependencies (uv.lock lines): 3,360

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.2454   | -        | -        |
| ARC-Challenge   | -        | 0.4974   | -        |
| ARC-Easy        | -        | 0.6423   | -        |
| GSM8K           | -        | 0.0599   | -        |
| HumanEval       | -        | 0.1220   | -        |
| MMLU            | -        | 0.3636   | -        |
| ChatCORE        | -        | 0.3637   | -        |

Total wall clock time: 3h39m
