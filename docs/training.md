# Training

Three runs on one rented H100: two pretraining phases and a chat fine-tune. About
**12 GPU-hours** and **$55** in total.

| Stage | Wall clock | Steps | Tokens | Result |
|---|---|---|---|---|
| Pretrain phase 1 | 227 min | 9,542 | ~2.8B | best val loss **3.1486** at step 9,500 |
| Pretrain phase 2 | 340 min | ~12,200 | ~3.1B | best at **step 11,000**, val loss **2.976** |
| Chat fine-tune | 80 min | 9,201 | 2.7B (1.7 epochs) | best at **step 7,800**, val loss **1.2484** |

Each stage restarts its own step counter, so phase 2's step 11,000 is 11,000 steps into
phase 2 and not into the run as a whole.

**The step-by-step logs are incomplete, and the reason is worth recording.** Phase 1 was
attempted twice. The first try died at step 1,730 when its client connection dropped, and
`docs/logs/base1_train_log.jsonl` is that failed attempt rather than the real run. The
successful phase 1 ran to step 9,542 over 227 minutes at 183,076 tokens per second and
reached its best validation loss, 3.1486, at step 9,500; its log lived on a Modal volume
that was deleted with the account before anyone downloaded it. Phase 2's log was retrieved
only up to 105 minutes. The chat log is complete.

What could be recovered was. Modal keeps the tail of a stopped app's output, so
`training/recover_logs.py` pulls the last fifty or so lines of each surviving app, and that
is where phase 1's figures above come from. The fragments are in `docs/logs/recovered/`
with a note on what each one settles. The middle of both pretraining runs is gone.
`docs/charts/training_loss.png` plots what was logged and labels each panel with its
coverage rather than implying whole runs.

## Data

| Stage | Source | Amount |
|---|---|---|
| Pretraining | FineWeb-Edu `sample/10BT` | shards 0-2, then 3-7 |
| Chat | smol-smoltalk | ~460k conversations, 84% of rows |
| Chat | smoltalk `everyday-conversations`, x25 | ~55k conversations, 12% of rows |
| Chat | [persona set](https://huggingface.co/datasets/whoashish115/Moonfrost-Persona-SFT) | 140k conversations, 3.3% of rows |

The two pretraining phases read **disjoint shards**, so no text was seen twice.

`everyday-conversations` is repeated 25 times because small talk is a tiny fraction of
smol-smoltalk. Without the repeats the model answered "hi" with a lecture on quantum
physics.

Conversations are packed several to a 1,024-token row rather than padded, which raised
useful tokens per batch from roughly 40% to 61%.

## Hyperparameters

| Setting | Pretraining | Fine-tune |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Peak LR | 6e-4 | 2e-4 |
| Min LR | 6e-5 | 2e-5 |
| Schedule | time-based cosine | time-based cosine |
| Warmup | 300 / 150 steps | 100 steps |
| Micro-batch | 24 | 24 |
| Grad accumulation | 12 | 3 |
| Tokens per step | 294,912 | 73,728 |
| Precision | bf16 autocast, fp32 master | same |

## Splitting a run across two machines

The run spanned two machines. The schedule is **time-based
rather than step-based**: it takes a fraction of total training time and reads the
learning rate off one cosine curve. Phase 2 starts at `--schedule-start 0.5227`, exactly
where phase 1 stopped, so the learning rate anneals once across both machines instead of
restarting.

Only the weights cross the boundary. The optimizer state stays behind, which is why phase
2 re-warms for 150 steps.
