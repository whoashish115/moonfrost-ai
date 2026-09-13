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

## Budget guards

`cloud_run.py` refuses to import if any account's worst case exceeds its cap:

```
account 3 (cap $7.80):
  chat data prep       $  0.42
  chat tuning          $  6.44
  export               $  0.25
  orchestrator         $  0.12
  TOTAL                $  7.23
```

Worst case means every stage hitting its full Modal timeout. `verify_cloud_run.py` runs
17 checks before anything starts: that every CLI flag exists in the script it is passed
to, that the preset matches the data, that the schedule is continuous, that timeouts
leave room, that the phases read different shards, and that the bill cannot exceed the
cap.

## What went wrong

**A detached job died with the terminal.** `modal run --detach` was not enough: a
blocking `.remote()` keeps a client connection open, and the job died when that client
did, 1,730 steps into phase 1. `.spawn()` returns immediately and the work survives.
Cost: about $0.50.

**The persona data was 0.18% of the tokens.** Caught 8 minutes into a paid run. 20,000
short exchanges packed into 650 rows of 368,175 - the model would have seen each identity
fact about twice. Rebuilt as 140,000 multi-turn sessions, 12,739 rows, **3.3%**. Cost of
stopping: about $0.75.

**A checkpoint downloaded corrupt.** One tensor,
`blocks.11.feed_forward.expert_up_weights`, arrived with 67,407 NaNs. Every generation
came back empty. The file on the volume was fine; re-downloading fixed it.
`modal volume get` does not verify what it writes.

**The Hugging Face export was silently broken.** RoPE tables are `persistent=False`
buffers, so they are not in the checkpoint, and `transformers` fills unlisted tensors with
**zeros** rather than running `__init__`. Zero cosine and sine erase position entirely:
the exported model answered `"made made made made"` while the same weights answered
correctly through the native path. Fixed by shipping the tables in the safetensors file
and marking them persistent, so a regression shows up as missing keys instead of
nonsense.

## Throughput

**179,000 tokens/second** on one H100, against 6,056 on the first attempt. The difference
was the MoE implementation: a Python loop over 32 experts with a CUDA sync per expert,
replaced by stacked tensors and capacity dispatch.

## Reproducing

```bash
python training/verify_cloud_run.py                      # 17 pre-flight checks
modal run training/cloud_run.py --stage account1 --spawn # phase 1
modal run training/cloud_run.py --stage account2 --spawn # phase 2
modal run training/cloud_run.py --stage account3 --spawn # chat tune + export
```

Seeds are fixed: 1337 for the data shuffle, 5 and 11 for sampling probes. Training logs
are in `docs/logs/` as JSON Lines, `training/export_to_wandb.py` replays them into
[Weights & Biases](https://wandb.ai/whoashish115-base/moonfrost-777m), and the replayed runs are public
there.
