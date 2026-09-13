# Recovered log fragments

These were pulled out of the retained console output with `training/recover_logs.py`,
long after the volume that held the original JSONL files was gone.

Only the **tail** of a stopped job's output is kept, roughly the last fifty lines, so these
are fragments rather than whole runs. They are kept because they settle facts the surviving
logs in `../` do not:

| File | What it establishes |
|---|---|
| `phase1_tail_train_log.jsonl` | Pretraining phase 1 ran to step 9,542 over 227 minutes and completed. 39 points, steps 9,160-9,540, loss 3.115 to 3.084, throughput 183,076 tokens/second. |
| `phase1_tail_val_log.jsonl` | Phase 1's best validation loss was **3.1486** at step 9,500, perplexity 23.3. The final eval at step 9,542 was 3.1571. |
| `chat_aborted_train_log.jsonl` | The first chat fine-tune, killed from the CLI at step 960 after 8.9 minutes when the persona data was found to be 0.18% of tokens. |

The file `../base1_train_log.jsonl` is **not** this run. It ends at step 1,730 after 49
minutes and is the earlier phase 1 attempt that died when its client connection dropped.
The successful phase 1 above never had its log downloaded before the machine was released.

Phase 2's middle is not recoverable by this route either: `../base2_train_log.jsonl` covers
its first 105 minutes, and the job that produced the rest is no longer listed.
