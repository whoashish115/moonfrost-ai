"""Replays this project's training logs into Weights & Biases, and writes the
same data as CSV for anything that reads CSV instead.

The runs happened on rented H100s with no W&B in the loop, so there is nothing
to "sync" -- the logs are JSON Lines written by train.py and sft_train.py, and
this replays them step by step so the charts look the way they would have if
W&B had been attached at the time.

Three runs are created, one per stage, sharing a group so they stack in the UI:

    moonfrost-pretrain-phase1   FineWeb-Edu shards 0-2
    moonfrost-pretrain-phase2   FineWeb-Edu shards 3-7, resumed schedule
    moonfrost-sft               chat fine-tune

Each carries `logged/coverage`, saying how much of its run the log actually covers, and
`actual/*` fields read from checkpoint metadata, saying where the stage really ended. Loss
is logged only where it was recorded; nothing between the surviving points is filled in.

Usage:

    pip install wandb
    wandb login
    python export_to_wandb.py --project moonfrost-777m
    python export_to_wandb.py --csv-only          # no account needed
"""
import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIRECTORY = os.path.join(HERE, "..", "docs", "logs")
EVAL_PATH = os.path.join(HERE, "..", "docs", "eval.json")

# (run name, subdirectory, train log, val log, stage notes)
#
# One run per stage. base1_*.jsonl is deliberately not among them: it is an attempt that
# died at step 1,730, not phase 1, and the real phase 1 survives only as a tail recovered
# from Modal. The failed attempts are kept in docs/logs for the record, not published.
STAGES = [
    ("pretrain-phase1", "recovered", "phase1_tail_train_log.jsonl", "phase1_tail_val_log.jsonl",
     {"stage": "pretraining", "phase": 1, "data": "FineWeb-Edu sample/10BT shards 0-2",
      "minutes_budget": 276, "coverage": "tail only, steps 9160-9542 recovered from Modal",
      "outcome": "completed, best val loss 3.1486 at step 9,500"}),
    ("pretrain-phase2", "", "base2_train_log.jsonl", "base2_val_log.jsonl",
     {"stage": "pretraining", "phase": 2, "data": "FineWeb-Edu sample/10BT shards 3-7",
      "minutes_budget": 340, "coverage": "first 105 minutes, the rest lost with the volume",
      "outcome": "completed, best val loss 2.976 at step 11,000"}),
    ("sft", "", "chat_train_log.jsonl", "chat_val_log.jsonl",
     {"stage": "supervised fine-tuning", "phase": 3,
      "data": "smol-smoltalk + everyday-conversations x25 + persona set at 3.3%",
      "minutes_budget": 80, "coverage": "complete",
      "outcome": "completed, best val loss 1.2484 at step 7,800"}),
]

# Where each stage really finished, read off the checkpoints rather than the logs. The
# logs are missing their middles; the checkpoints are intact, and they record the step they
# were saved at and the best validation loss seen up to that point. These are measured
# values, not interpolations: no point between them is invented anywhere in this file.
ACTUAL_OUTCOME = {
    "pretrain-phase1": {"final_step": 9542, "best_val_loss": 3.1486, "best_val_step": 9500,
                        "elapsed_minutes": 227.0, "tokens_per_second": 183076,
                        "source": "recovered stdout + base1_best.pt"},
    "pretrain-phase2": {"final_step": 11000, "best_val_loss": 2.9763,
                        "source": "moonfrost-777m-base.pt metadata"},
    "sft": {"final_step": 7800, "best_val_loss": 1.2484,
            "source": "moonfrost-777m-instruct-v2.pt metadata"},
}
