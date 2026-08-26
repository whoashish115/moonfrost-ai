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

ARCHITECTURE = {
    "total_parameters": 777_000_000, "active_parameters_per_token": 161_000_000,
    "layers": 14, "dense_layers": 1, "moe_layers": 13,
    "hidden_size": 896, "attention_heads": 14, "attention": "multi-head latent attention",
    "kv_compressed_latent_dim": 320, "query_compressed_latent_dim": 512,
    "routed_experts": 32, "experts_per_token": 3, "shared_experts": 1,
    "context_length": 1024, "vocabulary_size": 32768, "tokenizer": "byte-level BPE, trained from scratch",
    "precision": "bfloat16 autocast, float32 master weights",
    "optimizer": "AdamW", "schedule": "time-based cosine with warmup",
    "micro_batch": 24, "gradient_accumulation": 12, "tokens_per_step": 294_912,
    "gpu": "1x H100 80GB (rented)",
}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue      # a truncated final line from an interrupted run
    return rows


def write_csv(name, train_rows, val_rows, directory):
    """One CSV per stage, with the validation points merged in on their step."""
    validation_by_step = {row["step"]: row for row in val_rows}
    path = os.path.join(directory, f"{name}.csv")
    fields = ["step", "train_loss", "lr", "grad_norm", "tokens_per_second",
              "elapsed_minutes", "tokens_processed", "val_loss", "val_train_loss"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in train_rows:
            validation = validation_by_step.get(row["step"], {})
            writer.writerow({
                "step": row.get("step"),
                "train_loss": row.get("loss"),
                "lr": row.get("lr"),
                "grad_norm": row.get("grad_norm"),
                "tokens_per_second": row.get("tok_s"),
                "elapsed_minutes": row.get("elapsed_m"),
                "tokens_processed": (row.get("step") or 0) * ARCHITECTURE["tokens_per_step"],
                "val_loss": validation.get("val_loss"),
                "val_train_loss": validation.get("train_loss"),
            })
    return path


def summarise(train_rows, val_rows):
    if not train_rows:
        return {}
    throughputs = [row["tok_s"] for row in train_rows if row.get("tok_s")]
    losses = [row["val_loss"] for row in val_rows if row.get("val_loss") is not None]
    return {
        "steps": train_rows[-1].get("step"),
        "minutes": round(train_rows[-1].get("elapsed_m", 0), 1),
        "tokens_processed": (train_rows[-1].get("step") or 0) * ARCHITECTURE["tokens_per_step"],
        "median_tokens_per_second": round(sorted(throughputs)[len(throughputs) // 2]) if throughputs else None,
        "final_train_loss": round(train_rows[-1].get("loss", 0), 4),
        "best_val_loss": round(min(losses), 4) if losses else None,
        "final_val_loss": round(losses[-1], 4) if losses else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="moonfrost-777m")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--csv-only", action="store_true", help="write CSVs and skip W&B entirely")
    parser.add_argument("--out", default=os.path.join(HERE, "..", "docs", "wandb"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    evaluation = json.load(open(EVAL_PATH, encoding="utf-8")) if os.path.exists(EVAL_PATH) else {}

    written = []
    for name, subdirectory, train_file, val_file, notes in STAGES:
        directory = os.path.join(LOG_DIRECTORY, subdirectory) if subdirectory else LOG_DIRECTORY
        train_rows = read_jsonl(os.path.join(directory, train_file))
        # a run killed before its first evaluation has no validation log at all
        val_rows = read_jsonl(os.path.join(directory, val_file)) if val_file else []
        if not train_rows:
            print(f"skipping {name}: no log at {train_file}")
            continue

        written.append(write_csv(name, train_rows, val_rows, args.out))
        print(f"{name}: {len(train_rows)} train points, {len(val_rows)} validation points"
              f"  [{notes.get('coverage')}]")

        if args.csv_only:
            continue

        import wandb
        # a fixed id keyed on the stage name, so running this again corrects the existing
        # run rather than stacking another copy beside it
        run = wandb.init(project=args.project, entity=args.entity, name=f"moonfrost-{name}",
                         id=f"moonfrost-{name}", resume="allow",
                         group="moonfrost-777m", job_type=notes["stage"], reinit=True,
                         notes=notes.get("outcome", ""),
                         config={**ARCHITECTURE, **notes})
        validation_by_step = {row["step"]: row for row in val_rows}
        for row in train_rows:
            step = row.get("step", 0)
            metrics = {
                "train/loss": row.get("loss"),
                "train/learning_rate": row.get("lr"),
                "train/grad_norm": row.get("grad_norm"),
                "throughput/tokens_per_second": row.get("tok_s"),
                "progress/elapsed_minutes": row.get("elapsed_m"),
                "progress/tokens_processed": step * ARCHITECTURE["tokens_per_step"],
            }
            validation = validation_by_step.get(step)
            if validation:
                metrics["val/loss"] = validation.get("val_loss")
                metrics["val/train_loss"] = validation.get("train_loss")
            run.log({k: v for k, v in metrics.items() if v is not None}, step=step)

        run.summary.update(summarise(train_rows, val_rows))

        # what the logs cover, and what the stage actually did, kept as separate fields so
        # neither is mistaken for the other
        run.summary["logged/last_step"] = train_rows[-1].get("step") if train_rows else None
        run.summary["logged/points"] = len(train_rows)
        run.summary["logged/coverage"] = notes.get("coverage")
        for key, value in (ACTUAL_OUTCOME.get(name) or {}).items():
            run.summary[f"actual/{key}"] = value
        # the evaluation belongs on the fine-tuned run, since that is the model it measured
        if name == "sft" and evaluation:
            if evaluation.get("pretraining_eval"):
                run.summary["eval/held_out_perplexity"] = evaluation["pretraining_eval"]["perplexity"]
                run.summary["eval/held_out_loss"] = evaluation["pretraining_eval"]["loss"]
            if evaluation.get("inference"):
                run.summary["eval/tokens_per_second"] = evaluation["inference"]["tokens_per_second"]
            table = wandb.Table(columns=["benchmark", "shots", "accuracy",
                                         "accuracy_length_normalised", "chance", "examples"])
            for key, scores in (evaluation.get("benchmarks") or {}).items():
                if "error" in scores:
                    continue
                benchmark, shots = key.split("|")
                run.summary[f"bench/{benchmark}_{shots}"] = scores["accuracy"]
                table.add_data(benchmark, scores["shots"], scores["accuracy"],
                               scores["accuracy_length_normalised"], scores["chance"],
                               scores["examples"])
            run.log({"benchmarks": table})
        run.finish()

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "architecture": ARCHITECTURE,
            "stages": {
                name: {
                    **summarise(
                        read_jsonl(os.path.join(LOG_DIRECTORY, sub or "", train_file)),
                        read_jsonl(os.path.join(LOG_DIRECTORY, sub or "", val_file)) if val_file else [],
                    ),
                    "coverage": notes.get("coverage"),
                    "outcome": notes.get("outcome"),
                }
                for name, sub, train_file, val_file, notes in STAGES
            },
            "evaluation": evaluation,
        }, handle, indent=2)
    written.append(summary_path)

    print("\nwrote:")
    for path in written:
        print("  " + os.path.relpath(path, HERE))
    if args.csv_only:
        print("\nUpload the CSVs to W&B with: wandb artifact put <dir> --type dataset\n"
              "or rerun without --csv-only to create real runs with charts.")


if __name__ == "__main__":
    main()
