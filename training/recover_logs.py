"""Rebuilds training logs from Modal's retained stdout.

The JSONL logs written to the volume only survived in part: phase 1's died with the job at
step 1,730, phase 2's stopped syncing at step 3,760, and the volume they lived on is gone.
The console output of a stopped app is retained for far longer, though, and train.py prints
a line per logging interval, so a run's curves can be read back out of it.

Get the App IDs from the Modal dashboard (modal.com, Apps, including stopped ones); each
looks like ap-XXXXXXXXXXXXXXXXXXXXXX and appears in the URL of its page.

    python training/recover_logs.py ap-aaa ap-bbb ap-ccc ap-ddd
    python training/recover_logs.py --profile sweetberry042 ap-aaa

Writes one JSONL per app into docs/logs/recovered/ and prints what it found, so nothing
overwrites the existing files until you have looked at them.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "logs", "recovered")

# train.py line 400 prints:
#   step   1730 | loss 3.5902 | lr 5.92e-04 | grad_norm  0.199 |   183786 tok/s |  48.5m elapsed
LINE = re.compile(
    r"step\s+(?P<step>\d+)\s*\|\s*"
    r"loss\s+(?P<loss>[\d.]+)\s*\|\s*"
    r"lr\s+(?P<lr>[\d.eE+-]+)\s*\|\s*"
    r"grad_norm\s+(?P<grad>[\d.]+)\s*\|\s*"
    r"(?P<toks>[\d.]+)\s*tok/s\s*\|\s*"
    r"(?P<minutes>[\d.]+)m"
)
# and line 313 prints:
#   eval @ step 1500: train 3.7401  val 3.8312  (val perplexity 46.1)
VAL = re.compile(r"@\s*step\s+(?P<step>\d+)\s*:\s*"
                 r"train\s+(?P<train>[\d.]+)\s+val\s+(?P<val>[\d.]+)")


def fetch(app_id, profile=None):
    command = ["modal", "app", "logs", app_id]
    if profile:
        command += ["--profile", profile]
    try:
        # the Modal CLI draws box characters; on a Windows console the default codec
        # cannot encode them and the whole fetch fails, so decode as UTF-8 explicitly
        result = subprocess.run(command, capture_output=True, text=True, timeout=900,
                                encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        return None, "timed out after 10 minutes"
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()[:200]
    return result.stdout, None


def parse(text):
    train, val = [], []
    for line in text.splitlines():
        match = LINE.search(line)
        if match:
            train.append({"step": int(match["step"]), "loss": float(match["loss"]),
                          "lr": float(match["lr"]), "grad_norm": float(match["grad"]),
                          "tok_s": float(match["toks"]), "elapsed_m": float(match["minutes"])})
            continue
        match = VAL.search(line)
        if match:
            val.append({"step": int(match["step"]),
                        "val_loss": float(match["val"]),
                        "train_loss": float(match["train"])})
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("app_ids", nargs="+")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--raw", action="store_true", help="also keep the unparsed stdout")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for app_id in args.app_ids:
        print(f"\n{app_id}")
        text, error = fetch(app_id, args.profile)
        if error:
            print(f"   could not fetch: {error}")
            continue
        print(f"   {len(text.splitlines()):,} lines of stdout")

        if args.raw:
            with open(os.path.join(OUT, f"{app_id}.stdout.txt"), "w", encoding="utf-8") as handle:
                handle.write(text)

        train, val = parse(text)
        if not train and not val:
            print("   no training lines matched; keep --raw and check the format")
            continue
        if train:
            print(f"   train: {len(train):,} points, step {train[0]['step']} -> "
                  f"{train[-1]['step']}, {train[-1]['elapsed_m']:.0f} minutes")
            with open(os.path.join(OUT, f"{app_id}_train_log.jsonl"), "w", encoding="utf-8") as handle:
                for row in train:
                    handle.write(json.dumps(row) + "\n")
        if val:
            print(f"   val:   {len(val):,} points, step {val[0]['step']} -> {val[-1]['step']}")
            with open(os.path.join(OUT, f"{app_id}_val_log.jsonl"), "w", encoding="utf-8") as handle:
                for row in val:
                    handle.write(json.dumps(row) + "\n")

    print(f"\nwritten to {os.path.relpath(OUT, os.path.join(HERE, '..'))}")


if __name__ == "__main__":
    main()
