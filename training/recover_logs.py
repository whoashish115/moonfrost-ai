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
