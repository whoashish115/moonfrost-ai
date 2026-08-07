"""
Whole-project health check. Runs in well under a minute, needs no GPU, and
NEVER starts a training run.

    python verify_all.py
    python verify_all.py --skip-slow      # skip the tiny synthetic train/SFT step checks

What it covers:
  1. every module imports cleanly
  2. every command-line script's --help parses (catches argparse mistakes
     that otherwise only surface when you finally run the thing on a rented GPU)
  3. the tokenizer has the chat special tokens the chat format depends on
  4. the prompt builder produces the exact format sft_prepare.py trains on,
     including its context-truncation behaviour
  5. the streaming text decoder reassembles multi-byte characters correctly
  6. checkpoints on disk are readable, and their architecture is reported
  7. the pretraining and SFT datasets are present, well-formed, and
     LABEL-ALIGNED -- the single most damaging silent bug in this pipeline
  8. one synthetic optimizer step for train.py's and sft_train.py's inner
     loops, on a randomly-initialized 'tiny' model with fabricated data, to
     prove the loop shapes and label shift are right (a few seconds, not a
     training run)

Anything that fails prints what to do about it.
"""

import os, sys
# these live in training/tests/, so the package directory is one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import importlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

# The Windows console defaults to cp1252, which cannot encode the emoji and accented text
# this script deliberately round-trips through the tokenizer. Without this, a FAILING check
# crashes with UnicodeEncodeError instead of printing what actually went wrong.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# the package directory, one level up from tests/, where the scripts actually live
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRECTORY = os.path.abspath(os.path.join(HERE, "..", "data"))
TOKENIZER_DIRECTORY = os.path.abspath(os.path.join(HERE, "..", "tokenizer"))
CHECKPOINT_DIRECTORY = os.path.abspath(os.path.join(HERE, "..", "checkpoints"))

PASSED, FAILED, WARNED = [], [], []


def check(name, condition, detail="", hint=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append((name, hint))
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))
        if hint:
            print(f"        -> {hint}")


def warn(name, detail="", hint=""):
    WARNED.append(name)
    print(f"  WARN  {name}" + (f"  ({detail})" if detail else ""))
    if hint:
        print(f"        -> {hint}")


def section(title):
    print(f"\n{title}")
