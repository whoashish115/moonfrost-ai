"""
Instruction-tuning (SFT, "supervised fine-tuning"): continue training a
pretrained checkpoint from train.py on the (input_ids, labels) rows built by
sft_data.py, so the model learns to answer after
<|assistant|> instead of just continuing arbitrary text.

Same optimizer/precision/distributed machinery as train.py, but it starts
from existing weights, uses a much lower learning rate (large updates here
would wreck what pretraining learned), and each batch item is a packed row of
complete conversations rather than a random window of a token stream.

Things this version does that earlier ones did not:

  * Gradient accumulation (--grad-accum). The original loop did one
    micro-batch of 2 sequences per optimizer step -- an effective batch of 2,
    far too small and noisy for fine-tuning.
  * Memory-mapped data (see sft_data.py) instead of materializing gigabytes.
  * Atomic checkpoints, read to CPU rather than straight onto the GPU.
  * Barriers around rank-0 evaluation so multi-GPU runs cannot deadlock.
  * --lr-schedule time: the cosine decay tracks the wall-clock budget, so the
    learning rate finishes annealing exactly when paid time runs out.
  * A torch.compile probe that falls back to eager mode on a compiler failure,
    and evaluation on the uncompiled module so eval mode never recompiles.

Usage:
    python sft_train.py --init-from ../checkpoints/small_best.pt --minutes 60

    # budget-driven, e.g. on a rented GPU
    python sft_train.py --init-from v2_best.pt --minutes 16 --lr-schedule time --batch-size 24 --grad-accum 3
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import dataclasses
import json
import math
import platform
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # otherwise output sits in a buffer when stdout isn't a terminal

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

import checkpoint_utils
import sft_data
from config import ModelConfig, count_active_parameters_per_token, count_total_parameters
from ddp_utils import (cleanup_distributed_training, setup_distributed_training, synchronize_float,
                       synchronize_stop_decision)
from model import GPT

LABEL_IGNORE_INDEX = -1  # matches model.py's F.cross_entropy(..., ignore_index=-1)


def barrier(distributed_context):
    if distributed_context.is_distributed:
        torch.distributed.barrier()
