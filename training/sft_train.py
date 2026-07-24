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


def load_random_batch(all_input_ids, all_labels, batch_size, device, random_state):
    """Samples `batch_size` packed rows at random.

    The label shift: the data is written with labels aligned position-for-
    position with the input (labels[i] is the token AT position i, or -1
    where the loss should ignore it). The model's loss compares the
    prediction made AT position i against the token at position i+1, so the
    labels have to move one step left. Without this shift the model is
    trained to predict the token it was just shown -- which produces a
    beautifully low training loss and a completely useless model.
    """
    chosen_indices = np.sort(random_state.randint(0, all_input_ids.shape[0], size=batch_size))
    # sorted indices keep reads roughly sequential, which matters because these arrays are
    # memory-mapped: random access across a multi-gigabyte file is dominated by page faults
    input_ids = torch.from_numpy(np.asarray(all_input_ids[chosen_indices], dtype=np.int64))
    labels = torch.from_numpy(np.asarray(all_labels[chosen_indices], dtype=np.int64))

    labels = torch.roll(labels, shifts=-1, dims=1)
    labels[:, -1] = LABEL_IGNORE_INDEX  # nothing follows the last position, so it teaches nothing

    if "cuda" in device:
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
    else:
        input_ids, labels = input_ids.to(device), labels.to(device)
    return input_ids, labels


def cosine_between(progress, max_learning_rate, min_learning_rate):
    progress = min(max(progress, 0.0), 1.0)
    return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
