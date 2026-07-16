"""
Pretraining loop: AdamW optimizer + cosine learning-rate decay with warmup +
gradient accumulation + bfloat16 mixed precision + gradient clipping +
periodic, atomic checkpointing.

Runs unmodified on a local GPU for a wall-clock-limited session, or on a
rented multi-GPU cloud box under torchrun for data-parallel training -- point
--data-dir/--tokenizer-dir/--ckpt-dir at persistent storage and it works in
either place. See ddp_utils.py and cloud_run.py.

Examples:
  # local, single GPU, stop after 110 minutes whatever step it has reached
  python train.py --size small --minutes 110

  # resume the SAME run (restores optimizer state and the step counter)
  python train.py --size small --minutes 180 --resume ../checkpoints/small_last.pt

  # start a NEW run from existing weights (fresh optimizer, step 0), with a learning-rate
  # schedule that finishes decaying exactly when the time budget runs out
  python train.py --size small --init-from ../checkpoints/small_upcycled_18.pt --tag v2 \
      --minutes 78 --lr-schedule time

  # cloud, 4 GPUs on one machine, data-parallel
  torchrun --standalone --nproc_per_node=4 train.py --size cloud --minutes 600

--resume vs --init-from
  --resume     continues an interrupted run: same optimizer moments, same step, same schedule.
  --init-from  begins a new run from existing weights. The architecture is read from the
               checkpoint, so it also accepts a grown model (see upcycle.py).

--lr-schedule
  steps  cosine decay over --max-steps (the preset's max_training_steps by default). If the
         time budget ends first, training stops with the learning rate still high, which
         leaves loss on the table.
  time   cosine decay over the WALL-CLOCK budget after warmup, so the learning rate reaches
         its minimum exactly at the deadline however fast the hardware turns out to be. This
         is the right choice whenever the budget is money or hours rather than tokens. If
         --max-steps is also given, whichever limit is nearer drives the decay.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid an MKL/libiomp5 conflict seen on some Anaconda+CUDA installs

import argparse
import dataclasses
import json
import math
import platform
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # otherwise output sits in a buffer when stdout isn't a terminal (e.g. redirected to a log file), so a running job looks stalled

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

import checkpoint_utils
from config import (MODEL_PRESETS, TRAIN_PRESETS, ModelConfig, count_active_parameters_per_token,
                    count_total_parameters)
from model import GPT
from ddp_utils import (cleanup_distributed_training, setup_distributed_training, synchronize_float,
                       synchronize_stop_decision)


def load_random_batch(binary_file_path, sequence_length, batch_size, device):
    """Loads a random batch of (input, target) sequences from a
    pre-tokenized .bin file. The file is memory-mapped rather than fully
    loaded into RAM, so this works even for files much bigger than
    available memory -- the same trick nanoGPT uses."""
    token_ids = np.memmap(binary_file_path, dtype=np.uint16, mode="r")
    start_indices = np.random.randint(0, len(token_ids) - sequence_length - 1, size=(batch_size,))
    input_sequences = torch.stack([torch.from_numpy(token_ids[i:i + sequence_length].astype(np.int64)) for i in start_indices])
    # the target for each position is simply the NEXT token -- this is what makes it "next-token prediction"
    target_sequences = torch.stack([torch.from_numpy(token_ids[i + 1:i + 1 + sequence_length].astype(np.int64)) for i in start_indices])
    if "cuda" in device:
        input_sequences = input_sequences.pin_memory().to(device, non_blocking=True)
        target_sequences = target_sequences.pin_memory().to(device, non_blocking=True)
    else:
        input_sequences = input_sequences.to(device)
        target_sequences = target_sequences.to(device)
    return input_sequences, target_sequences
