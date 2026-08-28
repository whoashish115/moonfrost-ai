"""
Load a checkpoint and generate text from a single prompt, from the command
line -- useful for quick one-off checks without starting the web interface
(server.py) for a full conversation.

Two modes:
  - completion (default): raw text continuation, for base/pretrained checkpoints
  - --chat: wraps the prompt in the <|user|>/<|assistant|> template and
    stops at <|endoftext|>, for checkpoints produced by sft_train.py

Usage:
    python sample.py --ckpt ../checkpoints/small_best.pt \
        --prompt "The history of the internet" --max-new-tokens 200

    python sample.py --ckpt ../checkpoints/sft_last.pt --chat \
        --prompt "What's a good way to learn a new language?"

    # side-by-side comparison of several settings on the same prompt
    python sample.py --ckpt ../checkpoints/sft_last.pt --chat --prompt "Hi" --sweep
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid an MKL/libiomp5 conflict seen on some Anaconda+CUDA installs

import argparse
import time

import torch
from tokenizers import Tokenizer

import checkpoint_utils
from chat_format import DEFAULT_SAMPLING, ChatSpecialTokens, IncrementalTextDecoder, build_prompt_token_ids
from config import ModelConfig, count_total_parameters
from model import GPT

# settings compared by --sweep, chosen to show the two failure modes of a small model:
# too-low temperature loops, too-high temperature turns to noise
SWEEP_SETTINGS = [
    ("greedy",       dict(temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0)),
    ("conservative", dict(temperature=0.5, top_k=40, top_p=0.9, min_p=0.05, repetition_penalty=1.1)),
    ("balanced",     dict(temperature=0.8, top_k=40, top_p=0.92, min_p=0.05, repetition_penalty=1.1)),
    ("creative",     dict(temperature=1.1, top_k=80, top_p=0.95, min_p=0.02, repetition_penalty=1.05)),
    ("no penalty",   dict(temperature=0.8, top_k=40, top_p=0.92, min_p=0.05, repetition_penalty=1.0)),
]
