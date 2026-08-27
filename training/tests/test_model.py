"""
Fast self-checks for model.py that need no trained checkpoint and no GPU:
builds a tiny randomly-initialized model and asserts the properties that
were silently broken before, so a regression shows up as a failed assert
rather than as garbled chat output weeks later.

Run:
    python test_model.py
"""

import os, sys
# these live in training/tests/, so the package directory is one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import torch

from config import MODEL_PRESETS
from model import GPT

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def build_tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(MODEL_PRESETS["tiny"]).eval()


def test_absorbed_path_matches_training_path():
    """The matrix-absorption inference path is an algebraic rewrite of the
    standard path, so both must produce the same logits. If they drift, the
    model silently generates different (worse) text at inference time than
    the loss it was trained against."""
    print("\nabsorbed vs. standard attention path")
    model = build_tiny_model()
    token_ids = torch.randint(0, model.config.vocabulary_size, (1, 32))

    model.disable_inference_absorption()
    with torch.no_grad():
        standard_logits, _ = model(token_ids)

    model.enable_inference_absorption()
    with torch.no_grad():
        absorbed_logits, _ = model(token_ids)

    largest_difference = (standard_logits - absorbed_logits).abs().max().item()
    check("absorbed path reproduces standard path", largest_difference < 2e-3, f"max abs diff {largest_difference:.2e}")


def test_cached_generation_matches_uncached():
    """Feeding tokens one at a time through the KV cache must give the same
    logits as running the whole sequence at once. This is the property that
    makes fast generation trustworthy."""
    print("\nKV cache correctness")
    model = build_tiny_model()
    model.enable_inference_absorption()
    token_ids = torch.randint(0, model.config.vocabulary_size, (1, 24))

    with torch.no_grad():
        full_pass_logits, _ = model(token_ids)
        caches = model.create_empty_key_value_caches()
        step_logits = None
        for position in range(token_ids.shape[1]):
            step_logits, _ = model(token_ids[:, position:position + 1], key_value_caches=caches, start_position=position)

    largest_difference = (full_pass_logits[:, -1] - step_logits[:, -1]).abs().max().item()
    check("token-by-token cache matches one-shot forward", largest_difference < 2e-3, f"max abs diff {largest_difference:.2e}")


def test_train_mode_releases_absorption():
    """Absorbed tensors are snapshots of the weights. If they survived into
    training, the model would attend through stale weights."""
    print("\nabsorption lifecycle")
    model = build_tiny_model()
    model.enable_inference_absorption()
    check("absorbed after enable", all(block.attention.is_absorbed for block in model.blocks))
    model.train()
    check("released by .train()", not any(block.attention.is_absorbed for block in model.blocks))
    model.eval()
    check("stays released until re-enabled", not any(block.attention.is_absorbed for block in model.blocks))


def test_greedy_decoding_is_deterministic():
    """temperature=0 used to divide by 1e-5, producing inf logits and then a
    NaN crash in multinomial. It must instead mean 'always take the argmax'."""
    print("\ntemperature 0 / greedy decoding")
    model = build_tiny_model()
    prompt = torch.randint(0, model.config.vocabulary_size, (1, 8))
    try:
        first = model.generate(prompt, max_new_tokens=12, temperature=0.0)
        second = model.generate(prompt, max_new_tokens=12, temperature=0.0)
        check("temperature=0 does not crash", True)
        check("temperature=0 is deterministic", torch.equal(first, second))
    except Exception as error:
        check("temperature=0 does not crash", False, f"{type(error).__name__}: {error}")
