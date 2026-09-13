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


def test_seed_reproducibility():
    print("\nseeded sampling")
    model = build_tiny_model()
    prompt = torch.randint(0, model.config.vocabulary_size, (1, 8))
    first = list(model.generate_stream(prompt, max_new_tokens=10, temperature=0.9, seed=1234))
    second = list(model.generate_stream(prompt, max_new_tokens=10, temperature=0.9, seed=1234))
    third = list(model.generate_stream(prompt, max_new_tokens=10, temperature=0.9, seed=4321))
    check("same seed gives same tokens", first == second)
    check("different seed gives different tokens", first != third)


def test_stop_token_ends_generation():
    print("\nstop token handling")
    model = build_tiny_model()
    prompt = torch.randint(0, model.config.vocabulary_size, (1, 8))
    # force a stop by declaring every token a stop token
    produced = list(model.generate_stream(prompt, max_new_tokens=50, temperature=1.0,
                                           stop_token_ids=range(model.config.vocabulary_size)))
    check("stops on the first stop token", len(produced) == 1, f"produced {len(produced)} tokens")


def test_should_stop_callback():
    print("\ncooperative cancellation")
    model = build_tiny_model()
    prompt = torch.randint(0, model.config.vocabulary_size, (1, 8))
    emitted = []
    for token_id in model.generate_stream(prompt, max_new_tokens=50, temperature=1.0,
                                           should_stop=lambda: len(emitted) >= 5):
        emitted.append(token_id)
    check("should_stop halts generation", len(emitted) == 5, f"emitted {len(emitted)}")


def test_repetition_penalty_spares_protected_tokens():
    """The stop token must never be penalized, or long replies never end."""
    print("\nrepetition penalty scope")
    model = build_tiny_model()
    logits = torch.zeros(1, 1, model.config.vocabulary_size)
    logits[0, 0, :] = 1.0
    recent = torch.tensor([[5, 5, 7]])

    penalized = model._apply_repetition_penalties(
        logits[:, -1, :].clone(), recent, repetition_penalty=2.0,
        frequency_penalty=0.0, presence_penalty=0.0, protected_token_ids={7},
    )
    check("repeated token is penalized", penalized[0, 5].item() < 1.0, f"logit {penalized[0, 5].item():.3f}")
    check("protected token is untouched", abs(penalized[0, 7].item() - 1.0) < 1e-6, f"logit {penalized[0, 7].item():.3f}")
    check("unseen token is untouched", abs(penalized[0, 9].item() - 1.0) < 1e-6)


def test_context_overflow_slides_instead_of_resetting():
    """Generating past max_sequence_length must keep working and must keep
    recent context, not silently restart from a single token."""
    print("\ncontext window overflow")
    model = build_tiny_model()
    context_limit = model.config.max_sequence_length
    prompt = torch.randint(0, model.config.vocabulary_size, (1, context_limit - 4))
    produced = list(model.generate_stream(prompt, max_new_tokens=20, temperature=1.0))
    check("keeps generating past the context limit", len(produced) == 20, f"produced {len(produced)}")


def test_long_prompt_is_truncated_not_crashed():
    print("\noversized prompt")
    model = build_tiny_model()
    prompt = torch.randint(0, model.config.vocabulary_size, (1, model.config.max_sequence_length + 50))
    produced = list(model.generate_stream(prompt, max_new_tokens=5, temperature=1.0))
    check("oversized prompt is truncated, not fatal", len(produced) == 5)


def main():
    print("model.py self-checks (tiny randomly-initialized model, CPU)")
    for test in [
        test_absorbed_path_matches_training_path,
        test_cached_generation_matches_uncached,
        test_train_mode_releases_absorption,
        test_greedy_decoding_is_deterministic,
        test_seed_reproducibility,
        test_stop_token_ends_generation,
        test_should_stop_callback,
        test_repetition_penalty_spares_protected_tokens,
        test_context_overflow_slides_instead_of_resetting,
        test_long_prompt_is_truncated_not_crashed,
    ]:
        test()

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  failed: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
