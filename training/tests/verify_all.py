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


# ---------------------------------------------------------------- 1. imports

MODULES = ["config", "model", "chat_format", "checkpoint_utils", "sft_data",
           "ddp_utils", "train", "sft_train", "sample", "tokenizer_train",
           "download_pretrain_data", "server", "export_to_huggingface"]

OPTIONAL_MODULES = {
    "server": "fastapi / uvicorn",
    "cloud_run": "modal",
    "download_pretrain_data": "datasets",
    "export_to_huggingface": "transformers",
}


def test_imports():
    section("1. module imports")
    for module_name in MODULES + ["cloud_run"]:
        try:
            importlib.import_module(module_name)
            check(f"import {module_name}", True)
        except ImportError as error:
            dependency = OPTIONAL_MODULES.get(module_name)
            if dependency and dependency.split(" / ")[0] in str(error):
                warn(f"import {module_name}", f"optional dependency missing: {dependency}",
                     f"pip install {dependency.split(' / ')[0]}")
            else:
                check(f"import {module_name}", False, str(error))
        except Exception as error:
            check(f"import {module_name}", False, f"{type(error).__name__}: {error}")


# ------------------------------------------------------------ 2. script --help

HELP_SCRIPTS = ["train.py", "sft_train.py", "sample.py", "tokenizer_train.py",
                "download_pretrain_data.py", "checkpoint_utils.py", "sft_data.py",
                "server.py", "export_to_huggingface.py"]


def test_help():
    section("2. command-line interfaces (--help)")
    for script in HELP_SCRIPTS:
        completed = subprocess.run([sys.executable, script, "--help"], cwd=HERE,
                                    capture_output=True, text=True, timeout=180)
        ok = completed.returncode == 0
        detail = ""
        if not ok:
            tail = (completed.stderr or completed.stdout).strip().splitlines()
            detail = tail[-1] if tail else f"exit {completed.returncode}"
        check(f"{script} --help", ok, detail)


# ------------------------------------------------------------ 3-5. chat format

def test_chat_format():
    section("3. tokenizer and chat format")
    tokenizer_path = os.path.join(TOKENIZER_DIRECTORY, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        check("tokenizer present", False, tokenizer_path,
              "run tokenizer_train.py")
        return None, None

    from tokenizers import Tokenizer

    from chat_format import ChatSpecialTokens, IncrementalTextDecoder, build_prompt_token_ids

    tokenizer = Tokenizer.from_file(tokenizer_path)
    check("tokenizer present", True, f"vocabulary size {tokenizer.get_vocab_size()}")

    try:
        special = ChatSpecialTokens(tokenizer)
        check("chat special tokens registered", True,
              f"user={special.user} assistant={special.assistant} endoftext={special.end_of_text}")
    except ValueError as error:
        check("chat special tokens registered", False, str(error), "retrain the tokenizer with tokenizer_train.py")
        return tokenizer, None

    # the prompt must END with <|assistant|>, or the model has not been cued to reply
    prompt = build_prompt_token_ids(tokenizer, [], "hello", special)
    check("prompt starts with <|user|>", prompt[0] == special.user)
    check("prompt ends with <|assistant|>", prompt[-1] == special.assistant)

    # multi-turn: each earlier assistant turn must be closed with <|endoftext|>
    history = [("user", "one"), ("assistant", "two"), ("user", "three"), ("assistant", "four")]
    multi = build_prompt_token_ids(tokenizer, history, "five", special)
    check("multi-turn history keeps every turn", multi.count(special.user) == 3,
          f"{multi.count(special.user)} user turns")
    check("assistant turns are terminated", multi.count(special.end_of_text) == 2,
          f"{multi.count(special.end_of_text)} endoftext tokens")

    # truncation drops the OLDEST turns and still ends correctly
    long_history = [("user", "q " * 200), ("assistant", "a " * 200)] * 6
    truncated = build_prompt_token_ids(tokenizer, long_history, "final question", special, max_prompt_tokens=256)
    check("oversized history is truncated to budget", len(truncated) <= 256, f"{len(truncated)} tokens")
    check("truncated prompt still ends with <|assistant|>", truncated[-1] == special.assistant)

    # the stop set must include endoftext, and specials must be protected from the penalty
    check("<|endoftext|> is a stop token", special.end_of_text in special.stop_ids)
    check("special tokens are penalty-protected", special.end_of_text in special.all_special_ids)

    section("4. streaming text decoder")
    # a decoder fed the tokens of a string must reproduce that string exactly, including
    # characters whose UTF-8 encoding spans several tokens
    for sample in ["Hello, world!", "naive cafe resume", "emoji: \U0001F600\U0001F680",
                   "accents: éèê", "math: ∑ x²", "code: `a < b`"]:
        token_ids = tokenizer.encode(sample).ids
        decoder = IncrementalTextDecoder(tokenizer, skip_special_tokens=True)
        streamed = decoder.push_all(token_ids)
        one_shot = tokenizer.decode(token_ids, skip_special_tokens=True)
        check(f"streamed == one-shot decode: {sample[:24]!r}", streamed == one_shot,
              f"streamed {streamed!r} vs {one_shot!r}")

    return tokenizer, special


# ------------------------------------------------------------ 6. checkpoints

def test_checkpoints():
    section("5. checkpoints")
    import checkpoint_utils

    entries = checkpoint_utils.list_checkpoints(CHECKPOINT_DIRECTORY)
    if not entries:
        warn("checkpoints present", f"none in {CHECKPOINT_DIRECTORY}",
             "train one, or copy one down from your training run")
        return

    for entry in entries:
        if "error" in entry:
            check(f"{entry['name']} readable", False, entry["error"])
            continue
        detail = (f"{entry['total_parameters']/1e6:.0f}M params, ctx {entry['max_sequence_length']}, "
                  f"step {entry['step']}, val {entry['best_val']:.3f}, "
                  f"{checkpoint_utils.format_bytes(entry['size_bytes'])}")
        check(f"{entry['name']} readable", True, detail)
        if entry["has_optimizer_state"]:
            warn(f"{entry['name']} carries optimizer state",
                 "about two thirds of the file is AdamW state that inference never touches",
                 f"python checkpoint_utils.py strip ../checkpoints/{entry['name']}")


# ------------------------------------------------------------ 7. datasets

def test_datasets():
    section("6. datasets")

    for split in ("train", "val"):
        path = os.path.join(DATA_DIRECTORY, f"{split}.bin")
        if not os.path.exists(path):
            warn(f"pretraining {split}.bin present", f"missing at {path}", "run data_prepare.py")
            continue
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        check(f"pretraining {split}.bin present", len(tokens) > 0,
              f"{len(tokens):,} tokens ({len(tokens)*2/1e9:.2f} GB)")

    import sft_data

    for split in ("train", "val"):
        try:
            ids, labels = sft_data.load_split(DATA_DIRECTORY, split, verbose=False)
        except FileNotFoundError as error:
            warn(f"SFT {split} data present", str(error).split(".")[0], "run sft_prepare.py")
            continue

        check(f"SFT {split} data present", True, f"{len(ids):,} examples of {ids.shape[1]} tokens")
        check(f"SFT {split} ids and labels are the same shape", ids.shape == labels.shape)

        sample_rows = np.asarray(ids[:64]), np.asarray(labels[:64])
        row_ids, row_labels = sample_rows

        # THE critical invariant. sft_prepare.py writes labels aligned position-for-position
        # with the input, and sft_train.py shifts them left by one at load time. If the file
        # were already shifted, that second shift would train the model to predict the token
        # AFTER the one it should -- so verify the on-disk alignment is the unshifted form.
        supervised = row_labels != -1
        aligned = int((row_labels[supervised] == row_ids[supervised]).sum())
        total_supervised = int(supervised.sum())
        check(f"SFT {split} labels are position-aligned with ids (unshifted on disk)",
              total_supervised > 0 and aligned == total_supervised,
              f"{aligned}/{total_supervised} supervised positions match",
              "if this fails, sft_train.py's torch.roll would double-shift the labels")

        supervised_fraction = float(supervised.mean())
        if supervised_fraction < 0.15:
            warn(f"SFT {split} supervision density", f"only {supervised_fraction*100:.1f}% of positions are supervised",
                 "most of each row is padding; consider packing more conversations per row")
        else:
            check(f"SFT {split} supervision density", True, f"{supervised_fraction*100:.1f}% of positions supervised")


# ------------------------------------------------- 8. synthetic training steps

def test_synthetic_training_steps():
    """Runs the inner loop of each trainer once, on a randomly-initialized
    'tiny' model with fabricated data. This is a shape-and-alignment check,
    not a training run: nothing is written to ../checkpoints and no real
    dataset is touched."""
    section("7. synthetic optimizer steps (fabricated data, nothing saved)")

    from config import MODEL_PRESETS
    from model import GPT
    from sft_train import LABEL_IGNORE_INDEX, load_random_batch as sft_load_random_batch

    model_config = MODEL_PRESETS["tiny"]
    torch.manual_seed(0)
    model = GPT(model_config)
    model.train()
    optimizer = model.configure_optimizer(0.1, 1e-3, (0.9, 0.95), "cpu")

    # --- pretraining-style step: targets are the inputs shifted by one ---
    tokens = torch.randint(0, model_config.vocabulary_size, (2, model_config.max_sequence_length + 1))
    input_ids, target_ids = tokens[:, :-1], tokens[:, 1:]
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(input_ids, target_ids)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    check("pretraining step produces a finite loss", torch.isfinite(loss).item(), f"loss {loss.item():.3f}")
    check("pretraining step produces finite gradients", np.isfinite(gradient_norm), f"grad norm {gradient_norm:.3f}")
    expected_loss = float(np.log(model_config.vocabulary_size))
    check("untrained loss is near ln(vocabulary_size)", abs(loss.item() - expected_loss) < 1.5,
          f"{loss.item():.2f} vs expected ~{expected_loss:.2f}")

    # --- SFT-style step: exercises the real batch loader, including the label shift ---
    example_count, sequence_length = 8, model_config.max_sequence_length
    fake_ids = np.random.randint(0, model_config.vocabulary_size, (example_count, sequence_length)).astype(np.int32)
    fake_labels = fake_ids.copy()
    fake_labels[:, : sequence_length // 2] = LABEL_IGNORE_INDEX  # first half is the "prompt"

    batch_ids, batch_labels = sft_load_random_batch(fake_ids, fake_labels, 2, "cpu", np.random.RandomState(0))
    check("SFT loader returns matching shapes", batch_ids.shape == batch_labels.shape, str(tuple(batch_ids.shape)))
    check("SFT loader ignores the final position", bool((batch_labels[:, -1] == LABEL_IGNORE_INDEX).all()),
          "nothing follows the last token, so it can teach nothing")

    # after the shift, a supervised position's label must be the NEXT input token
    supervised_positions = (batch_labels[0] != LABEL_IGNORE_INDEX).nonzero().flatten()
    position = int(supervised_positions[0])
    check("SFT label shift predicts the NEXT token",
          int(batch_labels[0, position]) == int(batch_ids[0, position + 1]),
          f"label at {position} = {int(batch_labels[0, position])}, next input = {int(batch_ids[0, position + 1])}",
          "a mismatch here means the model is trained to predict the token it was just shown")

    optimizer.zero_grad(set_to_none=True)
    _, sft_loss = model(batch_ids, batch_labels)
    sft_loss.backward()
    optimizer.step()
    check("SFT step produces a finite loss", torch.isfinite(sft_loss).item(), f"loss {sft_loss.item():.3f}")

    # --- atomic checkpoint round trip ---
    import checkpoint_utils
    import dataclasses
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = os.path.join(temporary_directory, "roundtrip.pt")
        checkpoint_utils.save_checkpoint_atomically({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": 7, "best_val": 1.25, "model_config": dataclasses.asdict(model_config),
        }, path)
        from config import ModelConfig
        reloaded, reloaded_config, info = checkpoint_utils.load_for_inference(path, "cpu", ModelConfig, GPT)
        check("checkpoint round-trips", info["step"] == 7 and reloaded_config.num_layers == model_config.num_layers,
              f"step {info['step']}, val {info['best_val']}")

        stripped, before, after = checkpoint_utils.strip_optimizer_state(path)
        check("stripping optimizer state shrinks the file", after < before,
              f"{checkpoint_utils.format_bytes(before)} -> {checkpoint_utils.format_bytes(after)}")

        # a stripped checkpoint must still load for inference
        _, _, stripped_info = checkpoint_utils.load_for_inference(stripped, "cpu", ModelConfig, GPT)
        check("stripped checkpoint still loads", not stripped_info["had_optimizer_state"])


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__,
                                               formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--skip-slow", action="store_true",
                                  help="skip the --help subprocesses and the synthetic optimizer steps")
    args = argument_parser.parse_args()

    print("project health check -- no training is started by this script\n" + "=" * 62)

    test_imports()
    if not args.skip_slow:
        test_help()
    test_chat_format()
    test_checkpoints()
    test_datasets()
    if not args.skip_slow:
        test_synthetic_training_steps()

    print("\n" + "=" * 62)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed, {len(WARNED)} warnings")
    if FAILED:
        print("\nfailures:")
        for name, hint in FAILED:
            print(f"  {name}" + (f"\n    -> {hint}" if hint else ""))
        sys.exit(1)
    print("\neverything checks out.")


if __name__ == "__main__":
    main()
