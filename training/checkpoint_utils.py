"""
Everything to do with reading, writing and describing checkpoint files, in
one place so train.py / sft_train.py / server.py / sample.py all
agree on the format instead of each doing their own slightly different
torch.load().

Three problems this module exists to solve:

1. Checkpoints written during training contain the AdamW optimizer state,
   which is two extra float32 buffers per parameter -- i.e. two thirds of
   the file. A 394M-parameter checkpoint is 4.7GB on disk, of which only
   1.6GB is the model. Loading that with map_location="cuda" (as the old
   server.py did) pushes all 4.7GB onto the GPU, which does not fit on a
   6GB card. load_for_inference() below reads to CPU (memory-mapped where
   possible), takes only the "model" entry, and moves just that to the GPU.

2. torch.save() writes in place. Interrupting a multi-gigabyte save leaves
   a truncated, unloadable file where a good checkpoint used to be.
   save_checkpoint_atomically() writes to a temporary file in the same
   directory and renames it over the target only once the write completed,
   so an interrupted save leaves the previous checkpoint intact.

3. Wrapping a model in torch.compile() or DistributedDataParallel prefixes
   every state-dict key ("_orig_mod." / "module."). A checkpoint saved from
   a wrapped model won't load into a bare one. normalize_state_dict_keys()
   strips those prefixes so any checkpoint loads into any model.

Command line:
    python checkpoint_utils.py list
    python checkpoint_utils.py strip ../checkpoints/sft_best.pt
    python checkpoint_utils.py strip-all --in-place
"""
import argparse
import dataclasses
import os
import tempfile

import torch


CHECKPOINT_DIRECTORY_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

# state-dict key prefixes added by wrappers that aren't part of the model itself
WRAPPER_KEY_PREFIXES = ("_orig_mod.", "module.")


def normalize_state_dict_keys(state_dict):
    """Strips torch.compile's '_orig_mod.' and DistributedDataParallel's
    'module.' prefixes (possibly both, possibly nested) from every key, so a
    checkpoint saved from a wrapped model loads into a plain GPT."""
    normalized = {}
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in WRAPPER_KEY_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        normalized[key] = value
    return normalized


def save_checkpoint_atomically(checkpoint_dict, destination_path):
    """torch.save() to a temporary file next to the destination, then rename
    it into place. os.replace() is atomic on both Windows and POSIX, so the
    destination is either the old checkpoint or the new one -- never a
    half-written file."""
    destination_directory = os.path.dirname(os.path.abspath(destination_path)) or "."
    os.makedirs(destination_directory, exist_ok=True)
    temporary_file_descriptor, temporary_path = tempfile.mkstemp(
        dir=destination_directory, prefix=os.path.basename(destination_path) + ".tmp-"
    )
    os.close(temporary_file_descriptor)
    try:
        torch.save(checkpoint_dict, temporary_path)
        os.replace(temporary_path, destination_path)
    except BaseException:
        # includes KeyboardInterrupt: clean up the partial file, leave the old checkpoint alone
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        raise


def _torch_load_tolerant(path, map_location, mmap):
    """torch.load with weights_only=True (the PyTorch 2.6+ default, and the
    safe option) but falling back to weights_only=False for older
    checkpoints that pickled a real ModelConfig object rather than a plain
    dict. mmap is skipped on the fallback because the two aren't always
    compatible."""
    try:
        return torch.load(path, map_location=map_location, mmap=mmap, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint_metadata(path):
    """Reads a checkpoint's small bookkeeping entries (step, best_val,
    model_config) WITHOUT pulling its tensors into memory, by memory-mapping
    the file. Fast enough to call on every file in a directory."""
    checkpoint = _torch_load_tolerant(path, map_location="cpu", mmap=True)
    raw_config = checkpoint["model_config"]
    if not isinstance(raw_config, dict):
        raw_config = dataclasses.asdict(raw_config)
    return {
        "step": int(checkpoint.get("step", 0)),
        "best_val": float(checkpoint.get("best_val", float("nan"))),
        "model_config": dict(raw_config),
        "has_optimizer_state": "optimizer" in checkpoint,
    }


def load_for_inference(path, device, model_config_class, gpt_class):
    """Builds a ready-to-generate model from a checkpoint without ever
    putting the optimizer state on the GPU.

    Returns (model, model_config, info_dict)."""
    checkpoint = _torch_load_tolerant(path, map_location="cpu", mmap=True)

    raw_config = checkpoint["model_config"]
    if not isinstance(raw_config, dict):
        raw_config = dataclasses.asdict(raw_config)
    model_config = model_config_class(**raw_config)

    model = gpt_class(model_config)
    state_dict = normalize_state_dict_keys(checkpoint["model"])
    # assign=True hands the loaded (memory-mapped) tensors straight to the module instead of
    # copying them into the freshly-allocated random weights, which roughly halves peak host RAM
    model.load_state_dict(state_dict, assign=True)
    # .to(device) materializes the memory-mapped tensors as it copies them across
    model = model.to(device)
    model.eval()

    info = {
        "step": int(checkpoint.get("step", 0)),
        "best_val": float(checkpoint.get("best_val", float("nan"))),
        "had_optimizer_state": "optimizer" in checkpoint,
        "path": os.path.abspath(path),
    }
    del checkpoint, state_dict
    return model, model_config, info


def load_for_training(path):
    """Loads a checkpoint for resuming training. Always reads to CPU first:
    optimizer state is moved onto the GPU by optimizer.load_state_dict()
    itself, once the parameters it belongs to are already there, so reading
    it straight to "cuda" only doubles peak VRAM for no benefit."""
    return _torch_load_tolerant(path, map_location="cpu", mmap=False)
