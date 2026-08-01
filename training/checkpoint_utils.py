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
