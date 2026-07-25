"""Chat data: turning conversations into packed rows, and moving those rows around.

Two jobs that always travel together. `pack_conversations` takes a list of conversations
and returns the fixed-width token rows the fine-tune trains on, with a label mask that
hides everything except assistant text. `load_split` and `save_split` move those rows on
and off disk.

Packing rather than padding is what makes the run affordable: conversations are laid end to
end into 1,024-token rows and a new row only starts when the next conversation will not
fit, which took useful tokens per batch from roughly 40% to 61%.
"""
from chat_format import ChatSpecialTokens
import numpy as np
import os
import shutil
import zipfile



# --------------------------------------------------------------------------
# Storage: those rows on and off disk
# --------------------------------------------------------------------------

ARRAY_NAMES = ("ids", "labels")


def _npy_path(npz_path, array_name):
    root, _ = os.path.splitext(npz_path)
    return f"{root}_{array_name}.npy"


def _is_up_to_date(npy_path, npz_path):
    return (os.path.exists(npy_path)
            and os.path.getmtime(npy_path) >= os.path.getmtime(npz_path))


def convert_npz_to_npy(npz_path, verbose=True):
    """Extracts each member of the .npz to a sibling .npy file, streaming the
    bytes so peak memory stays flat. Returns {array_name: npy_path}.

    Skips any member whose .npy is already present and newer than the .npz.
    """
    produced = {}
    needed = [name for name in ARRAY_NAMES if not _is_up_to_date(_npy_path(npz_path, name), npz_path)]
    for name in ARRAY_NAMES:
        produced[name] = _npy_path(npz_path, name)
    if not needed:
        return produced

    if verbose:
        print(f"preparing memory-mappable copies of {os.path.basename(npz_path)} "
              f"({', '.join(needed)}) -- one-time step")

    with zipfile.ZipFile(npz_path) as archive:
        member_names = {os.path.splitext(member)[0]: member for member in archive.namelist()}
        for name in needed:
            if name not in member_names:
                raise KeyError(f"{npz_path} has no array named '{name}' (found: {sorted(member_names)})")
            destination = produced[name]
            temporary = destination + ".tmp"
            with archive.open(member_names[name]) as source, open(temporary, "wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            os.replace(temporary, destination)
            if verbose:
                print(f"  {name}: {os.path.getsize(destination) / 1e9:.2f} GB -> {os.path.basename(destination)}")
    return produced


def load_split(data_directory, split_name, verbose=True):
    """Returns (ids, labels) as memory-mapped arrays for "train" or "val".

    Prefers standalone .npy files (written directly by a current
    sft_prepare.py); falls back to converting an existing .npz once.
    """
    direct_ids = os.path.join(data_directory, f"sft_{split_name}_ids.npy")
    direct_labels = os.path.join(data_directory, f"sft_{split_name}_labels.npy")
    npz_path = os.path.join(data_directory, f"sft_{split_name}.npz")
    have_direct = os.path.exists(direct_ids) and os.path.exists(direct_labels)

    # A current sft_prepare.py writes the .npy files directly. Only fall back to converting a
    # .npz when there is no .npy, or when the .npz is newer -- otherwise a stale .npz left over
    # from an earlier run would silently overwrite freshly prepared data.
    npz_is_newer = (os.path.exists(npz_path) and have_direct
                    and os.path.getmtime(npz_path) > os.path.getmtime(direct_ids))
    if os.path.exists(npz_path) and (not have_direct or npz_is_newer):
        paths = convert_npz_to_npy(npz_path, verbose=verbose)
        direct_ids, direct_labels = paths["ids"], paths["labels"]
    elif not have_direct:
        raise FileNotFoundError(
            f"no SFT data for split '{split_name}' in {data_directory}. "
            f"Expected sft_{split_name}.npz or sft_{split_name}_ids.npy + sft_{split_name}_labels.npy. "
            f"Run sft_prepare.py first."
        )

    ids = np.load(direct_ids, mmap_mode="r")
    labels = np.load(direct_labels, mmap_mode="r")
    if ids.shape != labels.shape:
        raise ValueError(f"ids shape {ids.shape} != labels shape {labels.shape} for split '{split_name}'")
    return ids, labels


def save_split(out_directory, split_name, ids_array, labels_array):
    """Writes a split as two standalone .npy files, which need no conversion
    step at training time."""
    os.makedirs(out_directory, exist_ok=True)
    np.save(os.path.join(out_directory, f"sft_{split_name}_ids.npy"), ids_array)
    np.save(os.path.join(out_directory, f"sft_{split_name}_labels.npy"), labels_array)


def describe(data_directory, split_name="train"):
    """Summary used by the CLI below and by sft_train.py's startup banner:
    how much of the data is actually supervised, and how much is padding."""
    ids, labels = load_split(data_directory, split_name, verbose=False)
    sample_size = min(len(ids), 2000)
    sample_indices = np.linspace(0, len(ids) - 1, sample_size).astype(np.int64)
    label_sample = np.asarray(labels[sample_indices])
    supervised_fraction = float((label_sample != -1).mean())
    return {
        "examples": int(len(ids)),
        "sequence_length": int(ids.shape[1]),
        "supervised_fraction": supervised_fraction,
        "supervised_tokens_per_example": supervised_fraction * int(ids.shape[1]),
    }


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser(description=__doc__,
                                               formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--data-dir", default="../data")
    args = argument_parser.parse_args()

    for split in ("train", "val"):
        try:
            summary = describe(args.data_dir, split)
        except FileNotFoundError as error:
            print(f"{split}: {error}")
            continue
        print(f"{split}: {summary['examples']:,} examples of {summary['sequence_length']} tokens; "
              f"{summary['supervised_fraction']*100:.1f}% of positions are supervised "
              f"(~{summary['supervised_tokens_per_example']:.0f} answer tokens per example, "
              f"the rest is the question and padding)")
