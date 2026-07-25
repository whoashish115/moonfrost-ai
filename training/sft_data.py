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
# Packing: conversations in, fixed-width rows and label masks out
# --------------------------------------------------------------------------

LABEL_IGNORE_INDEX = -1


def encode_conversation(tokenizer, special_tokens: ChatSpecialTokens, messages, max_length):
    """messages: a list of {"role": ..., "content": ...} dicts, in order.

    Returns (token_ids, labels) as Python lists no longer than max_length, or
    None if the conversation has no complete user->assistant exchange or its
    last exchange alone is too long."""
    def encode(text):
        return tokenizer.encode((text or "").strip()).ids

    system_ids = []
    exchanges = []
    pending_user_text = None
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            # a system prompt only means something at the very start of a conversation
            if special_tokens.system is not None and not exchanges and pending_user_text is None and not system_ids:
                system_ids = [special_tokens.system] + encode(content)
        elif role == "user":
            pending_user_text = content  # of two consecutive user messages, the later one is answered
        elif role == "assistant" and pending_user_text is not None:
            exchanges.append((pending_user_text, content))
            pending_user_text = None
    if not exchanges:
        return None

    blocks = []
    for user_text, assistant_text in exchanges:
        prompt_ids = [special_tokens.user] + encode(user_text) + [special_tokens.assistant]
        reply_ids = encode(assistant_text) + [special_tokens.end_of_text]
        blocks.append((prompt_ids, reply_ids))

    def total_length():
        return len(system_ids) + sum(len(prompt) + len(reply) for prompt, reply in blocks)

    while total_length() > max_length:
        if len(blocks) > 1:
            blocks.pop(0)          # oldest exchange first
        elif system_ids:
            system_ids = []        # then the system prompt
        else:
            return None            # the final exchange alone does not fit

    token_ids = list(system_ids)
    labels = [LABEL_IGNORE_INDEX] * len(system_ids)
    for prompt_ids, reply_ids in blocks:
        token_ids += prompt_ids + reply_ids
        labels += [LABEL_IGNORE_INDEX] * len(prompt_ids) + reply_ids
    return token_ids, labels


_WORKER_TOKENIZERS = {}


def encode_examples(batch, tokenizer_path, block_size):
    """Batched function for datasets.Dataset.map(num_proc=...). Loads the
    tokenizer once per worker process. Conversations that cannot be encoded
    come back as empty lists and are skipped by the packer."""
    if tokenizer_path not in _WORKER_TOKENIZERS:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        _WORKER_TOKENIZERS[tokenizer_path] = (tokenizer, ChatSpecialTokens(tokenizer))
    tokenizer, special_tokens = _WORKER_TOKENIZERS[tokenizer_path]

    all_ids, all_labels = [], []
    for messages in batch["messages"]:
        encoded = encode_conversation(tokenizer, special_tokens, messages, block_size)
        if encoded is None:
            all_ids.append([])
            all_labels.append([])
        else:
            all_ids.append(encoded[0])
            all_labels.append(encoded[1])
    return {"ids": all_ids, "labels": all_labels}


class RowPacker:
    """Greedy first-fit-in-order packing into a growable int32 array pair."""

    def __init__(self, block_size, pad_token_id, initial_rows=1024):
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.ids = np.full((initial_rows, block_size), pad_token_id, dtype=np.int32)
        self.labels = np.full((initial_rows, block_size), LABEL_IGNORE_INDEX, dtype=np.int32)
        self.row = 0
        self.fill = 0
        self.conversations = 0
        self.skipped = 0

    def _grow(self):
        extra = self.ids.shape[0]
        self.ids = np.concatenate([self.ids, np.full((extra, self.block_size), self.pad_token_id, dtype=np.int32)])
        self.labels = np.concatenate([self.labels, np.full((extra, self.block_size), LABEL_IGNORE_INDEX, dtype=np.int32)])

    def add(self, token_ids, labels):
        length = len(token_ids)
        if length == 0 or length > self.block_size:
            self.skipped += 1
            return
        if self.fill + length > self.block_size:
            self.row += 1
            self.fill = 0
        if self.row >= self.ids.shape[0]:
            self._grow()
        self.ids[self.row, self.fill:self.fill + length] = token_ids
        self.labels[self.row, self.fill:self.fill + length] = labels
        self.fill += length
        self.conversations += 1

    def result(self):
        used_rows = self.row + (1 if self.fill > 0 else 0)
        ids, labels = self.ids[:used_rows], self.labels[:used_rows]
        stats = {
            "rows": int(used_rows),
            "conversations": int(self.conversations),
            "skipped": int(self.skipped),
            "conversations_per_row": self.conversations / max(1, used_rows),
            "non_padding_fraction": float((labels != LABEL_IGNORE_INDEX).sum() + 0) / max(1, labels.size),
        }
        # count real tokens (anything not in the padded tail) separately from supervised ones
        stats["supervised_fraction"] = float((labels != LABEL_IGNORE_INDEX).mean()) if labels.size else 0.0
        return ids, labels, stats


def pack_encoded(encoded_datasets, block_size, pad_token_id, initial_rows=1024, progress_every=50000):
    """Packs one or more datasets that have "ids"/"labels" columns (the
    output of encode_examples) into (ids, labels, stats)."""
    packer = RowPacker(block_size, pad_token_id, initial_rows)
    seen = 0
    for dataset in encoded_datasets:
        for batch in dataset.iter(batch_size=2000):
            for token_ids, labels in zip(batch["ids"], batch["labels"]):
                packer.add(token_ids, labels)
                seen += 1
                if progress_every and seen % progress_every == 0:
                    print(f"  packed {seen:,} conversations into {packer.row + 1:,} rows", flush=True)
    return packer.result()


def pack_conversations(tokenizer, special_tokens, conversations, block_size):
    """In-process convenience for tests and small datasets: conversations is
    an iterable of message lists."""
    packer = RowPacker(block_size, special_tokens.end_of_text)
    for messages in conversations:
        encoded = encode_conversation(tokenizer, special_tokens, messages, block_size)
        if encoded is None:
            packer.skipped += 1
            continue
        packer.add(*encoded)
    return packer.result()


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
