"""
Checks that sft_data.py builds training rows in exactly the layout the chat
server feeds the model at inference time, and that packing never corrupts a
conversation. Needs only the local tokenizer; no GPU, no downloads, no
training.

    python test_sft_data.py

The first check is the important one. If a training row and an inference
prompt disagreed by even one token -- a stray newline after <|assistant|>, a
system prompt placed differently -- the model would be fine-tuned on one
format and then asked questions in another, and the damage would show up
only as vaguely worse answers.
"""

import os, sys
# these live in training/tests/, so the package directory is one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import numpy as np
from tokenizers import Tokenizer

import sft_data
from chat_format import ChatSpecialTokens, build_prompt_token_ids
from sft_train import LABEL_IGNORE_INDEX, load_random_batch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main():
    tokenizer = Tokenizer.from_file(os.path.join(HERE, "..", "..", "tokenizer", "tokenizer.json"))
    special = ChatSpecialTokens(tokenizer)

    def encode(text):
        return tokenizer.encode(text.strip()).ids

    system_text = "You are a friendly assistant."
    conversation = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "Name two colours."},
        {"role": "assistant", "content": "Blue and yellow."},
    ]

    print("\nencoding matches inference")
    ids, labels = sft_data.encode_conversation(tokenizer, special, conversation, 1024)
    inference_prompt = build_prompt_token_ids(
        tokenizer, [("user", "Hi there!"), ("assistant", "Hello! How can I help you today?")],
        "Name two colours.", special, system_prompt=system_text,
    )
    check("training row begins with the exact prompt the server builds", ids[:len(inference_prompt)] == inference_prompt,
          f"{len(inference_prompt)} prompt tokens")
    final_reply = encode("Blue and yellow.") + [special.end_of_text]
    check("training row ends with the final reply and <|endoftext|>", ids[len(inference_prompt):] == final_reply)

    expected_supervised = encode("Hello! How can I help you today?") + [special.end_of_text] + final_reply
    actual_supervised = [label for label in labels if label != LABEL_IGNORE_INDEX]
    check("only assistant replies (and their <|endoftext|>) are supervised", actual_supervised == expected_supervised,
          f"{len(actual_supervised)} supervised tokens")
    check("labels are position-aligned with ids on disk",
          all(label == LABEL_IGNORE_INDEX or label == token for token, label in zip(ids, labels)))
    check("system prompt, user turns and <|assistant|> markers are not supervised",
          all(labels[position] == LABEL_IGNORE_INDEX for position, token in enumerate(ids)
              if token in (special.system, special.user, special.assistant)))

    print("\ntruncation")
    long_conversation = []
    for index in range(12):
        long_conversation.append({"role": "user", "content": f"Question number {index}: " + "tell me more " * 30})
        long_conversation.append({"role": "assistant", "content": f"Answer number {index}: " + "here is more " * 30})
    truncated = sft_data.encode_conversation(tokenizer, special, long_conversation, 256)
    check("over-long conversation is cut to fit", truncated is not None and len(truncated[0]) <= 256,
          f"{len(truncated[0]) if truncated else 0} tokens")
    check("truncation keeps the NEWEST exchange", truncated is not None and "Answer number 11" in tokenizer.decode(truncated[0]))
    check("truncation drops the OLDEST exchange", truncated is not None and "Question number 0" not in tokenizer.decode(truncated[0]))
    check("conversation without an assistant reply is skipped",
          sft_data.encode_conversation(tokenizer, special, [{"role": "user", "content": "hello?"}], 256) is None)
    oversized_single = [{"role": "user", "content": "word " * 600}, {"role": "assistant", "content": "ok"}]
    check("exchange too long to fit at all is skipped", sft_data.encode_conversation(tokenizer, special, oversized_single, 64) is None)

    print("\npacking")
    conversations = []
    for index in range(40):
        conversations.append([
            {"role": "user", "content": f"What is {index} plus {index}?"},
            {"role": "assistant", "content": f"{index} plus {index} is {2 * index}."},
        ])
    block_size = 96
    packed_ids, packed_labels, stats = sft_data.pack_conversations(tokenizer, special, conversations, block_size)
    individually = [sft_data.encode_conversation(tokenizer, special, messages, block_size) for messages in conversations]
    check("several conversations share each row", stats["conversations_per_row"] > 1.5,
          f"{stats['conversations_per_row']:.1f} per row across {stats['rows']} rows")
    check("every conversation was packed", stats["conversations"] == 40 and stats["skipped"] == 0)
    check("packing preserves every supervised token",
          int((packed_labels != LABEL_IGNORE_INDEX).sum()) == sum(sum(1 for l in lab if l != LABEL_IGNORE_INDEX) for _, lab in individually))
    check("every row starts at a conversation boundary", bool(np.isin(packed_ids[:, 0], [special.user, special.system]).all()))

    # a row must be exactly the concatenation of whole conversations followed by padding
    expected_rows, current = [], []
    for token_ids, _ in individually:
        if len(current) + len(token_ids) > block_size:
            expected_rows.append(current)
            current = []
        current = current + token_ids
    expected_rows.append(current)
    reconstructed_ok = len(expected_rows) == packed_ids.shape[0] and all(
        list(packed_ids[row, :len(expected)]) == expected
        and bool((packed_ids[row, len(expected):] == special.end_of_text).all())
        and bool((packed_labels[row, len(expected):] == LABEL_IGNORE_INDEX).all())
        for row, expected in enumerate(expected_rows)
    )
    check("rows are whole conversations followed by ignored padding", reconstructed_ok)

    print("\ninteraction with sft_train.py's label shift")
    batch_ids, batch_labels = load_random_batch(packed_ids, packed_labels, 4, "cpu", np.random.RandomState(0))
    batch_ids, batch_labels = batch_ids.numpy(), batch_labels.numpy()
    supervised_rows, supervised_cols = np.nonzero(batch_labels != LABEL_IGNORE_INDEX)
    check("after the shift, every supervised label is the NEXT input token",
          len(supervised_rows) > 0 and bool((batch_labels[supervised_rows, supervised_cols]
                                             == batch_ids[supervised_rows, supervised_cols + 1]).all()),
          f"{len(supervised_rows)} supervised positions")
    eot_positions = np.nonzero(batch_ids == special.end_of_text)
    predicts_user = [(r, c) for r, c in zip(*eot_positions) if c + 1 < block_size and batch_ids[r, c + 1] == special.user]
    check("the end of one conversation is never trained to predict the next one's <|user|>",
          all(batch_labels[r, c] == LABEL_IGNORE_INDEX for r, c in predicts_user), f"{len(predicts_user)} boundaries")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
