"""
Train a byte-level BPE (Byte-Pair Encoding) tokenizer from scratch on our
own downloaded corpus -- the same tokenization scheme GPT-2/GPT-4/Llama use,
but with a vocabulary we train ourselves instead of reusing anyone else's.
This is a real, standard, well-documented tokenization algorithm -- not a
copy of any proprietary tokenizer. Default vocabulary size is 65536, a solid
modern choice on its own merits (a larger vocabulary means fewer tokens per
unit of text, which means cheaper training and inference, with diminishing
returns above this scale for a model this size).

This is NOT "fine-tuning a tokenizer" -- there is no pretrained tokenizer
here. tokenizers.ByteLevelBPETokenizer just runs the BPE merge-learning
algorithm on whatever text file you point it at: it starts from individual
bytes and repeatedly merges the most frequent adjacent pair into a new
token, until it has learned --vocab-size tokens total.

Besides <|endoftext|> (marks the end of one document/conversation) and
<|pad|> (fills unused space at the end of a fixed-length training example),
this also reserves:
  - <|system|> / <|user|> / <|assistant|>   chat-turn markers (sft_prepare.py
    needs these as exact single tokens for instruction-tuning)
  - <|image_start|> / <|image|> / <|image_end|>   UNUSED during text-only
    training, reserved now so that adding a vision encoder later (see
    MULTIMODAL.md) never requires retraining this tokenizer -- which would
    otherwise force retraining the base model too, since the model's
    embedding table is indexed by token id.

Reserving these special tokens now means you don't have to retrain the
tokenizer (and therefore the base model) when you get to either of those
later stages.

Usage:
    python tokenizer_train.py --input ../data/raw_corpus.txt --vocab-size 65536 --out-dir ../tokenizer
"""
import argparse
import os
from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|image_start|>", "<|image|>", "<|image_end|>",
]


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input", type=str, default="../data/raw_corpus.txt")
    argument_parser.add_argument("--vocab-size", type=int, default=65536)
    argument_parser.add_argument("--out-dir", type=str, default="../tokenizer")
    args = argument_parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[args.input],
        vocab_size=args.vocab_size,
        min_frequency=2,  # a candidate merge must appear at least this many times in the corpus to be learned
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(args.out_dir)  # writes vocab.json + merges.txt (the raw BPE artifacts)

    # also save a single merged tokenizer.json, which is what every other script in this
    # project actually loads via `tokenizers.Tokenizer.from_file(...)`
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))

    print(f"trained BPE tokenizer, vocabulary_size={tokenizer.get_vocab_size()}, saved to {args.out_dir}")
    sample_sentence = "The quick brown fox jumps over the lazy dog. def foo(x): return x + 1"
    sample_token_ids = tokenizer.encode(sample_sentence).ids
    print(f"sanity check -> {sample_sentence!r}")
    print(f"  token ids: {sample_token_ids}")
    print(f"  decoded back: {tokenizer.decode(sample_token_ids)!r}")
    print(f"  special tokens reserved: {SPECIAL_TOKENS}")


if __name__ == "__main__":
    main()
