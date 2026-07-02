"""
Pull a real, diverse pretraining text corpus from public Hugging Face
datasets -- broad web text, encyclopedic text, and code -- so the base model
learns general language (and some code), not one narrow domain. Everything
here is real internet-scale data, not a small curated book/history corpus.

Writes plain text documents separated by "<|doc_end|>" to --out, which
data_prepare.py then tokenizes into train.bin/val.bin.

Streams the dataset (no need to download the whole thing -- FineWeb alone is
many terabytes) and stops once --target-mb is reached.

Usage:
    python download_pretrain_data.py --preset fineweb-edu --target-mb 500 \
        --out ../data/raw_corpus.txt --resume

    # mix in code so the model can also read/write code:
    python download_pretrain_data.py --preset code --target-mb 100 \
        --out ../data/raw_corpus.txt --resume

    # or point at any HF dataset directly:
    python download_pretrain_data.py --hf-path allenai/c4 --hf-config en \
        --text-field text --target-mb 300 --out ../data/raw_corpus.txt --resume
"""
import argparse
import re

from datasets import load_dataset

# each preset: (huggingface_dataset_path, huggingface_config_name, split, text_field_name, description)
DATASET_PRESETS = {
    "fineweb-edu": (
        "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text",
        "Educational-quality filtered web crawl (2024). The best general-purpose "
        "'broad internet text' source for a small model -- start here.",
    ),
    "openwebtext": (
        "Skylion007/openwebtext", None, "train", "text",
        "Reddit-linked web pages, an open reproduction of GPT-2's original training set. "
        "More casual/conversational tone than Wikipedia or FineWeb.",
    ),
    "c4": (
        "allenai/c4", "en", "train", "text",
        "Common Crawl filtered for reasonably clean English web text; huge and broad.",
    ),
    "wikipedia": (
        "wikimedia/wikipedia", "20231101.en", "train", "text",
        "Full Wikipedia dump via HF -- factual/encyclopedic text, good for grounding.",
    ),
    "code": (
        "bigcode/the-stack-smol", "data/python", "train", "content",
        "Small sample of permissively-licensed source code (Python). Adds code "
        "reading/writing ability, like a real assistant model has.",
    ),
    "bookcorpus": (
        "bookcorpus/bookcorpus", None, "train", "text",
        "Long-form narrative prose (novels). Good for fluent, coherent long passages; "
        "mix this in as a MINORITY of your corpus, not the whole thing.",
    ),
}


def clean_text(raw_text: str) -> str:
    cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text)  # collapse 3+ consecutive blank lines down to 1
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)  # collapse runs of spaces/tabs into a single space
    return cleaned_text.strip()


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--preset", choices=list(DATASET_PRESETS.keys()), default=None,
                                  help="use one of the curated datasets below instead of --hf-path/--hf-config/--text-field")
    argument_parser.add_argument("--hf-path", type=str, default=None, help="any dataset id on huggingface.co/datasets")
    argument_parser.add_argument("--hf-config", type=str, default=None)
    argument_parser.add_argument("--split", type=str, default="train")
    argument_parser.add_argument("--text-field", type=str, default="text")
    argument_parser.add_argument("--target-mb", type=float, default=300.0)
    argument_parser.add_argument("--out", type=str, default="../data/raw_corpus.txt")
    argument_parser.add_argument("--resume", action="store_true", help="append to --out instead of overwriting")
    argument_parser.add_argument("--list", action="store_true", help="print the available --preset options and exit")
    args = argument_parser.parse_args()

    if args.list or (args.preset is None and args.hf_path is None):
        print("available --preset options:\n")
        for preset_name, (dataset_path, config_name, split, text_field, description) in DATASET_PRESETS.items():
            config_suffix = f", config={config_name!r}" if config_name else ""
            print(f"  {preset_name:14s} {dataset_path}{config_suffix}\n{'':16s}{description}\n")
        print("or pass --hf-path/--hf-config/--text-field for any other Hugging Face dataset.")
        print("\nrecommended mix for a broad, ChatGPT-like base model:")
        print("  python download_pretrain_data.py --preset fineweb-edu --target-mb 600 --out ../data/raw_corpus.txt")
        print("  python download_pretrain_data.py --preset code        --target-mb 100 --out ../data/raw_corpus.txt --resume")
        print("  python download_pretrain_data.py --preset wikipedia   --target-mb 100 --out ../data/raw_corpus.txt --resume")
        return

    if args.preset:
        dataset_path, config_name, split, text_field, description = DATASET_PRESETS[args.preset]
        print(f"using preset {args.preset!r}: {dataset_path} ({description})")
    else:
        dataset_path, config_name, split, text_field = args.hf_path, args.hf_config, args.split, args.text_field

    print(f"streaming {dataset_path} (config={config_name}, split={split})...")
    dataset = load_dataset(dataset_path, config_name, split=split, streaming=True)

    target_bytes = int(args.target_mb * 1024 * 1024)
    file_mode = "a" if args.resume else "w"
    bytes_written_so_far = 0
    if args.resume:
        try:
            bytes_written_so_far = len(open(args.out, "r", encoding="utf-8").read().encode("utf-8"))
        except FileNotFoundError:
            bytes_written_so_far = 0

    with open(args.out, file_mode, encoding="utf-8") as output_file:
        for document_index, example in enumerate(dataset):
            document_text = example.get(text_field, "")
            if not document_text or len(document_text) < 200:
                continue  # skip very short documents/stubs, which add noise without much learning signal
            document_text = clean_text(document_text)
            output_file.write(document_text)
            output_file.write("\n\n<|doc_end|>\n\n")  # marks where one document ends and the next begins, for data_prepare.py
            bytes_written_so_far += len(document_text.encode("utf-8"))

            if document_index % 500 == 0:
                megabytes_written = bytes_written_so_far / (1024 * 1024)
                print(f"\r{megabytes_written:8.2f} / {args.target_mb:.0f} MB  ({document_index} documents)", end="", flush=True)

            if bytes_written_so_far >= target_bytes:
                break

    print(f"\ndone: wrote {bytes_written_so_far/1024/1024:.2f} MB to {args.out}")


if __name__ == "__main__":
    main()
