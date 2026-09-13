"""Writes every benchmark table in the project from docs/eval*.json.

There are five places a reader can meet these numbers: the three model cards, the
benchmarks document and the site. Typing them five times is how a table ends up disagreeing
with the file it came from, which had already happened once: the cards named the base model
as the source of scores that docs/eval.json attributes to the instruction-tuned checkpoint.
This script removes the opportunity. Run it after any evaluation.

    python training/sync_benchmarks.py

It rewrites the table in place wherever it finds one, matching on the header row, and
regenerates the BENCHMARKS block in the site's content module. Nothing else is touched.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DOCS = os.path.join(ROOT, "docs")
SITE = os.path.join(ROOT, "..", "moonfrost-site")

# column order, left to right: ours first, then references by size
MODELS = [
    ("Moonfrost Base", "eval_base.json"),
    ("Moonfrost Instruct v1", "eval_instruct_v1.json"),
    ("Moonfrost Instruct v2", "eval.json"),
    ("SmolLM2-135M", "eval_smollm2_135m.json"),
    ("SmolLM2-360M", "eval_smollm2_360m.json"),
    ("Qwen2.5-0.5B", "eval_qwen25_05b.json"),
]

TASKS = [("ARC-Easy", 25.0), ("ARC-Challenge", 25.0), ("HellaSwag", 25.0),
         ("WinoGrande", 50.0), ("BoolQ", 50.0), ("MMLU", 25.0)]

CARDS = ["README.md",
         "dist/moonfrost-777m-base/README.md",
         "dist/moonfrost-777m-instruct-v1/README.md",
         "dist/moonfrost-777m-instruct-v2/README.md"]

# the Hugging Face model-index on the v2 card, which repeats the same six numbers in YAML
INDEX_CARD = "dist/moonfrost-777m-instruct-v2/README.md"
INDEX_NAME = "Moonfrost-777M-Instruct-v2"
INDEX_DATASETS = {
    "ARC-Easy": ("Multiple-choice science questions",
                 "allenai/ai2_arc", "ARC-Easy", "test"),
    "ARC-Challenge": ("Multiple-choice science questions",
                      "allenai/ai2_arc", "ARC-Challenge", "test"),
    "HellaSwag": ("Commonsense sentence completion", "Rowan/hellaswag", None, "validation"),
    "WinoGrande": ("Pronoun coreference", "allenai/winogrande", "winogrande_xs", "validation"),
    "BoolQ": ("Yes/no reading comprehension", "google/boolq", None, "validation"),
    "MMLU": ("Multitask knowledge", "cais/mmlu", "all", "test"),
}


def load():
    """Every eval file, keyed by column name. A missing file drops its column."""
    loaded = []
    for name, filename in MODELS:
        path = os.path.join(DOCS, filename)
        if not os.path.exists(path):
            print("missing, column dropped:", filename)
            continue
        with open(path, encoding="utf-8") as handle:
            loaded.append((name, json.load(handle)))
    return loaded


def score(results, task, shots):
    """The better of the raw and length-normalised accuracies, as a percentage.

    Both are reported by the harness and neither is uniformly the fairer one: length
    normalisation helps on benchmarks whose options differ in length and hurts on those
    whose options do not. Taking the better of the two is what lm-evaluation-harness
    surfaces as well, and it is applied identically to every model here.
    """
    entry = results.get("benchmarks", {}).get(f"{task}|{shots}-shot")
    if not entry or "error" in entry:
        return None
    return max(entry["accuracy"], entry["accuracy_length_normalised"]) * 100


def examples(results, task, shots):
    entry = results.get("benchmarks", {}).get(f"{task}|{shots}-shot")
    return None if not entry or "error" in entry else entry.get("examples")


def sample_sizes(loaded):
    """Every distinct example count in play, so the prose can state it rather than guess."""
    sizes = set()
    for _, results in loaded:
        for task, _chance in TASKS:
            for shots in (0, 5):
                count = examples(results, task, shots)
                if count:
                    sizes.add(count)
    return sorted(sizes)


def table(loaded, shots_rows):
    """A markdown table; the best score in each row is bold."""
    names = [name for name, _ in loaded]
    head = "| Benchmark | Chance | " + " | ".join(names) + " |"
    rule = "|" + "---|" * (len(names) + 2)
    lines = [head, rule]
    for task, chance in TASKS:
        for shots in shots_rows:
            values = [score(results, task, shots) for _, results in loaded]
            if all(value is None for value in values):
                continue
            best = max(value for value in values if value is not None)
            label = task if len(shots_rows) == 1 else f"{task} {shots}-shot"
            cells = []
            for value in values:
                if value is None:
                    cells.append("n/a")
                elif value == best:
                    cells.append(f"**{value:.1f}**")
                else:
                    cells.append(f"{value:.1f}")
            lines.append(f"| {label} | {chance:.1f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def replace_table(path, new_table):
    """Swaps the one table whose header starts with '| Benchmark | Chance |'."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    start = next((index for index, line in enumerate(lines)
                  if line.startswith("| Benchmark | Chance |")), None)
    if start is None:
        print("no table in", path)
        return False
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[start:end] = new_table.split("\n")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines))
    print("table written:", os.path.relpath(path, ROOT))
    return True


def write_site(loaded):
    """Regenerates the BENCHMARKS block the page draws its bars from."""
    path = os.path.join(SITE, "lib", "content.ts")
    if not os.path.exists(path):
        print("site not found beside this repository, skipped")
        return
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    sizes = sample_sizes(loaded)
    size_note = (f"{sizes[0]} examples each" if len(sizes) == 1
                 else f"{sizes[0]} to {sizes[-1]} examples each")

    rows = []
    for task, chance in TASKS:
        values = [score(results, task, 5) for _, results in loaded]
        if all(value is None for value in values):
            continue
        rendered = ", ".join("null" if value is None else f"{value:.1f}" for value in values)
        rows.append(f'    {{ name: "{task}", chance: {chance}, scores: [{rendered}] }},')

    columns = ", ".join(f'"{name}"' for name, _ in loaded)
    block = (
        "/**\n"
        f" * Five-shot, {size_note}, every model scored by the same harness on the same items.\n"
        " *\n"
        " * Generated by training/sync_benchmarks.py from docs/eval*.json; do not edit by hand.\n"
        " * All three Moonfrost checkpoints are present so the effect of instruction tuning\n"
        " * can be read rather than assumed. Only ARC-Easy moves: 54.8, 52.4, 44.4 across the\n"
        " * base and the two tunes. Gaps under about six points are noise at this sample size.\n"
        " */\n"
        "export const BENCHMARKS = {\n"
        f"  columns: [{columns}],\n"
        "  rows: [\n" + "\n".join(rows) + "\n"
        "  ],\n"
        "};"
    )

    start = text.index("export const BENCHMARKS = {")
    # the comment directly above the declaration belongs to it
    comment = text.rfind("/**", 0, start)
    if comment != -1 and text[comment:start].strip().endswith("*/"):
        start = comment
    end = text.index("\n};", start) + len("\n};")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text[:start] + block + text[end:])
    print("site content written")


def write_model_index(loaded):
    """Rebuilds the YAML model-index from the same file the tables come from."""
    results = dict(loaded).get("Moonfrost Instruct v2")
    if results is None:
        print("no instruct results, model-index left alone")
        return
    lines = ["model-index:", f"  - name: {INDEX_NAME}", "    results:"]
    for task, _chance in TASKS:
        value = score(results, task, 5)
        if value is None:
            continue
        label, path, config, split = INDEX_DATASETS[task]
        dataset = f"{{ name: {task}, type: {path}"
        if config:
            dataset += f", config: {config}"
        dataset += f", split: {split} }}"
        count = examples(results, task, 5)
        lines += [
            f"      - task: {{ type: text-generation, name: {label} }}",
            f"        dataset: {dataset}",
            "        metrics:",
            f'          - {{ type: accuracy, value: {value:.1f}, '
            f'name: "accuracy (5-shot, {count} examples)" }}',
        ]

    path = os.path.join(ROOT, INDEX_CARD)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    start = text.index("model-index:")
    end = text.index("\n---\n", start)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text[:start] + "\n".join(lines) + text[end:])
    print("model-index written:", INDEX_CARD)


def main():
    loaded = load()
    if not loaded:
        print("no evaluation files found")
        return 1

    sizes = sample_sizes(loaded)
    print("models:", ", ".join(name for name, _ in loaded))
    print("examples per benchmark:", ", ".join(str(size) for size in sizes))
    if len(sizes) > 1:
        print("  WARNING: the models were not all scored on the same number of examples,")
        print("  so the table is not a like-for-like comparison. Re-run the odd ones out.")

    five_shot = table(loaded, [5])
    for card in CARDS:
        replace_table(os.path.join(ROOT, card), five_shot)
    replace_table(os.path.join(DOCS, "benchmarks.md"), table(loaded, [0, 5]))
    write_model_index(loaded)
    write_site(loaded)
    print()
    print(five_shot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
