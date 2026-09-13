"""Draws the charts that go on the model cards, from the measured results.

Everything here reads docs/eval*.json and docs/wandb/*.csv, so the pictures cannot drift
away from the numbers. Output lands in docs/charts/ as PNG at 2x for sharp rendering.

    python training/make_charts.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
OUT = os.path.join(DOCS, "charts")

PINK = "#c2428f"
GREY = "#9aa0ac"
DARK = "#31333f"
FAINT = "#d7dae1"

# three tints of one hue for our three checkpoints and three separable neutrals for the
# references, so the legend reads without a second hue and our group stays visibly a group
REFERENCES = [
    ("Moonfrost Base", "eval_base.json", "#f0c6e2"),
    ("Moonfrost Instruct v1", "eval_instruct_v1.json", "#dc8ec1"),
    ("Moonfrost Instruct v2", "eval.json", PINK),
    ("SmolLM2-135M", "eval_smollm2_135m.json", "#c8ccd4"),
    ("SmolLM2-360M", "eval_smollm2_360m.json", "#9aa0ac"),
    ("Qwen2.5-0.5B", "eval_qwen25_05b.json", "#6b7280"),
]

TASKS = [("ARC-Easy", 25), ("ARC-Challenge", 25), ("HellaSwag", 25),
         ("WinoGrande", 50), ("BoolQ", 50), ("MMLU", 25)]


def style(axes):
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(FAINT)
    axes.spines["bottom"].set_color(FAINT)
    axes.tick_params(colors="#6b7280", labelsize=9)
    axes.grid(axis="y", color=FAINT, linewidth=0.7)
    axes.set_axisbelow(True)


def best_score(results, task, shots=5):
    entry = results.get("benchmarks", {}).get(f"{task}|{shots}-shot")
    if not entry or "error" in entry:
        return None
    return max(entry["accuracy"], entry["accuracy_length_normalised"]) * 100


def load_results():
    loaded = []
    for name, filename, colour in REFERENCES:
        path = os.path.join(DOCS, filename)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            loaded.append((name, json.load(handle), colour))
    return loaded


def benchmark_chart(loaded):
    """Grouped bars, one group per benchmark, with the chance line drawn behind."""
    figure, axes = plt.subplots(figsize=(11.5, 4.6), dpi=200)
    style(axes)
    width = 0.8 / len(loaded)

    for index, (name, results, colour) in enumerate(loaded):
        values = [best_score(results, task) or 0 for task, _ in TASKS]
        offsets = [position + index * width - 0.4 + width / 2 for position in range(len(TASKS))]
        bars = axes.bar(offsets, values, width * 0.92, label=name, color=colour,
                        zorder=3)
        if name.startswith("Moonfrost"):
            for bar, value in zip(bars, values):
                axes.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.0f}",
                          ha="center", fontsize=7, color=colour if index else "#d79cc4",
                          fontweight="bold")

    # chance is the number that decides whether a score means anything at all
    for position, (_, chance) in enumerate(TASKS):
        axes.plot([position - 0.45, position + 0.45], [chance, chance],
                  color="#d0453b", linewidth=1.4, linestyle=(0, (4, 3)), zorder=4)

    axes.set_xticks(range(len(TASKS)))
    axes.set_xticklabels([task for task, _ in TASKS], fontsize=9.5, color=DARK)
    axes.set_ylabel("accuracy %", fontsize=9.5, color="#6b7280")
    axes.set_ylim(0, 78)
    axes.legend(frameon=False, fontsize=9, ncol=6, loc="upper center",
                bbox_to_anchor=(0.5, 1.12))
    axes.text(0.995, 0.02, "dashed red = chance", transform=axes.transAxes,
              ha="right", fontsize=8, color="#d0453b")
    figure.tight_layout()
    save(figure, "benchmarks.png")


def read_log(filename):
    """One JSONL log, sorted by step. Missing files give an empty list."""
    path = os.path.join(DOCS, "logs", filename)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: row.get("step", 0))
    return rows


# Each stage names the logs that survive for it. Phase 1 is two fragments: the run that
# finished it never had its log downloaded, so what is left is the attempt that dropped its
# connection at step 1,730 and the tail of the successful run, read back out of retained
# console output months later. The middle is gone, and is drawn as gone.
STAGES = [
    {"title": "Pretraining, phase 1",
     "colour": "#8b93a1",
     "segments": [("base1_train_log.jsonl", "base1_val_log.jsonl"),
                  ("recovered/phase1_tail_train_log.jsonl",
                   "recovered/phase1_tail_val_log.jsonl")],
     "ylim": (2.85, 5.4),
     "opens": 10.73,
     "note": ("9,542 steps in 227 min", "steps 1,730-9,160 unlogged")},
    {"title": "Pretraining, phase 2",
     "colour": "#5b626f",
     "segments": [("base2_train_log.jsonl", "base2_val_log.jsonl")],
     "note": ("340 min, first 105 logged", "phase ended at 2.9763")},
    {"title": "Chat fine-tune",
     "colour": PINK,
     "segments": [("chat_train_log.jsonl", "chat_val_log.jsonl")],
     "note": ("9,201 steps in 80 min", "released checkpoint at step 7,800")},
]


def loss_chart():
    """Train and validation loss per stage, one panel each.

    The stages share no axis: each restarts its step counter and the fine-tune runs at a
    different batch size, so a single joined curve would describe a run that never happened.
    """
    figure, axes_row = plt.subplots(1, 3, figsize=(11.4, 3.9), dpi=200)

    for axis, stage in zip(axes_row, STAGES):
        style(axis)
        axis.set_title(stage["title"], fontsize=10.5, color=DARK, pad=10)
        axis.set_xlabel("step", fontsize=9, color="#6b7280")
        axis.set_ylabel("loss", fontsize=9, color="#6b7280")

        spans, best = [], None
        for index, (train_file, val_file) in enumerate(stage["segments"]):
            train = read_log(train_file)
            val = read_log(val_file)
            if not train:
                continue
            spans.append((train[0]["step"], train[-1]["step"]))
            axis.plot([row["step"] for row in train], [row["loss"] for row in train],
                      color=stage["colour"], linewidth=0.9, alpha=0.4,
                      label="train" if index == 0 else None, zorder=3)
            if val:
                axis.plot([row["step"] for row in val], [row["val_loss"] for row in val],
                          color=stage["colour"], linewidth=2.0,
                          label="validation" if index == 0 else None, zorder=4)
                low = min(val, key=lambda row: row["val_loss"])
                if best is None or low["val_loss"] < best["val_loss"]:
                    best = low

        if "ylim" in stage:
            axis.set_ylim(*stage["ylim"])

        # the unlogged middle, drawn rather than quietly closed over. The label is placed in
        # axis-fraction height so it does not depend on the y limit being settled first.
        for (_, left), (right, _) in zip(spans, spans[1:]):
            axis.axvspan(left, right, color=FAINT, alpha=0.5, zorder=1)
            axis.text((left + right) / 2, 0.5, "not logged",
                      transform=axis.get_xaxis_transform(), ha="center", va="center",
                      fontsize=8.5, color="#8b93a1", zorder=5)

        if "opens" in stage:
            axis.annotate(f"opens at {stage['opens']}", (0.02, 0.94),
                          xycoords="axes fraction", fontsize=8.5, color="#8b93a1")

        if best is not None:
            axis.scatter([best["step"]], [best["val_loss"]], s=34, color=PINK,
                         edgecolors="white", linewidths=1.2, zorder=6)
            axis.annotate(f"{best['val_loss']:.4f}", (best["step"], best["val_loss"]),
                          textcoords="offset points", xytext=(0, 11), ha="center",
                          fontsize=8.5, color=PINK, fontweight="bold", zorder=6)

        axis.legend(frameon=False, fontsize=8.5, loc="upper right")
        axis.text(0.5, -0.40, "\n".join(stage["note"]), transform=axis.transAxes,
                  ha="center",
                  va="top", fontsize=8.5, color="#9aa0ac", linespacing=1.5)

    figure.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.28, wspace=0.28)
    save(figure, "training_loss.png")


def scale_chart():
    """Where the budget sat: tokens per parameter against Chinchilla's ratio of 20."""
    models = [("GPT-2", 1.5e9, 1e10), ("GPT-3", 1.75e11, 3e11),
              ("Chinchilla", 7e10, 1.4e12), ("Llama 2 7B", 7e9, 2e12),
              ("Llama 3 8B", 8e9, 1.5e13), ("DeepSeek-V3", 6.71e11, 1.48e13),
              ("SmolLM2-135M", 1.35e8, 2e12), ("SmolLM2-360M", 3.62e8, 4e12),
              ("Qwen2.5-0.5B", 5e8, 1.8e13),
              ("Moonfrost 777M (base)", 7.77e8, 6e9)]

    figure, axes = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    style(axes)
    axes.set_xscale("log")
    axes.set_yscale("log")

    line = [1e8, 1e12]
    axes.plot(line, [value * 20 for value in line], color="#6b7280", linewidth=1.2,
              linestyle=(0, (5, 4)), zorder=2)
    axes.text(3e11, 3e11 * 20 * 1.5, "20 tokens per parameter", fontsize=8.5,
              color="#6b7280", ha="right")

    for name, params, tokens in models:
        ours = name.startswith("Moonfrost")
        axes.scatter([params], [tokens], s=110 if ours else 42,
                     color=PINK if ours else GREY, zorder=4,
                     edgecolors="white", linewidths=1.6 if ours else 0)
        axes.annotate(name, (params, tokens), textcoords="offset points",
                      xytext=(9, -3), fontsize=8.5,
                      color=PINK if ours else "#6b7280",
                      fontweight="bold" if ours else "normal")

    def short(value):
        return (f"{value / 1e12:g}T" if value >= 1e12
                else f"{value / 1e9:g}B" if value >= 1e9
                else f"{value / 1e6:g}M")

    axes.set_xticks([1e8, 1e9, 1e10, 1e11, 1e12])
    axes.set_xticklabels([short(v) for v in (1e8, 1e9, 1e10, 1e11, 1e12)])
    axes.set_yticks([1e10, 1e11, 1e12, 1e13])
    axes.set_yticklabels([short(v) for v in (1e10, 1e11, 1e12, 1e13)])
    axes.minorticks_off()

    axes.set_xlabel("parameters", fontsize=9.5, color="#6b7280")
    axes.set_ylabel("training tokens", fontsize=9.5, color="#6b7280")
    axes.set_xlim(6e7, 4e12)
    axes.set_ylim(2e9, 1e14)
    figure.tight_layout()
    save(figure, "scale.png")


def save(figure, filename):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, filename)
    figure.savefig(path, transparent=False, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


if __name__ == "__main__":
    benchmark_chart(load_results())
    loss_chart()
    scale_chart()
