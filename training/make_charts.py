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
