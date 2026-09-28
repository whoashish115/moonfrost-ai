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
