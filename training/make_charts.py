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
