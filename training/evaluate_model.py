"""Measures what the model can actually do, and writes the numbers to JSON.

Three kinds of measurement, none of them estimated:

  * perplexity on held-out FineWeb-Edu the model never saw;
  * zero-shot and few-shot accuracy on standard multiple-choice benchmarks,
    scored the way lm-evaluation-harness scores them: the option whose
    continuation the model finds most likely wins, with a length-normalised
    variant reported alongside;
  * inference latency and throughput on this machine.

Chance level is reported next to every benchmark, because for a model this
size that is the number that matters -- "31%" means nothing until you know
whether guessing scores 25% or 50%.

    python evaluate_model.py --ckpt ../checkpoints/moonfrost-777m-instruct-v2.pt
    python evaluate_model.py --ckpt ... --limit 300 --out ../docs/eval.json
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from checkpoint_utils import load_for_inference
from config import ModelConfig, count_active_parameters_per_token, count_total_parameters
from model import GPT

# (name, hugging face path, config, split, chance level)
BENCHMARKS = [
    ("ARC-Easy", "allenai/ai2_arc", "ARC-Easy", "test", 0.25),
    ("ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge", "test", 0.25),
    ("PIQA", "ybisk/piqa", None, "validation", 0.50),
    ("HellaSwag", "Rowan/hellaswag", None, "validation", 0.25),
    ("WinoGrande", "allenai/winogrande", "winogrande_xs", "validation", 0.50),
    ("BoolQ", "google/boolq", None, "validation", 0.50),
    ("MMLU", "cais/mmlu", "all", "test", 0.25),
]


class HuggingFaceAdapter:
    """Lets a transformers model be scored by exactly the same code path as ours.

    Comparing our numbers against figures copied from other people's model cards would
    compare harnesses as much as models: prompt wording, whether scores are length
    normalised and how many examples are used all move the result by several points.
    Running the reference models here, on the same prompts, removes that.
    """

    def __init__(self, name, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float32).to(device).eval()
        self.device = device
        self.name = name

    def encode(self, text):
        return type("Encoded", (), {"ids": self.tokenizer.encode(text, add_special_tokens=False)})()

    def __call__(self, tokens, target_token_ids=None):
        return self.model(tokens).logits, None


def sequence_logprob(model, prompt_ids, continuation_ids, device):
    """Total and per-token log probability of `continuation_ids` following `prompt_ids`."""
    ids = (prompt_ids + continuation_ids)[-1024:]
    if len(continuation_ids) >= len(ids):
        continuation_ids = continuation_ids[-(len(ids) - 1):]
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(tensor, target_token_ids=tensor)   # targets force full-sequence logits
    log_probs = F.log_softmax(logits[0].float(), dim=-1)

    total = 0.0
    start = len(ids) - len(continuation_ids)
    for offset, token in enumerate(continuation_ids):
        total += log_probs[start + offset - 1, token].item()
    return total, total / max(len(continuation_ids), 1)


def as_choices(task, row):
    """Normalises one row of each benchmark into (context, [options], answer index)."""
    if task in ("ARC-Easy", "ARC-Challenge"):
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        if row["answerKey"] not in labels:
            return None
        return (f"Question: {row['question']}\nAnswer:",
                [" " + text for text in texts], labels.index(row["answerKey"]))
    if task == "PIQA":
        return (f"Question: {row['goal']}\nAnswer:",
                [" " + row["sol1"], " " + row["sol2"]], int(row["label"]))
    if task == "HellaSwag":
        return (row["ctx"], [" " + ending for ending in row["endings"]], int(row["label"]))
    if task == "WinoGrande":
        sentence, answer = row["sentence"], int(row["answer"]) - 1
        before, after = sentence.split("_", 1) if "_" in sentence else (sentence, "")
        return (before.strip(), [f" {row['option1']}{after}", f" {row['option2']}{after}"], answer)
    if task == "MMLU":
        return (f"Question: {row['question']}\nAnswer:",
                [" " + str(choice) for choice in row["choices"]], int(row["answer"]))
    if task == "BoolQ":
        return (f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:",
                [" no", " yes"], int(bool(row["answer"])))
    return None
