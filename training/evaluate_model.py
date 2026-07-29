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


def run_benchmark(model, tokenizer, device, task, path, config, split, limit, shots=0):
    from datasets import load_dataset

    dataset = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
    # MMLU ships grouped by subject, so the first N rows would all be abstract algebra
    if task == "MMLU":
        dataset = dataset.shuffle(seed=1337)
    rows = list(dataset.select(range(min(limit, len(dataset)))))

    # few-shot demonstrations are taken from rows that are not being scored
    prefix = ""
    if shots:
        pool = list(dataset.select(range(limit, min(limit + shots * 4, len(dataset)))))
        used = 0
        for row in pool:
            parsed = as_choices(task, row)
            if not parsed:
                continue
            context, options, answer = parsed
            prefix += context + options[answer] + "\n\n"
            used += 1
            if used == shots:
                break

    correct = correct_normalised = scored = 0
    for row in rows:
        parsed = as_choices(task, row)
        if not parsed:
            continue
        context, options, answer = parsed
        prompt_ids = tokenizer.encode(prefix + context).ids
        totals, averages = [], []
        for option in options:
            total, average = sequence_logprob(model, prompt_ids, tokenizer.encode(option).ids, device)
            totals.append(total)
            averages.append(average)
        correct += int(int(np.argmax(totals)) == answer)
        correct_normalised += int(int(np.argmax(averages)) == answer)
        scored += 1

    return {"accuracy": correct / max(scored, 1),
            "accuracy_length_normalised": correct_normalised / max(scored, 1),
            "examples": scored, "shots": shots}


def held_out_perplexity(model, device, path, block_size=1024, batches=60):
    """Perplexity on the FineWeb-Edu validation shard, which no phase trained on."""
    if not os.path.exists(path):
        return None
    data = np.memmap(path, dtype=np.uint16, mode="r")
    usable = (len(data) // block_size) * block_size
    generator = np.random.default_rng(0)
    total_loss, counted = 0.0, 0
    for _ in range(batches):
        start = int(generator.integers(0, max(usable - block_size - 1, 1)))
        chunk = torch.tensor(np.asarray(data[start:start + block_size + 1], dtype=np.int64),
                             device=device)
        inputs, targets = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
        with torch.no_grad():
            _, loss = model(inputs, target_token_ids=targets)
        total_loss += float(loss)
        counted += 1
    mean = total_loss / max(counted, 1)
    return {"loss": mean, "perplexity": math.exp(mean), "batches": counted, "block_size": block_size}
