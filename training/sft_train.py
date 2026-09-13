"""
Instruction-tuning (SFT, "supervised fine-tuning"): continue training a
pretrained checkpoint from train.py on the (input_ids, labels) rows built by
sft_data.py, so the model learns to answer after
<|assistant|> instead of just continuing arbitrary text.

Same optimizer/precision/distributed machinery as train.py, but it starts
from existing weights, uses a much lower learning rate (large updates here
would wreck what pretraining learned), and each batch item is a packed row of
complete conversations rather than a random window of a token stream.

Things this version does that earlier ones did not:

  * Gradient accumulation (--grad-accum). The original loop did one
    micro-batch of 2 sequences per optimizer step -- an effective batch of 2,
    far too small and noisy for fine-tuning.
  * Memory-mapped data (see sft_data.py) instead of materializing gigabytes.
  * Atomic checkpoints, read to CPU rather than straight onto the GPU.
  * Barriers around rank-0 evaluation so multi-GPU runs cannot deadlock.
  * --lr-schedule time: the cosine decay tracks the wall-clock budget, so the
    learning rate finishes annealing exactly when paid time runs out.
  * A torch.compile probe that falls back to eager mode on a compiler failure,
    and evaluation on the uncompiled module so eval mode never recompiles.

Usage:
    python sft_train.py --init-from ../checkpoints/small_best.pt --minutes 60

    # budget-driven, e.g. on a rented GPU
    python sft_train.py --init-from v2_best.pt --minutes 16 --lr-schedule time --batch-size 24 --grad-accum 3
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import dataclasses
import json
import math
import platform
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # otherwise output sits in a buffer when stdout isn't a terminal

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

import checkpoint_utils
import sft_data
from config import ModelConfig, count_active_parameters_per_token, count_total_parameters
from ddp_utils import (cleanup_distributed_training, setup_distributed_training, synchronize_float,
                       synchronize_stop_decision)
from model import GPT

LABEL_IGNORE_INDEX = -1  # matches model.py's F.cross_entropy(..., ignore_index=-1)


def barrier(distributed_context):
    if distributed_context.is_distributed:
        torch.distributed.barrier()


def load_random_batch(all_input_ids, all_labels, batch_size, device, random_state):
    """Samples `batch_size` packed rows at random.

    The label shift: the data is written with labels aligned position-for-
    position with the input (labels[i] is the token AT position i, or -1
    where the loss should ignore it). The model's loss compares the
    prediction made AT position i against the token at position i+1, so the
    labels have to move one step left. Without this shift the model is
    trained to predict the token it was just shown -- which produces a
    beautifully low training loss and a completely useless model.
    """
    chosen_indices = np.sort(random_state.randint(0, all_input_ids.shape[0], size=batch_size))
    # sorted indices keep reads roughly sequential, which matters because these arrays are
    # memory-mapped: random access across a multi-gigabyte file is dominated by page faults
    input_ids = torch.from_numpy(np.asarray(all_input_ids[chosen_indices], dtype=np.int64))
    labels = torch.from_numpy(np.asarray(all_labels[chosen_indices], dtype=np.int64))

    labels = torch.roll(labels, shifts=-1, dims=1)
    labels[:, -1] = LABEL_IGNORE_INDEX  # nothing follows the last position, so it teaches nothing

    if "cuda" in device:
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
    else:
        input_ids, labels = input_ids.to(device), labels.to(device)
    return input_ids, labels


def cosine_between(progress, max_learning_rate, min_learning_rate):
    progress = min(max(progress, 0.0), 1.0)
    return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)


def learning_rate_at_step(step, warmup_steps, max_steps, max_learning_rate, min_learning_rate):
    if step < warmup_steps:
        return max_learning_rate * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_learning_rate
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return cosine_between(progress, max_learning_rate, min_learning_rate)


@torch.no_grad()
def estimate_loss(model, splits, batch_size, device, autocast_context, eval_iterations, random_state):
    model.eval()
    average_losses = {}
    for split_name, (split_input_ids, split_labels) in splits.items():
        losses = torch.zeros(eval_iterations)
        for iteration in range(eval_iterations):
            input_ids, labels = load_random_batch(split_input_ids, split_labels, batch_size, device, random_state)
            with autocast_context:
                _, loss = model(input_ids, labels)
            losses[iteration] = loss.item()
        average_losses[split_name] = losses.mean().item()
    model.train()
    return average_losses


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--init-from", type=str, required=True, help="pretrained checkpoint to start from (see train.py)")
    argument_parser.add_argument("--data-dir", type=str, default="../data", help="directory containing the sft_* data files")
    argument_parser.add_argument("--ckpt-dir", type=str, default="../checkpoints")
    argument_parser.add_argument("--tag", type=str, default="sft", help="checkpoint filename tag: {tag}_last.pt / {tag}_best.pt")
    argument_parser.add_argument("--resume", type=str, default=None, help="resume an in-progress SFT run from this checkpoint")
    argument_parser.add_argument("--minutes", type=float, default=60.0)
    argument_parser.add_argument("--epochs", type=float, default=2.0, help="used to derive max training steps if --max-steps isn't given")
    argument_parser.add_argument("--max-steps", type=int, default=None)
    argument_parser.add_argument("--lr-schedule", choices=["steps", "time"], default="steps",
                                  help="time: cosine over the wall-clock budget (or the step limit, whichever is nearer)")
    argument_parser.add_argument("--batch-size", type=int, default=2,
                                  help="per-GPU MICRO batch size -- how many sequences fit in memory at once. "
                                       "Lower it if you run out of memory; raise --grad-accum to compensate.")
    argument_parser.add_argument("--grad-accum", type=int, default=8,
                                  help="micro-batches per optimizer step. effective batch = batch-size * grad-accum * num_gpus.")
    argument_parser.add_argument("--max-lr", type=float, default=3e-5,
                                  help="far below pretraining's rate -- this nudges an existing model rather than learning from scratch")
    argument_parser.add_argument("--min-lr", type=float, default=3e-6)
    argument_parser.add_argument("--warmup-steps", type=int, default=50)
    argument_parser.add_argument("--weight-decay", type=float, default=0.0)
    argument_parser.add_argument("--grad-clip", type=float, default=1.0)
    argument_parser.add_argument("--eval-interval", type=int, default=100)
    argument_parser.add_argument("--eval-iters", type=int, default=20)
    argument_parser.add_argument("--log-interval", type=int, default=10)
    argument_parser.add_argument("--checkpoint-interval", type=int, default=200)
    argument_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    argument_parser.add_argument("--no-compile", action="store_true")
    argument_parser.add_argument("--force-compile", action="store_true")
    argument_parser.add_argument("--seed", type=int, default=1337)
    args = argument_parser.parse_args()

    distributed_context = setup_distributed_training(args.device)
    device = distributed_context.device
    is_main = distributed_context.is_main_process
    torch.manual_seed(args.seed + distributed_context.global_rank)
    random_state = np.random.RandomState(args.seed + distributed_context.global_rank)  # different data order per process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if is_main:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    train_input_ids, train_labels = sft_data.load_split(args.data_dir, "train", verbose=is_main)
    val_input_ids, val_labels = sft_data.load_split(args.data_dir, "val", verbose=is_main)
    splits = {"train": (train_input_ids, train_labels), "val": (val_input_ids, val_labels)}

    if is_main:
        summary = sft_data.describe(args.data_dir, "train")
        print(f"SFT data: {summary['examples']:,} train / {len(val_input_ids):,} val rows, "
              f"sequence_length={summary['sequence_length']}, "
              f"{summary['supervised_fraction']*100:.0f}% of positions supervised")

    checkpoint_to_load = args.resume or args.init_from
    if is_main:
        print(f"loading {'resume' if args.resume else 'base'} checkpoint from {checkpoint_to_load}")
    checkpoint = checkpoint_utils.load_for_training(checkpoint_to_load)
    model_config = ModelConfig(**checkpoint["model_config"])
    if model_config.max_sequence_length != train_input_ids.shape[1]:
        raise SystemExit(
            f"model max_sequence_length ({model_config.max_sequence_length}) != SFT data sequence length "
            f"({train_input_ids.shape[1]}).\n"
            f"Rebuild the SFT data with a block size of {model_config.max_sequence_length}."
        )

    use_torch_compile = not args.no_compile
    if use_torch_compile and platform.system() == "Windows" and not args.force_compile:
        if is_main:
            print("Windows detected: skipping torch.compile (no stable Triton backend). Pass --force-compile to override.")
        use_torch_compile = False

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    autocast_context = torch.autocast(device_type=device_type, dtype=autocast_dtype)

    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint_utils.normalize_state_dict_keys(checkpoint["model"]))
    model.train()  # also releases any inference-time absorbed matrices; see model.py
    optimizer = model.configure_optimizer(args.weight_decay, args.max_lr, (0.9, 0.95), device_type)

    start_step = 0
    best_validation_loss = float("inf")
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        best_validation_loss = checkpoint.get("best_val", float("inf"))
    del checkpoint  # release the CPU-side copy before training allocates its own buffers

    sequences_per_step = args.batch_size * args.grad_accum * distributed_context.world_size
    steps_per_epoch = max(1, len(train_input_ids) // sequences_per_step)
    max_steps = args.max_steps or int(args.epochs * steps_per_epoch)

    if is_main:
        print(f"model: {count_total_parameters(model_config)/1e6:.1f}M total / "
              f"{count_active_parameters_per_token(model_config)/1e6:.1f}M active parameters, "
              f"context {model_config.max_sequence_length}")
        print(f"effective batch: {args.batch_size} micro x {args.grad_accum} accumulation x "
              f"{distributed_context.world_size} gpu = {sequences_per_step} sequences/step "
              f"({sequences_per_step * model_config.max_sequence_length:,} tokens/step)")
        print(f"training for up to {max_steps} steps (~{max_steps/steps_per_epoch:.2f} epochs) "
              f"or {args.minutes:.0f} minutes, whichever comes first; {args.lr_schedule}-based cosine")

    underlying_model = model  # un-wrapped reference for saving and evaluation
    if distributed_context.is_distributed:
        model = DistributedDataParallel(model, device_ids=[distributed_context.local_rank] if device_type == "cuda" else None)
    if use_torch_compile:
        if is_main:
            print("compiling model (torch.compile) and probing one step...")
        compiled_model = torch.compile(model)
        probe_started = time.time()
        try:
            probe_inputs, probe_labels = load_random_batch(train_input_ids, train_labels, args.batch_size, device, random_state)
            with autocast_context:
                _, probe_loss = compiled_model(probe_inputs, probe_labels)
            probe_loss.backward()
            model = compiled_model
            if is_main:
                print(f"  torch.compile OK in {time.time() - probe_started:.0f}s (probe loss {probe_loss.item():.4f})")
        except Exception as error:
            if is_main:
                print(f"  torch.compile failed ({type(error).__name__}: {str(error)[:400]}); continuing in eager mode")
        finally:
            optimizer.zero_grad(set_to_none=True)

    last_checkpoint_path = os.path.join(args.ckpt_dir, f"{args.tag}_last.pt")
    best_checkpoint_path = os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")

    def save_checkpoint(path, step):
        checkpoint_utils.save_checkpoint_atomically({
            "model": underlying_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_validation_loss,
            "model_config": dataclasses.asdict(model_config),
        }, path)

    def append_log(entry, suffix="train_log"):
        with open(os.path.join(args.ckpt_dir, f"{args.tag}_{suffix}.jsonl"), "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")

    def evaluate_and_maybe_save_best(step, label):
        nonlocal best_validation_loss
        losses = estimate_loss(underlying_model, splits, args.batch_size, device, autocast_context,
                                args.eval_iters, random_state)
        print(f"  {label} @ step {step}: train {losses['train']:.4f}  val {losses['val']:.4f}")
        append_log({"step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                    "elapsed_m": (time.time() - training_start_time) / 60}, suffix="val_log")
        if losses["val"] < best_validation_loss:
            best_validation_loss = losses["val"]
            save_checkpoint(best_checkpoint_path, step)
            print(f"    new best val loss -> {os.path.basename(best_checkpoint_path)}")

    training_start_time = time.time()  # after compilation, so compile time never eats into the schedule
    deadline = training_start_time + args.minutes * 60
    warmup_end_time = None
    step = start_step
    last_log_time = time.time()

    while step < max_steps:
        if synchronize_stop_decision(distributed_context, time.time() >= deadline):
            if is_main:
                print(f"\nhit {args.minutes:.0f}-minute wall-clock budget at step {step}, stopping")
            break

        now = time.time()
        if args.lr_schedule == "time" and step >= args.warmup_steps:
            if warmup_end_time is None:
                warmup_end_time = now
            time_progress = (now - warmup_end_time) / max(1e-6, deadline - warmup_end_time)
            step_progress = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
            current_learning_rate = cosine_between(max(time_progress, step_progress), args.max_lr, args.min_lr)
        else:
            current_learning_rate = learning_rate_at_step(step, args.warmup_steps, max_steps, args.max_lr, args.min_lr)
        current_learning_rate = synchronize_float(distributed_context, current_learning_rate)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        for micro_step in range(args.grad_accum):
            input_ids, labels = load_random_batch(train_input_ids, train_labels, args.batch_size, device, random_state)
            if distributed_context.is_distributed:
                # only all-reduce gradients on the final micro-step (the standard DDP +
                # gradient-accumulation trick); syncing every micro-step just wastes bandwidth
                model.require_backward_grad_sync = (micro_step == args.grad_accum - 1)
            with autocast_context:
                _, loss = model(input_ids, labels)
                loss = loss / args.grad_accum
            loss.backward()
            accumulated_loss += loss.detach()

        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not math.isfinite(gradient_norm):
            if is_main:
                print(f"  warning: non-finite gradient norm at step {step}; skipping this update")
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue
        optimizer.step()

        if is_main and step % args.log_interval == 0:
            elapsed_since_last_log = time.time() - last_log_time
            last_log_time = time.time()
            steps_since_log = args.log_interval if step > start_step else 1
            tokens_per_second = (sequences_per_step * model_config.max_sequence_length * steps_since_log
                                 / max(elapsed_since_last_log, 1e-6))
            elapsed_minutes = (time.time() - training_start_time) / 60
            loss_value = accumulated_loss.item()
            print(f"step {step:6d} | loss {loss_value:.4f} | lr {current_learning_rate:.2e} | "
                  f"grad_norm {gradient_norm:6.3f} | {tokens_per_second:8.0f} tok/s | {elapsed_minutes:5.1f}m elapsed")
            append_log({"step": step, "loss": loss_value, "lr": current_learning_rate,
                        "grad_norm": gradient_norm, "tok_s": tokens_per_second, "elapsed_m": elapsed_minutes})

        if step > start_step and step % args.eval_interval == 0:
            if is_main:
                evaluate_and_maybe_save_best(step, "eval")
            barrier(distributed_context)

        if step > start_step and step % args.checkpoint_interval == 0:
            if is_main:
                save_checkpoint(last_checkpoint_path, step)
            barrier(distributed_context)

        step += 1

    if is_main:
        evaluate_and_maybe_save_best(step, "final eval")
        save_checkpoint(last_checkpoint_path, step)
        print(f"\nfinal SFT checkpoint saved to {last_checkpoint_path} (best val loss: {best_validation_loss:.4f})")
        print(f"chat with it:  python server.py --checkpoint {os.path.basename(best_checkpoint_path)}")
        print(f"shrink it:     python checkpoint_utils.py strip {best_checkpoint_path}")
    barrier(distributed_context)

    cleanup_distributed_training(distributed_context)


if __name__ == "__main__":
    main()
