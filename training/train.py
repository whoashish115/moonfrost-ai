"""
Pretraining loop: AdamW optimizer + cosine learning-rate decay with warmup +
gradient accumulation + bfloat16 mixed precision + gradient clipping +
periodic, atomic checkpointing.

Runs unmodified on a local GPU for a wall-clock-limited session, or on a
rented multi-GPU cloud box under torchrun for data-parallel training -- point
--data-dir/--tokenizer-dir/--ckpt-dir at persistent storage and it works in
either place. See ddp_utils.py and cloud_run.py.

Examples:
  # local, single GPU, stop after 110 minutes whatever step it has reached
  python train.py --size small --minutes 110

  # resume the SAME run (restores optimizer state and the step counter)
  python train.py --size small --minutes 180 --resume ../checkpoints/small_last.pt

  # start a NEW run from existing weights (fresh optimizer, step 0), with a learning-rate
  # schedule that finishes decaying exactly when the time budget runs out
  python train.py --size small --init-from ../checkpoints/small_upcycled_18.pt --tag v2 \
      --minutes 78 --lr-schedule time

  # cloud, 4 GPUs on one machine, data-parallel
  torchrun --standalone --nproc_per_node=4 train.py --size cloud --minutes 600

--resume vs --init-from
  --resume     continues an interrupted run: same optimizer moments, same step, same schedule.
  --init-from  begins a new run from existing weights. The architecture is read from the
               checkpoint, so it also accepts a grown model (see upcycle.py).

--lr-schedule
  steps  cosine decay over --max-steps (the preset's max_training_steps by default). If the
         time budget ends first, training stops with the learning rate still high, which
         leaves loss on the table.
  time   cosine decay over the WALL-CLOCK budget after warmup, so the learning rate reaches
         its minimum exactly at the deadline however fast the hardware turns out to be. This
         is the right choice whenever the budget is money or hours rather than tokens. If
         --max-steps is also given, whichever limit is nearer drives the decay.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid an MKL/libiomp5 conflict seen on some Anaconda+CUDA installs

import argparse
import dataclasses
import json
import math
import platform
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # otherwise output sits in a buffer when stdout isn't a terminal (e.g. redirected to a log file), so a running job looks stalled

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

import checkpoint_utils
from config import (MODEL_PRESETS, TRAIN_PRESETS, ModelConfig, count_active_parameters_per_token,
                    count_total_parameters)
from model import GPT
from ddp_utils import (cleanup_distributed_training, setup_distributed_training, synchronize_float,
                       synchronize_stop_decision)


def load_random_batch(binary_file_path, sequence_length, batch_size, device):
    """Loads a random batch of (input, target) sequences from a
    pre-tokenized .bin file. The file is memory-mapped rather than fully
    loaded into RAM, so this works even for files much bigger than
    available memory -- the same trick nanoGPT uses."""
    token_ids = np.memmap(binary_file_path, dtype=np.uint16, mode="r")
    start_indices = np.random.randint(0, len(token_ids) - sequence_length - 1, size=(batch_size,))
    input_sequences = torch.stack([torch.from_numpy(token_ids[i:i + sequence_length].astype(np.int64)) for i in start_indices])
    # the target for each position is simply the NEXT token -- this is what makes it "next-token prediction"
    target_sequences = torch.stack([torch.from_numpy(token_ids[i + 1:i + 1 + sequence_length].astype(np.int64)) for i in start_indices])
    if "cuda" in device:
        input_sequences = input_sequences.pin_memory().to(device, non_blocking=True)
        target_sequences = target_sequences.pin_memory().to(device, non_blocking=True)
    else:
        input_sequences = input_sequences.to(device)
        target_sequences = target_sequences.to(device)
    return input_sequences, target_sequences


def barrier(distributed_context):
    """Makes every process wait here. Needed around any work only rank 0 does
    (evaluation, checkpointing) so the ranks stay in lockstep."""
    if distributed_context.is_distributed:
        torch.distributed.barrier()


def cosine_between(progress, max_learning_rate, min_learning_rate):
    progress = min(max(progress, 0.0), 1.0)
    return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)


def learning_rate_at_step(step, train_config):
    """Cosine decay schedule with linear warmup: ramps up from 0 to
    max_learning_rate over warmup_steps, then follows a cosine curve down to
    min_learning_rate by max_training_steps."""
    if step < train_config.warmup_steps:
        return train_config.max_learning_rate * (step + 1) / train_config.warmup_steps
    if step >= train_config.max_training_steps:
        return train_config.min_learning_rate
    progress_through_decay = (step - train_config.warmup_steps) / max(1, train_config.max_training_steps - train_config.warmup_steps)
    return cosine_between(progress_through_decay, train_config.max_learning_rate, train_config.min_learning_rate)


@torch.no_grad()
def estimate_loss(model, data_directory, model_config, train_config, device, autocast_context):
    model.eval()
    average_losses = {}
    for split_name in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iterations)
        for i in range(train_config.eval_iterations):
            input_ids, target_ids = load_random_batch(
                os.path.join(data_directory, f"{split_name}.bin"), model_config.max_sequence_length,
                train_config.micro_batch_size, device,
            )
            with autocast_context:
                _, loss = model(input_ids, target_ids)
            losses[i] = loss.item()
        average_losses[split_name] = losses.mean().item()
    model.train()
    return average_losses


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--size", choices=list(MODEL_PRESETS), default="small",
                                  help="training preset (batch size, learning rate...) and, for a fresh run, the architecture")
    argument_parser.add_argument("--data-dir", type=str, default="../data")
    argument_parser.add_argument("--tokenizer-dir", type=str, default="../tokenizer",
                                  help="used only to read the tokenizer's actual vocabulary size, so a FRESH model's "
                                       "embedding table always matches it exactly")
    argument_parser.add_argument("--ckpt-dir", type=str, default="../checkpoints")
    argument_parser.add_argument("--tag", type=str, default=None,
                                  help="checkpoint/log filename prefix: {tag}_last.pt, {tag}_best.pt (default: --size)")
    argument_parser.add_argument("--minutes", type=float, default=115.0, help="wall-clock training budget; stop cleanly at this limit")
    argument_parser.add_argument("--max-steps", type=int, default=None, help="override the preset's max_training_steps")
    argument_parser.add_argument("--resume", type=str, default=None, help="continue an interrupted run from this checkpoint")
    argument_parser.add_argument("--init-from", type=str, default=None, help="start a new run from these weights (fresh optimizer)")
    argument_parser.add_argument("--lr-schedule", choices=["steps", "time"], default="steps")
    argument_parser.add_argument("--schedule-start", type=float, default=0.0,
                                  help="where on the overall cosine curve this run begins (0-1). A run split across "
                                       "two machines uses 0->0.52 and then 0.52->1.0, so the learning rate follows "
                                       "ONE curve across both instead of annealing to zero halfway through.")
    argument_parser.add_argument("--schedule-end", type=float, default=1.0,
                                  help="where on the overall cosine curve this run ends (0-1)")
    argument_parser.add_argument("--micro-batch-size", type=int, default=None, help="override the preset's per-GPU micro batch")
    argument_parser.add_argument("--grad-accum", type=int, default=None, help="override the preset's gradient accumulation steps")
    argument_parser.add_argument("--max-lr", type=float, default=None)
    argument_parser.add_argument("--min-lr", type=float, default=None)
    argument_parser.add_argument("--warmup-steps", type=int, default=None)
    argument_parser.add_argument("--log-interval", type=int, default=None)
    argument_parser.add_argument("--eval-interval", type=int, default=None)
    argument_parser.add_argument("--eval-iters", type=int, default=None)
    argument_parser.add_argument("--checkpoint-interval", type=int, default=None)
    argument_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    argument_parser.add_argument("--no-compile", action="store_true")
    argument_parser.add_argument("--force-compile", action="store_true",
                                  help="try torch.compile even on Windows (needs a working Triton install)")
    argument_parser.add_argument("--seed", type=int, default=1337)
    args = argument_parser.parse_args()

    if args.resume and args.init_from:
        argument_parser.error("use --resume (continue a run) or --init-from (new run from weights), not both")

    distributed_context = setup_distributed_training(args.device)
    device = distributed_context.device
    is_main = distributed_context.is_main_process
    torch.manual_seed(args.seed + distributed_context.global_rank)
    np.random.seed(args.seed + distributed_context.global_rank)  # load_random_batch draws from numpy's global RNG
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    tag = args.tag or args.size
    if is_main:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    checkpoint = None
    source_path = args.resume or args.init_from
    if source_path:
        if not os.path.exists(source_path):
            raise SystemExit(f"checkpoint not found: {source_path}")
        if is_main:
            print(f"{'resuming from' if args.resume else 'initializing weights from'} {source_path}")
        # to CPU, never straight to the GPU: a training checkpoint also holds AdamW's optimizer
        # state, twice the size of the model. optimizer.load_state_dict() moves what it needs.
        checkpoint = checkpoint_utils.load_for_training(source_path)
        # the architecture must match the saved weights exactly, so it comes from the checkpoint
        model_config = ModelConfig(**checkpoint["model_config"])
    else:
        model_config = MODEL_PRESETS[args.size]
        tokenizer_json_path = os.path.join(args.tokenizer_dir, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            from tokenizers import Tokenizer
            actual_vocabulary_size = Tokenizer.from_file(tokenizer_json_path).get_vocab_size()
            if actual_vocabulary_size != model_config.vocabulary_size:
                if is_main:
                    print(f"tokenizer at {args.tokenizer_dir} has vocabulary_size={actual_vocabulary_size}, "
                          f"overriding preset's default of {model_config.vocabulary_size}")
                model_config = dataclasses.replace(model_config, vocabulary_size=actual_vocabulary_size)

    overrides = {}
    for argument_value, field_name in ((args.micro_batch_size, "micro_batch_size"),
                                        (args.grad_accum, "gradient_accumulation_steps"),
                                        (args.max_lr, "max_learning_rate"), (args.min_lr, "min_learning_rate"),
                                        (args.warmup_steps, "warmup_steps"), (args.eval_interval, "eval_interval_steps"),
                                        (args.eval_iters, "eval_iterations"),
                                        (args.log_interval, "log_interval_steps"),
                                        (args.checkpoint_interval, "checkpoint_interval_steps"),
                                        (args.max_steps, "max_training_steps")):
        if argument_value is not None:
            overrides[field_name] = argument_value
    train_config = dataclasses.replace(TRAIN_PRESETS[args.size], **overrides)
    step_limit_given = args.max_steps is not None
    if args.lr_schedule == "time" and not step_limit_given:
        train_config = dataclasses.replace(train_config, max_training_steps=10**12)  # the clock is the only limit

    use_compile = train_config.use_torch_compile and not args.no_compile
    if use_compile and platform.system() == "Windows" and not args.force_compile:
        # torch.compile's inductor backend needs Triton, which has no stable native-Windows
        # build. Fall back to eager mode automatically (--force-compile overrides).
        if is_main:
            print("Windows detected: skipping torch.compile (no stable Triton backend). Pass --force-compile to override.")
        use_compile = False

    tokens_per_step = (train_config.micro_batch_size * train_config.gradient_accumulation_steps
                       * distributed_context.world_size * model_config.max_sequence_length)
    if is_main:
        print(f"run '{tag}': {count_total_parameters(model_config)/1e6:.1f}M total / "
              f"{count_active_parameters_per_token(model_config)/1e6:.1f}M active parameters, "
              f"{model_config.num_routed_experts} routed experts, context {model_config.max_sequence_length}, "
              f"device={device}, world_size={distributed_context.world_size}")
        print(f"batch: {train_config.micro_batch_size} micro x {train_config.gradient_accumulation_steps} accumulation "
              f"x {distributed_context.world_size} gpu = {tokens_per_step:,} tokens/step; "
              f"lr {train_config.max_learning_rate:.1e} -> {train_config.min_learning_rate:.1e}, "
              f"{train_config.warmup_steps} warmup steps, {args.lr_schedule}-based cosine")

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    autocast_context = torch.autocast(device_type=device_type, dtype=autocast_dtype)

    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizer(
        train_config.weight_decay, train_config.max_learning_rate, (train_config.adam_beta1, train_config.adam_beta2), device_type
    )

    start_step = 0
    best_validation_loss = float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint_utils.normalize_state_dict_keys(checkpoint["model"]))
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"] + 1
            best_validation_loss = checkpoint.get("best_val", float("inf"))
        del checkpoint  # release the CPU copy before training allocates its own buffers
    model.train()

    underlying_model = model  # un-wrapped reference for saving and evaluation, kept before DDP/compile wrapping
    if distributed_context.is_distributed:
        model = DistributedDataParallel(model, device_ids=[distributed_context.local_rank] if device_type == "cuda" else None)

    train_data_path = os.path.join(args.data_dir, "train.bin")
    if use_compile:
        # Probe one real step under torch.compile before committing to it. A compiler failure
        # that surfaces on the first forward or backward then costs one step and a warning,
        # instead of ending a paid run. Evaluation always uses the uncompiled module (below),
        # so switching to eval mode never triggers a second compilation mid-run.
        if is_main:
            print("compiling model (torch.compile) and probing one step...")
        compiled_model = torch.compile(model)
        probe_started = time.time()
        try:
            probe_inputs, probe_targets = load_random_batch(
                train_data_path, model_config.max_sequence_length, train_config.micro_batch_size, device
            )
            with autocast_context:
                _, probe_loss = compiled_model(probe_inputs, probe_targets)
            probe_loss.backward()
            model = compiled_model
            if is_main:
                print(f"  torch.compile OK in {time.time() - probe_started:.0f}s (probe loss {probe_loss.item():.4f})")
        except Exception as error:
            if is_main:
                print(f"  torch.compile failed ({type(error).__name__}: {str(error)[:400]}); continuing in eager mode")
        finally:
            optimizer.zero_grad(set_to_none=True)

    last_checkpoint_path = os.path.join(args.ckpt_dir, f"{tag}_last.pt")
    best_checkpoint_path = os.path.join(args.ckpt_dir, f"{tag}_best.pt")

    def save_checkpoint(path, step):
        # written to a temporary file and renamed into place: interrupting a multi-gigabyte
        # torch.save() used to leave a truncated, unloadable file where the last good
        # checkpoint had been
        checkpoint_utils.save_checkpoint_atomically({
            "model": underlying_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_validation_loss,
            "model_config": dataclasses.asdict(model_config),  # plain dict: safe to load with torch.load(weights_only=True)
        }, path)

    def append_log(entry, suffix="train_log"):
        with open(os.path.join(args.ckpt_dir, f"{tag}_{suffix}.jsonl"), "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")

    def evaluate_and_maybe_save_best(step, label):
        nonlocal best_validation_loss
        losses = estimate_loss(underlying_model, args.data_dir, model_config, train_config, device, autocast_context)
        perplexity = math.exp(min(losses["val"], 20))
        print(f"  {label} @ step {step}: train {losses['train']:.4f}  val {losses['val']:.4f}  "
              f"(val perplexity {perplexity:.1f})")
        append_log({"step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                    "val_perplexity": perplexity, "elapsed_m": (time.time() - training_start_time) / 60},
                   suffix="val_log")
        if losses["val"] < best_validation_loss:
            best_validation_loss = losses["val"]
            save_checkpoint(best_checkpoint_path, step)
            print(f"    new best val loss -> {os.path.basename(best_checkpoint_path)}")

    # the clock starts AFTER compilation, so compile time never eats into the schedule
    training_start_time = time.time()
    deadline = training_start_time + args.minutes * 60
    warmup_end_time = None
    step = start_step
    last_log_time = time.time()
    warmup_start_step = start_step

    if is_main:
        limit = "the clock" if not step_limit_given and args.lr_schedule == "time" else f"{train_config.max_training_steps} steps"
        print(f"training until {args.minutes:.0f} minutes or {limit}, whichever comes first")

    while step < train_config.max_training_steps:
        if synchronize_stop_decision(distributed_context, time.time() >= deadline):
            if is_main:
                print(f"\nhit {args.minutes:.0f}-minute wall-clock budget at step {step}, stopping")
            break

        now = time.time()
        steps_into_run = step - warmup_start_step
        if args.lr_schedule == "time":
            if warmup_end_time is None and steps_into_run >= train_config.warmup_steps:
                warmup_end_time = now
            local_progress = 0.0 if warmup_end_time is None else (
                (now - warmup_end_time) / max(1e-6, deadline - warmup_end_time))
            if step_limit_given:
                remaining = max(1, train_config.max_training_steps - warmup_start_step - train_config.warmup_steps)
                local_progress = max(local_progress, (steps_into_run - train_config.warmup_steps) / remaining)
            # map this run's own 0..1 progress onto its slice of the overall schedule
            span = args.schedule_start + (args.schedule_end - args.schedule_start) * min(max(local_progress, 0.0), 1.0)
            current_learning_rate = cosine_between(span, train_config.max_learning_rate, train_config.min_learning_rate)
            if steps_into_run < train_config.warmup_steps:
                # ramp up to wherever the curve currently sits, not to the peak: a continuation
                # run must re-warm to its own point on the schedule, not jump back to the start
                current_learning_rate *= (steps_into_run + 1) / train_config.warmup_steps
        else:
            current_learning_rate = learning_rate_at_step(step, train_config)
        # every rank must apply the same learning rate, and clocks drift between processes
        current_learning_rate = synchronize_float(distributed_context, current_learning_rate)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate

        optimizer.zero_grad(set_to_none=True)
        # accumulated on the device: a .item() per micro-step would force a GPU->CPU sync each time
        accumulated_loss = torch.zeros((), device=device)
        for micro_step in range(train_config.gradient_accumulation_steps):
            input_ids, target_ids = load_random_batch(
                train_data_path, model_config.max_sequence_length, train_config.micro_batch_size, device,
            )
            if distributed_context.is_distributed:
                # only synchronize gradients across processes on the LAST accumulation micro-step
                # (the standard DistributedDataParallel + gradient-accumulation trick)
                model.require_backward_grad_sync = (micro_step == train_config.gradient_accumulation_steps - 1)
            with autocast_context:
                _, loss = model(input_ids, target_ids)
                loss = loss / train_config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.detach()

        # clip_grad_norm_ returns the norm BEFORE clipping: a spiking or non-finite value here is
        # the earliest visible sign that training is diverging, well before the loss shows it
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm))
        if not math.isfinite(gradient_norm):
            if is_main:
                print(f"  warning: non-finite gradient norm at step {step}; skipping this update")
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue
        optimizer.step()

        if is_main and step % train_config.log_interval_steps == 0:
            elapsed_since_last_log = time.time() - last_log_time
            last_log_time = time.time()
            steps_since_log = train_config.log_interval_steps if step > start_step else 1
            tokens_per_second = tokens_per_step * steps_since_log / max(elapsed_since_last_log, 1e-6)
            elapsed_minutes = (time.time() - training_start_time) / 60
            loss_value = accumulated_loss.item()
            print(f"step {step:6d} | loss {loss_value:.4f} | lr {current_learning_rate:.2e} | "
                  f"grad_norm {gradient_norm:6.3f} | {tokens_per_second:8.0f} tok/s | {elapsed_minutes:5.1f}m elapsed")
            append_log({"step": step, "loss": loss_value, "lr": current_learning_rate, "grad_norm": gradient_norm,
                        "tok_s": tokens_per_second, "elapsed_m": elapsed_minutes})

        if step > start_step and step % train_config.eval_interval_steps == 0:
            # Only rank 0 evaluates and saves, but EVERY rank must stop at the barrier below.
            # Without it the other ranks race ahead into the next step's gradient all-reduce
            # while rank 0 is still evaluating, and DistributedDataParallel deadlocks.
            if is_main:
                evaluate_and_maybe_save_best(step, "eval")
            barrier(distributed_context)

        if step > start_step and step % train_config.checkpoint_interval_steps == 0:
            if is_main:
                save_checkpoint(last_checkpoint_path, step)
            barrier(distributed_context)

        step += 1

    if is_main:
        # the last stretch of a decayed schedule is usually the best model of the run, and it
        # would otherwise never be measured or kept as "best"
        evaluate_and_maybe_save_best(step, "final eval")
        save_checkpoint(last_checkpoint_path, step)
        print(f"final checkpoint saved to {last_checkpoint_path} (best val loss: {best_validation_loss:.4f})")
    barrier(distributed_context)

    cleanup_distributed_training(distributed_context)


if __name__ == "__main__":
    main()
