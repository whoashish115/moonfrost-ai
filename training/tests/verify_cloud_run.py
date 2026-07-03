"""
Validates the cloud training plan WITHOUT starting anything or spending a
cent. Run this before every `modal run`.

    python verify_cloud_run.py

It exists because of a real near-miss: cloud_run.py's preflight passed
`--log-interval` to train.py, which had no such flag. Nothing local would
have caught it, and the failure would have surfaced as an instant crash on a
rented H100 after the container had already been paid for. Every command
line cloud_run.py intends to run is now checked, flag by flag, against the
target script's own --help output.

What it checks:
  1. every flag in every planned command exists in that script's parser
  2. the model preset named in the plan exists, and its sequence length and
     vocabulary match what the data pipeline will build
  3. the two learning-rate schedule spans join up and cover 0 -> 1 exactly
  4. each stage's Modal timeout leaves room for its planned training minutes
  5. the pretraining shards for phase 2 do not overlap phase 1
  6. the worst-case bill for each account is under the cap
"""

import os, sys
# these live in training/tests/, so the package directory is one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import subprocess
import sys

# the package directory, one level up from tests/, where the scripts actually live
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def help_text(script):
    completed = subprocess.run([sys.executable, script, "--help"], cwd=HERE, capture_output=True, text=True, timeout=300)
    if completed.returncode != 0:
        return None
    return completed.stdout


def flags_in(arguments):
    return [str(argument) for argument in arguments if str(argument).startswith("--")]


def main():
    print("cloud run plan check -- nothing is launched and nothing is billed\n" + "=" * 64)
    import cloud_run
    from config import MODEL_PRESETS

    print("\n1. every planned command line is valid")
    planned = [
        ("train.py", "preflight", cloud_run.preflight_arguments()),
        ("train.py", "pretrain phase 1", cloud_run.pretrain_arguments(1)),
        ("train.py", "pretrain phase 2", cloud_run.pretrain_arguments(2)),
        ("sft_train.py", "chat tuning", cloud_run.finetune_arguments("/vol/checkpoints/base2_best.pt")),
    ]
    help_cache = {}
    for script, label, arguments in planned:
        if script not in help_cache:
            help_cache[script] = help_text(script)
        text = help_cache[script]
        if text is None:
            check(f"{script} --help works", False, "the script itself fails to start")
            continue
        unknown = [flag for flag in flags_in(arguments) if flag not in text]
        check(f"{label}: {len(flags_in(arguments))} flags accepted by {script}", not unknown,
              f"unknown: {', '.join(unknown)}" if unknown else "")

    print("\n2. the model preset matches the data pipeline")
    preset = MODEL_PRESETS.get(cloud_run.MODEL_SIZE)
    check(f"preset '{cloud_run.MODEL_SIZE}' exists", preset is not None)
    if preset is not None:
        from config import count_active_parameters_per_token, count_total_parameters
        check("chat block size matches the model's context window",
              cloud_run.BLOCK_SIZE == preset.max_sequence_length,
              f"{cloud_run.BLOCK_SIZE} vs {preset.max_sequence_length}")
        check("tokenizer vocabulary matches the model's embedding table",
              cloud_run.VOCAB_SIZE == preset.vocabulary_size,
              f"{cloud_run.VOCAB_SIZE} vs {preset.vocabulary_size}")
        print(f"        model: {count_total_parameters(preset)/1e6:.0f}M total / "
              f"{count_active_parameters_per_token(preset)/1e6:.0f}M active, "
              f"{preset.num_routed_experts} routed experts, top-{preset.num_activated_experts_per_token}")

    print("\n3. the learning-rate schedule is continuous across the two phases")
    def span(phase):
        arguments = cloud_run.pretrain_arguments(phase)
        start = float(arguments[arguments.index("--schedule-start") + 1])
        end = float(arguments[arguments.index("--schedule-end") + 1])
        return start, end
    first, second = span(1), span(2)
    check("phase 1 starts at the beginning of the curve", first[0] == 0.0, f"{first[0]}")
    check("phase 2 starts exactly where phase 1 stopped", abs(first[1] - second[0]) < 1e-9,
          f"{first[1]:.4f} -> {second[0]:.4f}")
    check("phase 2 anneals to the end of the curve", second[1] == 1.0, f"{second[1]}")

    print("\n4. timeouts leave room for the planned training time")
    for phase in (1, 2):
        planned_minutes = cloud_run.PRETRAIN_MINUTES[phase]
        timeout = cloud_run.PRETRAIN_TIMEOUT_MINUTES[phase]
        check(f"pretrain phase {phase} timeout exceeds its budget", timeout >= planned_minutes + 15,
              f"{planned_minutes} min budget, {timeout} min timeout "
              f"({timeout - planned_minutes} min for startup, compile and the final save)")
    check("chat tuning timeout exceeds its budget",
          cloud_run.FINETUNE_TIMEOUT_MINUTES >= cloud_run.FINETUNE_MINUTES + 8,
          f"{cloud_run.FINETUNE_MINUTES} min budget, {cloud_run.FINETUNE_TIMEOUT_MINUTES} min timeout")

    print("\n5. the two phases read different data")
    first_shards, second_shards = set(cloud_run.FINEWEB_SHARDS[1]), set(cloud_run.FINEWEB_SHARDS[2])
    check("phase 2 uses shards phase 1 never saw", not (first_shards & second_shards),
          f"phase 1 {sorted(first_shards)}, phase 2 {sorted(second_shards)}")

    print("\n6. the bill cannot exceed the cap")
    for account, stages in cloud_run.worst_case_by_account().items():
        worst = sum(stages.values())
        cap = cloud_run.ACCOUNT_CAP_USD[account]
        check(f"account {account} worst case under ${cap:.2f}",
              worst <= cap, f"${worst:.2f}, leaving ${cap - worst:.2f} of headroom")

    cloud_run.print_budget()
    print("\n" + "=" * 64)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("\nfix these before running anything:")
        for name in FAILED:
            print(f"  {name}")
        sys.exit(1)
    print("\nplan is consistent. Launch with:  modal run cloud_run.py --stage preflight")


if __name__ == "__main__":
    main()
