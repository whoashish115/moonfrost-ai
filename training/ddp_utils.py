"""
Minimal multi-GPU (DistributedDataParallel, usually abbreviated "DDP")
helpers shared by train.py and sft_train.py, so the same scripts run
unmodified on your local single RTX 3050 AND on a rented multi-GPU cloud
box.

Single GPU / CPU (unchanged from before): just run the script normally.
    python train.py --size small --minutes 115

Multi-GPU on one rented machine (e.g. 4 GPUs): launch with torchrun instead
of python -- torchrun sets the RANK/LOCAL_RANK/WORLD_SIZE environment
variables, which setup_distributed_training() below picks up automatically.
Nothing else about the command changes.
    torchrun --standalone --nproc_per_node=4 train.py --size cloud --minutes 600

See CLOUD_TRAINING.md for renting a GPU box and running this end to end.
"""
import os

import torch
import torch.distributed as dist


class DistributedContext:
    """Describes this process's role in a (possibly multi-process)
    distributed training run. When only one process is running (the normal
    local case), is_distributed is False and every process-count-related
    field is trivially 1/0."""

    def __init__(self, is_distributed, global_rank, local_rank, world_size, device):
        self.is_distributed = is_distributed
        self.global_rank = global_rank    # this process's index among ALL processes across every machine (0-based)
        self.local_rank = local_rank      # this process's index among the processes on just THIS machine (selects which local GPU to use)
        self.world_size = world_size      # total number of processes participating in training
        self.device = device
        self.is_main_process = global_rank == 0  # convention: only the main process logs and saves checkpoints


def setup_distributed_training(requested_device: str) -> DistributedContext:
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ  # torchrun sets these; plain `python` doesn't
    if not is_distributed:
        return DistributedContext(is_distributed=False, global_rank=0, local_rank=0, world_size=1, device=requested_device)

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if global_rank == 0:
        print(f"distributed training: {world_size} process(es), this process's global_rank={global_rank} "
              f"local_rank={local_rank} device={device}")
    return DistributedContext(is_distributed=True, global_rank=global_rank, local_rank=local_rank,
                               world_size=world_size, device=device)


def cleanup_distributed_training(context: DistributedContext):
    if context.is_distributed:
        dist.destroy_process_group()


def synchronize_stop_decision(context: DistributedContext, this_process_wants_to_stop: bool) -> bool:
    """Wall-clock training budgets are checked with each process's own
    time.time(), which can drift slightly between processes (e.g. the main
    process spends extra time on evaluation/checkpointing that the other
    processes don't do). If only the main process broke out of the training
    loop, the other processes would keep calling loss.backward() -- which
    triggers DistributedDataParallel's gradient all-reduce -- with no
    matching call from the main process, hanging forever. Broadcasting the
    main process's stop decision to everyone keeps every process's loop
    iteration count identical, avoiding that hang."""
    if not context.is_distributed:
        return this_process_wants_to_stop
    stop_flag = torch.tensor(
        [1 if (context.is_main_process and this_process_wants_to_stop) else 0],
        device=context.device if "cuda" in context.device else "cpu",
    )
    dist.broadcast(stop_flag, src=0)
    return bool(stop_flag.item())


def synchronize_float(context: DistributedContext, value: float) -> float:
    """Returns rank 0's value on every rank.

    Needed for the time-based learning-rate schedule: each process computes
    the schedule from its own clock, and even a few milliseconds of drift
    gives each rank a slightly different learning rate. With
    DistributedDataParallel every rank then applies the same averaged
    gradient at a different step size, and the model replicas silently stop
    being identical."""
    if not context.is_distributed:
        return value
    value_tensor = torch.tensor([float(value)], dtype=torch.float64,
                                device=context.device if "cuda" in context.device else "cpu")
    dist.broadcast(value_tensor, src=0)
    return float(value_tensor.item())
