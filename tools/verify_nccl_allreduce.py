#!/usr/bin/env python3
"""Fail fast when a single-node four-rank NCCL all-reduce is unhealthy."""

from __future__ import annotations

import argparse
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--numel", type=int, default=19_991_936)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    if world_size != args.expected_world_size:
        raise SystemExit(
            f"Expected {args.expected_world_size} ranks, found {world_size}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    tensor = torch.full(
        (args.numel,),
        float(rank + 1),
        device=torch.device("cuda", local_rank),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)

    expected = float(world_size * (world_size + 1) // 2)
    if tensor[0].item() != expected or tensor[-1].item() != expected:
        raise RuntimeError("NCCL all-reduce returned an unexpected value")
    if rank == 0:
        print(
            "NCCL preflight OK:",
            f"world_size={world_size}",
            f"numel={args.numel}",
            f"sum={expected}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
