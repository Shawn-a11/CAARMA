"""Two-process CPU smoke test for synchronized Fisher-UCB registry state."""

from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper.synth_table import PersistentSynthState


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    state = PersistentSynthState(
        num_real=3,
        max_cols=6,
        pair_strategy="crp",
        crp_topk=2,
        reuse_policy="fisher_ucb",
    )
    key_a = state._key(0, 1)
    key_b = state._key(0, 2)
    state._ensure_col(key_a)
    state._ensure_col(key_b)
    state._rebuild_pairs_by_speaker()

    key = key_a if rank == 0 else key_b
    col = state.pair_col[key]
    probability = 0.5 if rank == 0 else 0.8
    utility = 4.0 * probability * (1.0 - probability)
    state.begin_batch_stats(batch_size=1)
    state.commit_visit(key, col, "new", "batch")
    state.record_utilities(
        [col], torch.tensor([probability]), torch.tensor([utility])
    )
    state.synchronize_pending(torch.device("cpu"))

    assert state.pair_visits[key_a] == 1
    assert state.pair_visits[key_b] == 1
    assert state.pair_reward_updates[key_a] == 1
    assert state.pair_reward_updates[key_b] == 1
    assert state.total_reward_updates == 2

    summary = (
        sorted((tuple(sorted(key)), value) for key, value in state.pair_visits.items()),
        sorted(
            (tuple(sorted(key)), value)
            for key, value in state.pair_reward_sum.items()
        ),
        state.total_reward_updates,
    )
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, summary)
    assert all(item == gathered[0] for item in gathered)
    if rank == 0:
        print("DDP Fisher-UCB state synchronization: OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
