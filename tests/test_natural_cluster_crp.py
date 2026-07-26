import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from helper.synth_table import PersistentSynthState


def _grouped_prototypes(groups=4, per_group=4, dim=16, noise=0.05, seed=0):
    """(D, C) prototypes with `groups` well-separated direction bundles."""
    generator = torch.Generator().manual_seed(seed)
    columns = []
    for g in range(groups):
        base = torch.zeros(dim)
        base[g] = 1.0
        for _ in range(per_group):
            columns.append(base + noise * torch.randn(dim, generator=generator))
    return torch.stack(columns, dim=1)


def _cluster_state(num_real, max_cols, cluster_size=4, crp_topk=4):
    return PersistentSynthState(num_real, max_cols, bank_size=2,
                                pair_strategy="crp", crp_alpha=1.0,
                                crp_topk=crp_topk, candidate_pool="cluster",
                                cluster_size=cluster_size)


class NaturalClusterPoolTest(unittest.TestCase):
    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            PersistentSynthState(8, 8, candidate_pool="cluster", pair_strategy="fixed_nn")
        with self.assertRaises(ValueError):
            PersistentSynthState(8, 8, candidate_pool="nonsense", pair_strategy="crp")
        with self.assertRaises(ValueError):
            PersistentSynthState(8, 8, candidate_pool="cluster", pair_strategy="crp",
                                 cluster_size=1)

    def test_deterministic_across_fresh_states(self):
        W = _grouped_prototypes()
        first = _cluster_state(W.size(1), 32)
        second = _cluster_state(W.size(1), 32)
        first.rebuild_pairing(W)
        second.rebuild_pairing(W.clone())

        self.assertEqual(first.cluster_assign, second.cluster_assign)
        self.assertEqual(dict(first.candidate_pairs), dict(second.candidate_pairs))
        self.assertEqual(first.spk_partner, second.spk_partner)

    def test_candidates_stay_within_assigned_cluster(self):
        W = _grouped_prototypes()
        state = _cluster_state(W.size(1), 64, cluster_size=4, crp_topk=3)
        state.rebuild_pairing(W)

        assign = state.cluster_assign
        self.assertEqual(len(assign), W.size(1))
        for s in range(W.size(1)):
            candidates = state.candidate_pairs[s]
            self.assertGreaterEqual(len(candidates), 1)
            self.assertLessEqual(len(candidates), 3)
            mates = [j for j in range(W.size(1))
                     if j != s and assign[j] == assign[s]]
            for key in candidates:
                partner = state._other(key, s)
                if mates:
                    self.assertEqual(assign[partner], assign[s])
                else:                      # singleton cluster -> global NN fallback
                    self.assertEqual(partner, state.spk_partner[s])

    def test_singleton_cluster_falls_back_to_global_nn(self):
        # Three tight groups plus one isolated outlier direction; five clusters
        # (cluster_size=2) deterministically leave the outlier alone under the
        # fixed k-means seed, and it must still get its global-NN candidate.
        W = _grouped_prototypes(groups=3, per_group=3, dim=16, noise=0.02)
        outlier = torch.zeros(16)
        outlier[10] = 1.0
        W = torch.cat([W, outlier.unsqueeze(1)], dim=1)          # C = 10
        state = PersistentSynthState(10, 64, pair_strategy="crp",
                                     candidate_pool="cluster", cluster_size=2,
                                     crp_topk=2)
        state.rebuild_pairing(W)

        assign = state.cluster_assign
        outlier_id = 9
        mates = [j for j in range(10) if j != outlier_id and assign[j] == assign[outlier_id]]
        self.assertEqual(mates, [])          # isolated as intended
        keys = state.candidate_pairs[outlier_id]
        self.assertEqual(len(keys), 1)
        self.assertEqual(state._other(keys[0], outlier_id),
                         state.spk_partner[outlier_id])
        self.assertTrue(all(len(state.candidate_pairs[s]) >= 1 for s in range(10)))

    def test_select_pair_and_saturation_fallback(self):
        # Mirrors _gen_persistent: anchors whose pair has no free column are
        # skipped (col is None); training stays valid through saturation.
        random.seed(0)
        W = _grouped_prototypes()
        state = _cluster_state(W.size(1), max_cols=6, cluster_size=4, crp_topk=3)
        events, successes = set(), 0
        for _ in range(2):                   # second epoch reuses assigned pairs
            state.rebuild_pairing(W)
            state.begin_batch_stats(64)
            for step in range(64):
                key, col, partner, event = state.select_pair(step % W.size(1))
                if col is None:
                    continue
                self.assertLess(col, 6)
                self.assertIsNotNone(partner)
                events.add(event)
                successes += 1
                state.commit_visit(key, col, event, "batch")
            state.finalize_batch_stats()

        self.assertLessEqual(len(state.pair_col), 6)
        self.assertGreaterEqual(successes, 12)
        self.assertIn("reuse", events)      # saturated table must keep reusing

    def test_columns_persist_across_cluster_drift(self):
        W = _grouped_prototypes()
        state = _cluster_state(W.size(1), 64)
        state.rebuild_pairing(W)
        random.seed(1)
        state.begin_batch_stats(16)
        for s in range(16):
            key, col, _, event = state.select_pair(s)
            state.commit_visit(key, col, event, "batch")
        state.finalize_batch_stats()
        before = dict(state.pair_col)

        drifted = W + 0.10 * torch.randn(W.shape, generator=torch.Generator().manual_seed(7))
        state.rebuild_pairing(drifted)
        for key, col in before.items():
            self.assertEqual(state.pair_col[key], col)

    def test_state_dict_round_trip(self):
        W = _grouped_prototypes()
        state = _cluster_state(W.size(1), 64)
        state.rebuild_pairing(W)
        payload = state.state_dict()

        restored = _cluster_state(W.size(1), 64)
        restored.load_state_dict(payload)
        self.assertEqual(restored.candidate_pool, "cluster")
        self.assertEqual(restored.cluster_size, 4)
        self.assertEqual(restored.cluster_assign, state.cluster_assign)
        self.assertEqual(restored.pair_col, state.pair_col)

    def test_default_topk_pool_unchanged(self):
        W = _grouped_prototypes()
        state = PersistentSynthState(W.size(1), 64, pair_strategy="crp",
                                     crp_alpha=1.0, crp_topk=4)
        state.rebuild_pairing(W)
        self.assertEqual(state.candidate_pool, "topk")
        self.assertEqual(state.cluster_assign, [])
        for s in range(W.size(1)):
            self.assertEqual(len(state.candidate_pairs[s]), 4)


if __name__ == "__main__":
    unittest.main()
