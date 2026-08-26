import unittest

import torch

from criterion.amsoftmax_mix_gan import amsoftmax_gan
from helper.synth_table import PersistentSynthState, slerp


def _grouped_prototypes(groups=4, per_group=4, dim=16, noise=0.05, seed=0):
    generator = torch.Generator().manual_seed(seed)
    columns = []
    for group in range(groups):
        center = torch.zeros(dim)
        center[group] = 1.0
        for _ in range(per_group):
            columns.append(
                center + noise * torch.randn(dim, generator=generator)
            )
    return torch.stack(columns, dim=1)


class NaturalClusterCrpTest(unittest.TestCase):
    @staticmethod
    def _state(num_real, max_cols=64, cluster_size=4, topk=3):
        return PersistentSynthState(
            num_real=num_real,
            max_cols=max_cols,
            pair_strategy="crp",
            crp_topk=topk,
            candidate_pool="cluster",
            cluster_size=cluster_size,
        )

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            PersistentSynthState(
                8, 16, pair_strategy="fixed_nn", candidate_pool="cluster"
            )
        with self.assertRaises(ValueError):
            PersistentSynthState(
                8, 16, pair_strategy="crp", candidate_pool="unknown"
            )
        with self.assertRaises(ValueError):
            PersistentSynthState(
                8, 16, pair_strategy="crp",
                candidate_pool="cluster", cluster_size=1,
            )

    def test_clustering_and_candidates_are_deterministic(self):
        weights = _grouped_prototypes()
        first = self._state(weights.size(1))
        second = self._state(weights.size(1))

        first.rebuild_pairing(weights)
        second.rebuild_pairing(weights.clone())

        self.assertEqual(first.cluster_assign, second.cluster_assign)
        self.assertEqual(
            dict(first.candidate_pairs), dict(second.candidate_pairs)
        )

    def test_candidates_stay_inside_cluster(self):
        weights = _grouped_prototypes()
        state = self._state(weights.size(1))
        state.rebuild_pairing(weights)

        for speaker, keys in state.candidate_pairs.items():
            for key in keys:
                partner = state._other(key, speaker)
                same_cluster = (
                    state.cluster_assign[partner]
                    == state.cluster_assign[speaker]
                )
                cluster_members = [
                    index for index, cluster in enumerate(state.cluster_assign)
                    if cluster == state.cluster_assign[speaker]
                    and index != speaker
                ]
                self.assertTrue(same_cluster or not cluster_members)

    def test_default_topk_behavior_is_unchanged(self):
        weights = _grouped_prototypes()
        state = PersistentSynthState(
            weights.size(1), 64, pair_strategy="crp", crp_topk=4
        )
        state.rebuild_pairing(weights)

        self.assertEqual(state.candidate_pool, "topk")
        self.assertEqual(state.cluster_assign, [])
        self.assertTrue(
            all(len(keys) == 4 for keys in state.candidate_pairs.values())
        )

    def test_cluster_state_round_trip(self):
        weights = _grouped_prototypes()
        state = self._state(weights.size(1))
        state.rebuild_pairing(weights)
        restored = self._state(weights.size(1))
        restored.load_state_dict(state.state_dict())

        self.assertEqual(restored.candidate_pool, "cluster")
        self.assertEqual(restored.cluster_size, 4)
        self.assertEqual(restored.cluster_assign, state.cluster_assign)

    def test_cluster_reservations_receive_slerp_initialization(self):
        criterion = amsoftmax_gan(
            embedding_dim=16,
            num_classes=16,
            persistence=True,
            synth_max_factor=4,
            pair_strategy="crp",
            crp_topk=3,
            candidate_pool="cluster",
            cluster_size=4,
            synth_init="slerp_parents",
        )
        with torch.no_grad():
            criterion.W.copy_(_grouped_prototypes())

        reserved = criterion.rebuild_persistent_pairing()
        self.assertGreater(len(reserved), 0)
        for key, column in reserved:
            first, second = criterion.synth._members(key)
            expected = slerp(
                criterion.W[:, first],
                criterion.W[:, second],
                criterion.slerp_t,
            )
            self.assertTrue(torch.allclose(
                criterion.W_syn[:, column], expected, atol=1e-6, rtol=1e-6
            ))

    def test_one_to_one_capacity_assigns_every_anchor(self):
        weights = _grouped_prototypes()
        num_real = weights.size(1)
        state = self._state(
            num_real=num_real,
            max_cols=num_real,
            cluster_size=4,
            topk=3,
        )

        state.rebuild_pairing(weights)

        self.assertEqual(len(state.pair_col), num_real)
        for speaker in range(num_real):
            assigned = [
                key for key in state.candidate_pairs[speaker]
                if key in state.pair_col
            ]
            self.assertTrue(assigned, f"speaker {speaker} has no assigned pair")
            key, column, partner, event = state.select_pair(speaker)
            self.assertIsNotNone(key)
            self.assertIsNotNone(column)
            self.assertIsNotNone(partner)
            self.assertIn(event, {"new", "reuse"})

    def test_one_to_one_capacity_preserves_coverage_after_rebuild(self):
        weights = _grouped_prototypes()
        num_real = weights.size(1)
        state = self._state(
            num_real=num_real,
            max_cols=num_real,
            cluster_size=4,
            topk=3,
        )
        state.rebuild_pairing(weights)

        ordered = sorted(state.pair_col, key=state._serialise_key)
        state.created_pairs = set(ordered[::2])
        persistent_columns = {
            key: state.pair_col[key] for key in state.created_pairs
        }

        state.rebuild_pairing(weights)

        for key, column in persistent_columns.items():
            self.assertEqual(state.pair_col[key], column)
        for speaker in range(num_real):
            self.assertTrue(any(
                key in state.pair_col
                for key in state.candidate_pairs[speaker]
            ), f"speaker {speaker} lost coverage after rebuild")

    def test_one_to_one_capacity_generates_on_first_batch(self):
        num_real = 16
        criterion = amsoftmax_gan(
            embedding_dim=16,
            num_classes=num_real,
            persistence=True,
            synth_max_factor=1,
            pair_strategy="crp",
            crp_topk=3,
            candidate_pool="cluster",
            cluster_size=4,
            synth_init="slerp_parents",
        )
        with torch.no_grad():
            criterion.W.copy_(_grouped_prototypes())
        criterion.rebuild_persistent_pairing()

        embeddings = torch.randn(num_real, 16)
        labels = torch.arange(num_real)
        synthetic, columns, _ = criterion._gen_persistent(
            embeddings, labels, update_state=False
        )

        self.assertGreater(synthetic.size(0), 0)
        self.assertEqual(synthetic.size(0), len(columns))
        self.assertTrue(all(0 <= column < num_real for column in columns))


if __name__ == "__main__":
    unittest.main()
