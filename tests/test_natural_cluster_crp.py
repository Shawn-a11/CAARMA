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
    def _state(num_real, max_cols=64, cluster_size=4, topk=3,
               cluster_candidate_selection="nearest"):
        return PersistentSynthState(
            num_real=num_real,
            max_cols=max_cols,
            pair_strategy="crp",
            crp_topk=topk,
            candidate_pool="cluster",
            cluster_size=cluster_size,
            cluster_candidate_selection=cluster_candidate_selection,
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
        with self.assertRaises(ValueError):
            PersistentSynthState(
                8, 16, pair_strategy="crp",
                candidate_pool="cluster",
                cluster_candidate_selection="unknown",
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

    def test_random_candidates_are_deterministic_and_not_cosine_ranked(self):
        weights = _grouped_prototypes(groups=1, per_group=12)
        nearest = self._state(
            weights.size(1), cluster_size=12, topk=4,
            cluster_candidate_selection="nearest",
        )
        first = self._state(
            weights.size(1), cluster_size=12, topk=4,
            cluster_candidate_selection="random",
        )
        second = self._state(
            weights.size(1), cluster_size=12, topk=4,
            cluster_candidate_selection="random",
        )

        nearest.rebuild_pairing(weights)
        first.rebuild_pairing(weights)
        second.rebuild_pairing(weights.clone())

        self.assertEqual(
            dict(first.candidate_pairs), dict(second.candidate_pairs)
        )
        self.assertTrue(any(
            first.candidate_pairs[speaker]
            != nearest.candidate_pairs[speaker]
            for speaker in range(weights.size(1))
        ))
        self.assertTrue(all(
            len(first.candidate_pairs[speaker]) == 4
            for speaker in range(weights.size(1))
        ))

    def test_all_candidates_replace_topk_with_complete_cluster(self):
        weights = _grouped_prototypes(groups=1, per_group=12)
        state = self._state(
            weights.size(1), cluster_size=12, topk=4,
            cluster_candidate_selection="all",
        )
        state.rebuild_pairing(weights)

        for speaker in range(weights.size(1)):
            expected = {
                other for other in range(weights.size(1))
                if other != speaker
                and state.cluster_assign[other]
                == state.cluster_assign[speaker]
            }
            actual = {
                state._other(key, speaker)
                for key in state.candidate_pairs[speaker]
            }
            self.assertEqual(actual, expected)
            self.assertGreater(len(actual), state.crp_topk)

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
        self.assertEqual(restored.cluster_candidate_selection, "nearest")
        self.assertEqual(restored.cluster_candidate_seed, 1729)
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


if __name__ == "__main__":
    unittest.main()
