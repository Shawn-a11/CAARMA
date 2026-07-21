import unittest

import torch

from criterion.amsoftmax_mix_gan import amsoftmax_gan
from helper.synth_table import PersistentSynthState


class FisherUCBStateTest(unittest.TestCase):
    def _state_with_two_classes(self):
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
        state.created_pairs.update((key_a, key_b))
        return state, key_a, key_b

    def test_ucb_explores_a_less_observed_class(self):
        state, key_a, key_b = self._state_with_two_classes()
        state.pair_visits[key_a] = 10
        state.pair_visits[key_b] = 1
        state.pair_reward_sum[key_a] = 9.0
        state.pair_reward_updates[key_a] = 10
        state.pair_reward_sum[key_b] = 0.2
        state.pair_reward_updates[key_b] = 1
        state.total_reward_updates = 11

        self.assertEqual(state._select_reusable([key_a, key_b]), key_b)

    def test_ucb_exploits_higher_utility_at_equal_counts(self):
        state, key_a, key_b = self._state_with_two_classes()
        state.pair_visits[key_a] = state.pair_visits[key_b] = 5
        state.pair_reward_updates[key_a] = state.pair_reward_updates[key_b] = 5
        state.pair_reward_sum[key_a] = 4.0
        state.pair_reward_sum[key_b] = 1.0
        state.total_reward_updates = 10

        self.assertEqual(state._select_reusable([key_a, key_b]), key_a)

    def test_pending_reward_is_committed_and_checkpointed(self):
        state, key_a, _ = self._state_with_two_classes()
        col = state.pair_col[key_a]
        state.begin_batch_stats(batch_size=4)
        state.commit_visit(key_a, col, "new", "batch")
        state.record_utilities([col], torch.tensor([0.5]), torch.tensor([1.0]))
        stats = state.synchronize_pending(torch.device("cpu"))

        self.assertEqual(state.pair_visits[key_a], 1)
        self.assertEqual(state.pair_reward_updates[key_a], 1)
        self.assertEqual(state.total_reward_updates, 1)
        self.assertAlmostEqual(state.pair_reward_sum[key_a], 1.0)
        self.assertAlmostEqual(state.pair_last_p[key_a], 0.5)
        self.assertAlmostEqual(stats["mean_fisher_ucb_reward"], 1.0)

        restored, _, _ = self._state_with_two_classes()
        restored.load_state_dict(state.state_dict())
        self.assertEqual(restored.reuse_policy, "fisher_ucb")
        self.assertEqual(restored.pair_visits[key_a], 1)
        self.assertEqual(restored.pair_reward_updates[key_a], 1)
        self.assertEqual(restored.total_reward_updates, 1)
        self.assertAlmostEqual(restored.pair_last_p[key_a], 0.5)

    def test_pair_selection_does_not_allocate_columns(self):
        state = PersistentSynthState(
            num_real=4,
            max_cols=16,
            pair_strategy="crp",
            crp_topk=2,
            reuse_policy="fisher_ucb",
        )
        weights = torch.tensor(
            [[1.0, 0.8, 0.0, -1.0], [0.0, 0.2, 1.0, 0.0]]
        )
        state.rebuild_pairing(weights)
        before = dict(state.pair_col)

        for speaker in range(4):
            state.select_pair(speaker)

        self.assertEqual(state.pair_col, before)

    def test_epoch_rebuild_recycles_only_uncreated_reservations(self):
        state = PersistentSynthState(
            num_real=4,
            max_cols=8,
            pair_strategy="crp",
            crp_topk=2,
            reuse_policy="fisher_ucb",
        )
        weights = torch.tensor(
            [[1.0, 0.8, 0.0, -1.0], [0.0, 0.2, 1.0, 0.0]]
        )
        state.rebuild_pairing(weights)
        persistent_key = next(iter(state.pair_col))
        persistent_col = state.pair_col[persistent_key]
        state.created_pairs.add(persistent_key)

        state.rebuild_pairing(weights.roll(shifts=1, dims=1))

        self.assertEqual(state.pair_col[persistent_key], persistent_col)
        self.assertEqual(len(state.col_pair), len(set(state.col_pair)))
        self.assertEqual(len(state.pair_col.values()), len(set(state.pair_col.values())))


class FisherUtilityCriterionTest(unittest.TestCase):
    def test_fisher_utility_peaks_at_the_binary_boundary(self):
        logits = torch.tensor([[0.0, 0.0], [10.0, 0.0], [-10.0, 0.0]])
        target = torch.tensor([0, 0, 0])

        probability, utility = amsoftmax_gan._fisher_boundary_utility(
            logits, target
        )

        self.assertAlmostEqual(probability[0].item(), 0.5, places=6)
        self.assertAlmostEqual(utility[0].item(), 1.0, places=6)
        self.assertLess(utility[1].item(), 0.001)
        self.assertLess(utility[2].item(), 0.001)

    def test_d_and_m_calls_reuse_the_same_pair_plan(self):
        criterion = amsoftmax_gan(
            embedding_dim=3,
            num_classes=3,
            persistence=True,
            synth_max_factor=3,
            pair_strategy="crp",
            crp_topk=2,
            reuse_policy="fisher_ucb",
        )
        with torch.no_grad():
            criterion.W.copy_(torch.eye(3))
        criterion.synth.rebuild_pairing(criterion.W)
        embeddings = torch.eye(3)
        labels = torch.tensor([0, 1, 2])

        _, _, synth_d = criterion(
            embeddings, labels, cache_selection=True
        )
        cached_plan = list(criterion._cached_selection)
        _, _, synth_m = criterion(
            embeddings, labels, update_state=True, reuse_selection=True
        )

        self.assertTrue(cached_plan)
        self.assertIsNone(criterion._cached_selection)
        self.assertTrue(torch.allclose(synth_d, synth_m))

        criterion(embeddings, labels, flagSyn=True)
        stats = criterion.synchronize_synth_state(torch.device("cpu"))
        self.assertGreater(stats["utility_coverage"], 0.0)

    def test_joint_losses_backpropagate_with_utility_tracking(self):
        criterion = amsoftmax_gan(
            embedding_dim=4,
            num_classes=4,
            persistence=True,
            synth_max_factor=3,
            pair_strategy="crp",
            crp_topk=2,
            reuse_policy="fisher_ucb",
        )
        criterion.synth.rebuild_pairing(criterion.W)
        embeddings = torch.randn(4, 4, requires_grad=True)
        labels = torch.arange(4)

        with torch.no_grad():
            criterion(embeddings.detach(), labels, cache_selection=True)
        real_loss, _, _ = criterion(
            embeddings, labels, update_state=True, reuse_selection=True
        )
        synth_loss, _, _ = criterion(embeddings, labels, flagSyn=True)
        criterion.synchronize_synth_state(torch.device("cpu"))
        (real_loss + synth_loss).backward()

        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(torch.isfinite(criterion.W.grad).all())
        self.assertTrue(torch.isfinite(criterion.W_syn.grad).all())


if __name__ == "__main__":
    unittest.main()
