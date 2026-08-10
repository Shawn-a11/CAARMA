import unittest

import torch

from criterion.amsoftmax_mix_gan import amsoftmax_gan


def _criterion(ratio):
    criterion = amsoftmax_gan(
        embedding_dim=4,
        num_classes=4,
        persistence=True,
        synth_max_factor=4,
        pair_strategy="crp",
        crp_topk=3,
        reuse_policy="popularity",
        synth_init="slerp_parents",
        synthetic_sample_ratio=ratio,
    )
    with torch.no_grad():
        criterion.W.copy_(torch.eye(4))
    criterion.rebuild_persistent_pairing()
    return criterion


class SyntheticSampleRatioTest(unittest.TestCase):
    def test_half_ratio_generates_half_as_many_synthetic_rows(self):
        criterion = _criterion(0.5)
        embeddings = torch.eye(4)
        labels = torch.arange(4)

        _, _, synthetic = criterion(
            embeddings, labels, update_state=True
        )

        self.assertEqual(synthetic.size(0), 2)
        self.assertEqual(len(criterion.last_synth_cols), 2)

    def test_ratio_above_one_revisits_anchors(self):
        criterion = _criterion(1.5)
        embeddings = torch.eye(4)
        labels = torch.arange(4)

        _, _, synthetic = criterion(
            embeddings, labels, update_state=True
        )

        self.assertEqual(synthetic.size(0), 6)
        self.assertEqual(len(criterion.last_synth_cols), 6)

    def test_unit_ratio_preserves_historical_anchor_order(self):
        criterion = _criterion(1.0)
        self.assertEqual(criterion._sample_anchor_indices(4), [0, 1, 2, 3])

    def test_d_and_m_reuse_the_same_ratio_limited_plan(self):
        criterion = _criterion(0.5)
        embeddings = torch.eye(4)
        labels = torch.arange(4)

        _, _, synth_d = criterion(
            embeddings, labels, cache_selection=True
        )
        cached_plan = list(criterion._cached_selection)
        _, _, synth_m = criterion(
            embeddings, labels, update_state=True, reuse_selection=True
        )

        self.assertEqual(len(cached_plan), 2)
        self.assertTrue(torch.allclose(synth_d, synth_m))
        self.assertIsNone(criterion._cached_selection)

    def test_non_positive_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            _criterion(0.0)


if __name__ == "__main__":
    unittest.main()
