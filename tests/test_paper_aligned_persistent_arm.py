import unittest
from pathlib import Path

import torch

from criterion.build_criterion import build_criterion
from model.discriminator_mix import (
    Discriminator_spectral,
    ProjectionDiscriminator_spectral,
)


ROOT = Path(__file__).resolve().parents[1]


class PaperAlignedPersistentArmTest(unittest.TestCase):
    def _criterion(self, synth_init="xavier"):
        return build_criterion({
            "criterion": "AMSoftmaxGAN",
            "embedding_dim": 8,
            "num_spk": 6,
            "am_margin": 0.2,
            "am_scale": 30,
            "persistence": True,
            "slerp_t": 0.5,
            "synth_bank_size": 2,
            "synth_max_factor": 4,
            "pair_strategy": "crp",
            "crp_alpha": 1.0,
            "crp_topk": 4,
            "reuse_policy": "popularity",
            "candidate_pool": "topk",
            "synth_init": synth_init,
        })

    def test_persistent_crp_registry_is_constructed(self):
        criterion = self._criterion()
        criterion.rebuild_persistent_pairing()
        self.assertTrue(criterion.persistence)
        self.assertEqual(criterion.synth.pair_strategy, "crp")
        self.assertGreater(len(criterion.synth.pair_col), 0)

    def test_parent_slerp_initializes_reserved_prototypes(self):
        criterion = self._criterion("slerp_parents")
        reserved = criterion.rebuild_persistent_pairing()
        self.assertGreater(len(reserved), 0)
        key, col = reserved[0]
        i, j = criterion.synth._members(key)
        expected = criterion.W[:, [i, j]].t()
        prototype = torch.nn.functional.normalize(
            criterion.W_syn[:, col], dim=0
        )
        self.assertGreater(
            float((prototype @ torch.nn.functional.normalize(
                expected.mean(dim=0), dim=0
            )).detach()),
            0.5,
        )

    def test_discriminator_arms_have_expected_interfaces(self):
        embeddings = torch.randn(5, 8)
        conditions = torch.randn(5, 8)
        self.assertEqual(
            Discriminator_spectral(8)(embeddings).shape, (5, 1)
        )
        projection = ProjectionDiscriminator_spectral(8)
        self.assertTrue(projection.requires_condition)
        self.assertEqual(
            projection(embeddings, conditions).shape, (5, 1)
        )

    def test_training_loop_reuses_one_pair_sample_for_d_and_m(self):
        source = (ROOT / "train_paper_aligned.py").read_text()
        for required in (
            "cache_selection=True",
            "update_state=True",
            "reuse_selection=True",
            "synchronize_synth_state",
            "preds[:Ns]",
            "preds[Ns:]",
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
