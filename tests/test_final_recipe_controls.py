import unittest

import torch

from criterion.amsoftmax_mix_gan import amsoftmax_gan
from model.discriminator_mix import (
    ConcatConditionDiscriminator_spectral,
    ProjectionDiscriminator_spectral,
    QOnlyConditionDiscriminator_spectral,
)


class FinalRecipeDiscriminatorControlsTest(unittest.TestCase):
    def test_concat_matches_projection_parameter_count(self):
        projection = ProjectionDiscriminator_spectral(192)
        concat = ConcatConditionDiscriminator_spectral(192)
        self.assertEqual(
            sum(parameter.numel() for parameter in projection.parameters()),
            sum(parameter.numel() for parameter in concat.parameters()),
        )

    def test_q_only_ignores_embedding(self):
        torch.manual_seed(3)
        discriminator = QOnlyConditionDiscriminator_spectral(16).eval()
        first_embedding = torch.randn(5, 16)
        second_embedding = torch.randn(5, 16)
        condition = torch.randn(5, 16)
        self.assertTrue(torch.allclose(
            discriminator(first_embedding, condition),
            discriminator(second_embedding, condition),
        ))


class MatchedOneShotControlTest(unittest.TestCase):
    def test_topk_one_shot_reuses_selection_without_persistent_identity(self):
        torch.manual_seed(5)
        criterion = amsoftmax_gan(
            embedding_dim=8,
            num_classes=6,
            persistence=False,
            one_shot_pairing="topk",
            pair_strategy="crp",
            crp_topk=2,
            synth_bank_size=2,
            synth_max_factor=4,
            synth_init="slerp_parents",
        )
        criterion.rebuild_persistent_pairing()
        self.assertFalse(hasattr(criterion, "W_syn"))

        bank_embeddings = torch.randn(6, 8)
        bank_labels = torch.arange(6)
        criterion.oneshot_state.update_bank(bank_embeddings, bank_labels)

        embeddings = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])
        _, _, first_synthetic = criterion(
            embeddings, labels, cache_selection=True
        )
        cached_selection = list(criterion._cached_selection)
        for batch_index, key, _, _ in cached_selection:
            anchor = int(labels[batch_index])
            self.assertIn(key, criterion.oneshot_state.candidate_pairs[anchor])

        _, _, second_synthetic = criterion(
            embeddings,
            labels,
            update_state=True,
            reuse_selection=True,
        )
        self.assertTrue(torch.allclose(first_synthetic, second_synthetic))
        self.assertEqual(
            criterion.last_synth_conditions.size(0),
            second_synthetic.size(0),
        )

        synth_loss, _, cached_synthetic = criterion(
            embeddings, labels, flagSyn=True
        )
        self.assertTrue(torch.isfinite(synth_loss))
        self.assertTrue(torch.allclose(second_synthetic, cached_synthetic))


if __name__ == "__main__":
    unittest.main()
