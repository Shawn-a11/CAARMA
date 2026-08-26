import unittest

import torch

from criterion.amsoftmax_mix_gan import amsoftmax_gan
from helper.synth_table import slerp


def _criterion(synth_init):
    torch.manual_seed(7)
    loss = amsoftmax_gan(
        embedding_dim=8,
        num_classes=6,
        persistence=True,
        synth_max_factor=3,
        pair_strategy="crp",
        crp_alpha=1.0,
        crp_topk=2,
        reuse_policy="popularity",
        synth_init=synth_init,
    )
    with torch.no_grad():
        loss.W.copy_(torch.randn(8, 6, generator=torch.Generator().manual_seed(11)))
    return loss


class SlerpSynthInitializationTest(unittest.TestCase):
    def test_xavier_control_does_not_reinitialise_reserved_columns(self):
        loss = _criterion("xavier")
        before = loss.W_syn.detach().clone()
        reserved = loss.rebuild_persistent_pairing()

        self.assertGreater(len(reserved), 0)
        self.assertTrue(torch.equal(before, loss.W_syn.detach()))

    def test_new_columns_are_parent_slerp_midpoints(self):
        loss = _criterion("slerp_parents")
        reserved = loss.rebuild_persistent_pairing()

        self.assertGreater(len(reserved), 0)
        for key, col in reserved:
            i, j = loss.synth._members(key)
            expected = slerp(loss.W[:, i], loss.W[:, j], loss.slerp_t)
            self.assertTrue(
                torch.allclose(loss.W_syn[:, col], expected, atol=1e-6, rtol=1e-6)
            )

    def test_created_class_is_not_reinitialised_next_epoch(self):
        loss = _criterion("slerp_parents")
        reserved = loss.rebuild_persistent_pairing()
        key, col = reserved[0]

        loss.synth.begin_batch_stats(1)
        loss.synth.commit_visit(key, col, "new", "batch")
        loss.synth.synchronize_pending(loss.W.device)

        sentinel = torch.arange(8, dtype=loss.W.dtype)
        with torch.no_grad():
            loss.W_syn[:, col].copy_(sentinel)
            loss.W.add_(0.25)

        loss.rebuild_persistent_pairing()
        self.assertTrue(torch.equal(loss.W_syn[:, col], sentinel))

    def test_invalid_initialisation_mode_fails_fast(self):
        with self.assertRaises(ValueError):
            _criterion("not-a-mode")


if __name__ == "__main__":
    unittest.main()
