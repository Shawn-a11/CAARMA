import unittest

import torch

from helper.mixup_avg import mixup_data_euc_avg


class MixupAverageTest(unittest.TestCase):
    def test_uses_true_self_excluded_nearest_neighbor(self):
        embeddings = torch.tensor([[0.0], [10.0], [100.0]])
        weights = torch.tensor([[0.0, 10.0, 1.0]], requires_grad=True)
        labels = torch.tensor([0, 1, 2])

        synthetic, synthetic_labels, synthetic_weights = mixup_data_euc_avg(
            embeddings, weights, labels
        )

        torch.testing.assert_close(
            synthetic, torch.tensor([[50.0], [55.0], [50.0]])
        )
        torch.testing.assert_close(synthetic_labels, torch.tensor([0, 1, 0]))
        torch.testing.assert_close(
            synthetic_weights, torch.tensor([[0.5, 5.5]])
        )

    def test_deduplicates_unordered_pairs_without_decimal_collisions(self):
        embeddings = torch.arange(4.0).unsqueeze(1)
        weights = torch.full((1, 24), 100.0, requires_grad=True)
        with torch.no_grad():
            weights[0, 1] = 0.0
            weights[0, 23] = 0.1
            weights[0, 12] = 10.0
            weights[0, 3] = 10.1
        labels = torch.tensor([1, 23, 12, 3])

        _, synthetic_labels, synthetic_weights = mixup_data_euc_avg(
            embeddings, weights, labels
        )

        self.assertEqual(synthetic_weights.shape[1], 2)
        self.assertEqual(synthetic_labels[0].item(), synthetic_labels[1].item())
        self.assertEqual(synthetic_labels[2].item(), synthetic_labels[3].item())
        self.assertNotEqual(synthetic_labels[0].item(), synthetic_labels[2].item())

        synthetic_weights.sum().backward()
        self.assertIsNotNone(weights.grad)
        self.assertGreater(weights.grad.abs().sum().item(), 0.0)

    def test_requires_two_speakers(self):
        with self.assertRaisesRegex(ValueError, "at least two speakers"):
            mixup_data_euc_avg(
                torch.randn(2, 3),
                torch.randn(3, 1),
                torch.zeros(2, dtype=torch.long),
            )


if __name__ == '__main__':
    unittest.main()
