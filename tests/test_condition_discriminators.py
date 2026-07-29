import unittest

import torch

from model.discriminator_mix import (
    ConcatConditionDiscriminator_spectral,
    ProjectionDiscriminator_spectral,
)


class ConditionDiscriminatorTest(unittest.TestCase):
    def test_concat_and_projection_are_parameter_matched(self):
        projection = ProjectionDiscriminator_spectral(192)
        concat = ConcatConditionDiscriminator_spectral(192)

        projection_params = sum(
            parameter.numel() for parameter in projection.parameters()
        )
        concat_params = sum(
            parameter.numel() for parameter in concat.parameters()
        )
        self.assertEqual(projection_params, concat_params)

    def test_concat_forward_shape_and_condition_requirement(self):
        discriminator = ConcatConditionDiscriminator_spectral(16)
        embedding = torch.randn(7, 16)
        condition = torch.randn(7, 16)

        self.assertEqual(
            tuple(discriminator(embedding, condition).shape), (7, 1)
        )
        with self.assertRaises(ValueError):
            discriminator(embedding, None)

    def test_concat_score_depends_on_condition(self):
        torch.manual_seed(7)
        discriminator = ConcatConditionDiscriminator_spectral(16)
        embedding = torch.randn(7, 16)
        first_condition = torch.randn(7, 16)
        second_condition = first_condition.roll(1, dims=0)

        first = discriminator(embedding, first_condition)
        second = discriminator(embedding, second_condition)
        self.assertFalse(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
