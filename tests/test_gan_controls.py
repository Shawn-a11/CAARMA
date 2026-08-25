import unittest

from helper.gan_controls import (
    generator_lambda_plan,
    resolve_gan_config,
    scheduled_updates,
)


class GanScheduleTest(unittest.TestCase):
    def test_effective_config_preserves_zero_fixed_lambda_and_d_lr(self):
        resolved = resolve_gan_config(
            {
                "lambda_adv_mode": "fixed",
                "lambda_adv_fixed": 0.0,
                "discriminator_lr": 5e-5,
                "gan_schedule": "paired",
            },
            main_lr=1e-3,
            weight_decay=1e-7,
        )
        self.assertEqual(resolved["lambda_adv_fixed"], 0.0)
        self.assertEqual(resolved["discriminator_lr"], 5e-5)
        self.assertEqual(resolved["main_lr"], 1e-3)

    def test_effective_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            resolve_gan_config(
                {"discriminator_lr": 0.0},
                main_lr=1e-3,
                weight_decay=1e-7,
            )

    def test_paired_updates_both_optimizers(self):
        self.assertEqual(scheduled_updates("paired", 0, 0), ("d", "g"))
        self.assertEqual(scheduled_updates("paired", 20, 99), ("d", "g"))

    def test_source_pretrain_uses_five_g_then_one_d(self):
        updates = [
            scheduled_updates("source_state_machine", 0, batch_idx)[0]
            for batch_idx in range(12)
        ]
        self.assertEqual(updates, ["g"] * 5 + ["d"] + ["g"] * 5 + ["d"])

    def test_source_post_pretrain_alternates(self):
        updates = [
            scheduled_updates("source_state_machine", 16, batch_idx)[0]
            for batch_idx in range(4)
        ]
        self.assertEqual(updates, ["d", "g", "d", "g"])

    def test_fixed_lambda_never_adjusts(self):
        value, adjust = generator_lambda_plan(
            "fixed", 0.05, 0.25, 0.0005, True
        )
        self.assertEqual(value, 0.05)
        self.assertFalse(adjust)

    def test_dynamic_lambda_respects_pretrain(self):
        self.assertEqual(
            generator_lambda_plan("dynamic", 0.05, 0.25, 0.0005, True),
            (0.0005, False),
        )
        self.assertEqual(
            generator_lambda_plan("dynamic", 0.05, 0.25, 0.0005, False),
            (0.25, True),
        )


if __name__ == "__main__":
    unittest.main()
