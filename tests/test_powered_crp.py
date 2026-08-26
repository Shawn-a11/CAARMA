import unittest

from helper.synth_table import PersistentSynthState


class PoweredCrpTest(unittest.TestCase):
    @staticmethod
    def _state(policy="powered", power=0.75):
        state = PersistentSynthState(
            num_real=3,
            max_cols=6,
            pair_strategy="crp",
            crp_topk=2,
            reuse_policy=policy,
            reuse_power=power,
        )
        key_small = state._key(0, 1)
        key_large = state._key(0, 2)
        state._ensure_col(key_small)
        state._ensure_col(key_large)
        state._rebuild_pairs_by_speaker()
        state.created_pairs.update((key_small, key_large))
        state.pair_visits[key_small] = 4
        state.pair_visits[key_large] = 64
        return state, key_small, key_large

    def test_constructor_rejects_invalid_power(self):
        for power in (0.0, -0.5, 1.01):
            with self.assertRaises(ValueError):
                self._state(power=power)

    def test_power_one_matches_popularity_mass(self):
        popularity, small, large = self._state("popularity", 1.0)
        powered, powered_small, powered_large = self._state("powered", 1.0)

        self.assertEqual(
            popularity._reuse_mass(small),
            powered._reuse_mass(powered_small),
        )
        self.assertEqual(
            popularity._reuse_mass(large),
            powered._reuse_mass(powered_large),
        )

    def test_sublinear_power_reduces_rich_get_richer_ratio(self):
        state, small, large = self._state(power=0.5)
        ratio = state._reuse_mass(large) / state._reuse_mass(small)

        self.assertAlmostEqual(ratio, 4.0)
        self.assertLess(ratio, 64.0 / 4.0)

    def test_state_dict_round_trip(self):
        state, _, _ = self._state(power=0.75)
        restored, _, _ = self._state(policy="popularity", power=1.0)
        restored.load_state_dict(state.state_dict())

        self.assertEqual(restored.reuse_policy, "powered")
        self.assertAlmostEqual(restored.reuse_power, 0.75)


if __name__ == "__main__":
    unittest.main()
