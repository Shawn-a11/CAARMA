import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "tools" / "audit_crp_occupancy.py"
SPEC = importlib.util.spec_from_file_location("audit_crp_occupancy", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class CrpOccupancyAuditTest(unittest.TestCase):
    def setUp(self):
        self.state = {
            "max_cols": 8,
            "pair_strategy": "crp",
            "crp_alpha": 1.0,
            "crp_topk": 4,
            "pair_col": [
                ((0, 1), 0),
                ((1, 2), 1),
                ((2, 3), 2),
                ((3, 4), 3),
                ((4, 5), 4),
            ],
            "created_pairs": [(0, 1), (1, 2), (2, 3), (3, 4)],
            "pair_visits": [
                ((0, 1), 1),
                ((1, 2), 2),
                ((2, 3), 3),
                ((3, 4), 4),
            ],
            "total_pair_visits": 10,
        }

    def test_metrics_distinguish_allocated_from_occupied(self):
        result = audit.audit_state(self.state)
        self.assertEqual(result["allocated_pairs"], 5)
        self.assertEqual(result["occupied_classes"], 4)
        self.assertEqual(result["singleton_ratio"], 0.25)
        self.assertEqual(result["doubleton_ratio"], 0.25)
        self.assertEqual(result["median_occupancy"], 2.5)
        self.assertEqual(result["p90_occupancy"], 4)
        self.assertEqual(result["p99_occupancy"], 4)
        self.assertEqual(result["maximum_occupancy"], 4)
        self.assertAlmostEqual(result["gini_coefficient"], 0.25)
        self.assertAlmostEqual(
            result["effective_number_of_classes"],
            math.exp(-sum(p * math.log(p) for p in (0.1, 0.2, 0.3, 0.4))),
        )
        self.assertFalse(result["registry_capacity_reached"])

    def test_checkpoint_discovery_and_epoch(self):
        checkpoint = {
            "epoch": 6,
            "state_dict": {
                "loss._extra_state": {
                    "persistence": True,
                    "synth": self.state,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch=6.ckpt"
            torch.save(checkpoint, path)
            result = audit.audit_checkpoint(path)

        self.assertEqual(result["checkpoint_epoch_zero_based"], 6)
        self.assertEqual(result["display_epoch_one_based"], 7)
        self.assertEqual(
            result["state_path"],
            "checkpoint.state_dict.loss._extra_state.synth",
        )

    def test_positive_visits_recover_missing_created_pairs(self):
        state = dict(self.state)
        state.pop("created_pairs")
        result = audit.audit_state(state)
        self.assertEqual(result["occupied_classes"], 4)
        self.assertEqual(result["total_visits"], 10)


if __name__ == "__main__":
    unittest.main()
