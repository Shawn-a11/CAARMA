import argparse
import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "tools" / "audit_crp_occupancy.py"
SPEC = importlib.util.spec_from_file_location("audit_crp_occupancy", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)

from helper.synth_table import PersistentSynthState  # noqa: E402


def _visited_state(num_real=6, max_cols=6, visits=5):
    torch.manual_seed(0)
    random.seed(0)
    state = PersistentSynthState(num_real, max_cols, bank_size=2,
                                 pair_strategy="crp", crp_alpha=1.0, crp_topk=2)
    state.rebuild_pairing(torch.randn(8, num_real))
    state.begin_batch_stats(visits)
    for anchor in range(visits):
        key, col, _, event = state.select_pair(anchor % num_real)
        if key is None or col is None:
            continue
        state.commit_visit(key, col, event, "batch")
    state.finalize_batch_stats()
    return state


class SnapshotLiveStateTest(unittest.TestCase):
    def test_snapshot_matches_live_registry(self):
        state = _visited_state()
        row = audit.snapshot_live_state(state, epoch=3, global_step=1200)

        self.assertEqual(row["epoch_zero_based"], 3)
        self.assertEqual(row["display_epoch_one_based"], 4)
        self.assertEqual(row["global_step"], 1200)
        self.assertEqual(row["max_cols"], state.max_cols)
        self.assertEqual(row["allocated_pairs"], len(state.pair_col))
        positive = [count for count in state.pair_visits.values() if count > 0]
        self.assertEqual(row["occupied_classes"], len(positive))
        self.assertEqual(row["total_visits"], sum(positive))
        self.assertEqual(row["active_synth_cols_epoch"], len(state.activated_cols))
        self.assertEqual(row["pair_strategy"], "crp")

    def test_snapshot_without_epoch_fields(self):
        row = audit.snapshot_live_state(_visited_state())
        self.assertIsNone(row["epoch_zero_based"])
        self.assertIsNone(row["display_epoch_one_based"])
        self.assertIsNone(row["global_step"])


class EpochLogRoundTripTest(unittest.TestCase):
    def test_append_read_and_epoch_ordering(self):
        state = _visited_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "crp_occupancy_timeline.jsonl"
            for epoch in (2, 0, 1):
                audit.append_epoch_log(
                    path,
                    audit.snapshot_live_state(state, epoch=epoch,
                                              global_step=100 * (epoch + 1)),
                )
            rows = audit.read_epoch_log(path)

        self.assertEqual([row["epoch_zero_based"] for row in rows], [0, 1, 2])
        self.assertTrue(all(row["allocated_pairs"] == len(state.pair_col) for row in rows))

    def test_malformed_line_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            path.write_text('{"epoch_zero_based": 0}\nnot json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.read_epoch_log(path)


class SaturationTimelineTest(unittest.TestCase):
    def _row(self, epoch, saturated):
        return {
            "epoch_zero_based": epoch,
            "display_epoch_one_based": epoch + 1,
            "global_step": 10 * (epoch + 1),
            "registry_capacity_reached": saturated,
        }

    def test_first_saturated_row(self):
        rows = [self._row(0, False), self._row(1, True), self._row(2, True)]
        self.assertEqual(audit.first_saturated_row(rows)["display_epoch_one_based"], 2)
        self.assertIsNone(audit.first_saturated_row([self._row(0, False)]))

    def test_build_payload_reports_exact_epoch(self):
        rows = [self._row(0, False), self._row(1, False), self._row(2, True)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            for row in rows:
                audit.append_epoch_log(path, row)
            args = argparse.Namespace(checkpoint=None, checkpoint_dir=None,
                                      epoch_log=path)
            payload = audit._build_payload(args)

        self.assertEqual(payload["timeline"]["epochs_logged"], 3)
        self.assertEqual(payload["timeline"]["exact_saturation_epoch_one_based"], 3)
        self.assertIsNone(payload["timeline"]["first_observed_saturation"])


if __name__ == "__main__":
    unittest.main()
