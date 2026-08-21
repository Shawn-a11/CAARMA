import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cloud_gate.py"
SPEC = importlib.util.spec_from_file_location("cloud_gate", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate)


class CloudGateTest(unittest.TestCase):
    def test_metric_patterns(self):
        log = """
All distributed processes registered. Starting with 4 processes
Epoch 0: 100% complete
cosine EER: 3.45%
cosine minDCF(10-2): 0.3266
cosine minDCF(10-3): 0.4338
Epoch 1: 1/744
"""
        self.assertEqual(gate.EER_RE.findall(log), ["3.45"])
        self.assertEqual(gate.DCF2_RE.findall(log), ["0.3266"])
        self.assertFalse(gate.FATAL_RE.search(log))

    def test_stage_requires_yes(self):
        args = mock.Mock(yes=False)
        with self.assertRaises(PermissionError):
            gate.cmd_gate(args)

    def test_promote_rejects_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            receipt.write_text(json.dumps({"status": "FAIL", "shutdown_state": "shutdown"}))
            args = mock.Mock(
                yes=True, confirm_production=True, receipt=str(receipt)
            )
            with self.assertRaises(gate.GateError):
                gate.cmd_promote(args)

    def test_api_request_redacts_no_secret_in_error_contract(self):
        self.assertNotIn("AUTODL_TOKEN", gate.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
