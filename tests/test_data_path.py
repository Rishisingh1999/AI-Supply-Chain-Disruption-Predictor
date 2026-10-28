"""Lightweight tests for data-path resolution (Supply Chain)."""
import unittest
from pathlib import Path
import tempfile
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_experiment import resolve_data_path, FILENAME

class TestResolveDataPath(unittest.TestCase):
    def test_prefers_root_over_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / FILENAME).write_text("a")
            (root / "data").mkdir()
            (root / "data" / FILENAME).write_text("b")
            old = os.getcwd()
            try:
                os.chdir(root)
                p = resolve_data_path([Path(FILENAME), Path("data") / FILENAME])
                self.assertEqual(p.name, FILENAME)
                self.assertEqual(p.parent.resolve(), root.resolve())
            finally:
                os.chdir(old)

    def test_falls_back_to_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / FILENAME).write_text("x")
            old = os.getcwd()
            try:
                os.chdir(root)
                p = resolve_data_path([Path(FILENAME), Path("data") / FILENAME])
                self.assertIn("data", str(p))
            finally:
                os.chdir(old)

    def test_missing_raises_helpful_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            try:
                os.chdir(tmp)
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_data_path([Path(FILENAME), Path("data") / FILENAME])
                msg = str(ctx.exception)
                self.assertIn("global_supply_chain_risk_2026.csv", msg)
                self.assertIn("kaggle", msg.lower())
            finally:
                os.chdir(old)

if __name__ == "__main__":
    unittest.main()
