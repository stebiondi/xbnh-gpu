import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        base = self.work / "base1d"
        deck = base / "deck"
        deck.mkdir(parents=True)
        (deck / "params.txt").write_text("nx = 2\nny = 0\ntstop = 3\ntintg = 1\ndepfile = z.grd\n")
        (deck / "z.grd").write_text("-2 -2 -2\n")
        self.ref, self.test = base / "xb", base / "gpu"
        fields = {
            "zs": np.tile([0.0, 1.0, -1.0, 0.0], (3, 1)).T,
            "u": np.ones((4, 3)), "v": np.ones((4, 3)),
            "zb": np.full((4, 3), -2.0), "ccg": np.ones((8, 3)),
            "point001": np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
        }
        for directory in (self.ref, self.test):
            directory.mkdir()
            for name, values in fields.items():
                values.astype("<f8").tofile(directory / f"{name}.dat")
            np.ones((4, 3), dtype="<i4").tofile(directory / "wetz.dat")

    def compare(self, expected_status):
        report = self.work / "report.json"
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("features.py")),
                                 "compare", str(self.work), "base1d", "--json", str(report)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)
        row = json.loads(report.read_text())["decks"]["base1d"]
        self.assertEqual(row["pass"], expected_status == 0)
        return result.stdout, row

    def test_matching_files_still_compare_values(self):
        _, row = self.compare(0)
        self.assertEqual(row["Hs_rel"], 0.0)
        self.assertEqual(row["files_rel"], 0.0)
        self.assertNotIn("errors", row)
        np.full((4, 2), 100.0, dtype="<f8").tofile(self.test / "point001.dat")
        _, row = self.compare(1)
        self.assertGreater(row["files_rel"], 1.0)

    def test_mismatched_file_sizes_fail(self):
        cases = [("zs.dat", -24), ("u.dat", -24), ("v.dat", -24), ("zb.dat", -24),
                 ("ccg.dat", -1), ("point001.dat", -8), ("point001.dat", -64),
                 ("point001.dat", 8), ("wetz.dat", -4)]
        for name, size_change in cases:
            with self.subTest(file=name, size_change=size_change):
                original = (self.ref / name).read_bytes()
                changed = original[:size_change] if size_change < 0 else original + bytes(size_change)
                (self.test / name).write_bytes(changed)
                try:
                    output, row = self.compare(1)
                    self.assertIn(name, output)
                    self.assertTrue(any(name in error for error in row["errors"]))
                finally:
                    (self.test / name).write_bytes(original)

    def test_missing_output_fails(self):
        for name in ("u.dat", "point001.dat"):
            with self.subTest(file=name):
                (self.test / name).unlink()
                try:
                    output, row = self.compare(1)
                    self.assertIn(name, output)
                    self.assertTrue(any(name in error for error in row["errors"]))
                finally:
                    (self.test / name).write_bytes((self.ref / name).read_bytes())

    def test_unexpected_output_fails(self):
        (self.test / "unexpected.dat").write_bytes(bytes(8))
        output, row = self.compare(1)
        self.assertIn("unexpected.dat", output)
        self.assertTrue(any("unexpected.dat" in error for error in row["errors"]))


if __name__ == "__main__":
    unittest.main()
