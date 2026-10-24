import tempfile
import unittest
from pathlib import Path

import numpy as np

from features import compare


class Run:
    def __init__(self, directory):
        self.d = Path(directory)
        self.times = np.array([0.0, 1.0])
        self.tspin = 0.0
        self.mask = np.array([True, True])

    def load(self, var):
        path = self.d / f"{var}.dat"
        return np.fromfile(path, "<f8").reshape(-1, 2) if path.exists() else None

    def sel(self, values):
        return values[:len(self.times)][self.times[:len(values)] >= self.tspin]


class CompareTests(unittest.TestCase):
    def test_compare_rejects_incomplete_or_unmatched_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_dir, test_dir = root / "ref", root / "test"
            ref_dir.mkdir()
            test_dir.mkdir()
            values = np.array([0.0, 1.0, 1.0, 0.0])
            values.tofile(ref_dir / "zs.dat")
            values.tofile(test_dir / "zs.dat")
            np.array([1.0, 2.0, 3.0, 4.0]).tofile(ref_dir / "points.dat")

            np.array([1.0, 2.0]).tofile(test_dir / "points.dat")
            self.assertFalse(compare(Run(ref_dir), Run(test_dir))["finite"])

            (test_dir / "points.dat").unlink()
            np.array([1.0, 2.0, 3.0, 4.0]).tofile(test_dir / "points.dat")
            np.array([5.0]).tofile(test_dir / "unexpected.dat")
            self.assertFalse(compare(Run(ref_dir), Run(test_dir))["finite"])

            (test_dir / "unexpected.dat").unlink()
            np.array([0.0, 1.0]).tofile(test_dir / "points.dat")
            self.assertFalse(compare(Run(ref_dir), Run(test_dir))["finite"])


if __name__ == "__main__":
    unittest.main()
