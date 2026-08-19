from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import opinion_model


class PackageStructureTests(unittest.TestCase):
    def test_shared_package_imports(self):
        self.assertEqual(opinion_model.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
