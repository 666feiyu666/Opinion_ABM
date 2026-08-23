from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import opinion_model
from opinion_model import baseline, core, interfaces


class PackageStructureTests(unittest.TestCase):
    def test_shared_package_imports(self):
        self.assertEqual(opinion_model.__version__, "0.1.0")

    def test_framework_layers_import(self):
        self.assertTrue(core.__doc__)
        self.assertTrue(interfaces.__doc__)
        self.assertTrue(baseline.__doc__)

    def test_core_contracts_are_structural(self):
        class Formation:
            def form_information(self, agent_state, context):
                return (agent_state, context)

        class Effect:
            def apply_information(self, agent_state, consumed_information):
                return agent_state

        self.assertIsInstance(Formation(), core.InformationFormation)
        self.assertIsInstance(Effect(), core.InformationEffect)


if __name__ == "__main__":
    unittest.main()
