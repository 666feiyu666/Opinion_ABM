from __future__ import annotations

import sys
import unittest
from math import log
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.opleader import (
    OpinionLeaderMessageAggregation,
    OpinionLeaderMessageOrigination,
)
from opinion_model.platform import PlatformMessageSelection, PlatformNetworkUpdate
from opinion_model.scenarios.baseline import (
    BaselineConfig,
    OpinionLeaderMechanismConfig,
    PlatformMechanismConfig,
    assemble_baseline_components,
)
from opinion_model.shared import (
    DEFAULT_COMPONENTS,
    SimulationConfig,
    initialize_default,
    run_simulation,
)


class BaselineScenarioStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BaselineConfig(
            simulation=SimulationConfig(
                agent_count=4,
                rounds=1,
                seed=20260910,
                base_origination_probability=1.0,
                consumption_capacity=3,
            ),
            opinion_leader=OpinionLeaderMechanismConfig(
                interest_decay=0.03,
                leader_log_odds_advantage=log(4.0),
                leader_evidence_multiplier=4.0,
            ),
            platform=PlatformMechanismConfig(
                out_of_network_availability_probability=0.5,
                formation_midpoint_probability=0.1,
                formation_degree_log_odds_strength=log(3.0),
                formation_alignment_log_odds_strength=log(3.0),
                dissolution_midpoint_probability=0.1,
                dissolution_degree_log_odds_strength=log(3.0),
                dissolution_alignment_log_odds_strength=log(3.0),
            ),
        )

    def test_assembly_combines_the_intended_mechanisms(self) -> None:
        components = assemble_baseline_components(
            self.config,
            initializer=initialize_default,
            leader_ids={0},
        )

        self.assertIsInstance(
            components.message_origination,
            OpinionLeaderMessageOrigination,
        )
        self.assertIsInstance(
            components.message_selection,
            PlatformMessageSelection,
        )
        self.assertIsInstance(
            components.message_aggregation,
            OpinionLeaderMessageAggregation,
        )
        self.assertIsInstance(components.network_update, PlatformNetworkUpdate)
        self.assertEqual(components.message_origination.leader_ids, frozenset({0}))
        self.assertEqual(components.message_aggregation.leader_ids, frozenset({0}))
        self.assertIs(components.opinion_update, DEFAULT_COMPONENTS.opinion_update)

    def test_assembled_components_run_through_the_shared_scheduler(self) -> None:
        components = assemble_baseline_components(
            self.config,
            initializer=initialize_default,
            leader_ids={0},
        )

        result = run_simulation(self.config.simulation, components)

        self.assertEqual(result.final_state.round_index, 1)

    def test_out_of_range_leader_id_is_rejected_before_a_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "below agent_count"):
            assemble_baseline_components(
                self.config,
                initializer=initialize_default,
                leader_ids={self.config.simulation.agent_count},
            )


if __name__ == "__main__":
    unittest.main()
