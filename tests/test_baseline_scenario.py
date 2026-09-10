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
    BaselineInitializationConfig,
    OpinionLeaderMechanismConfig,
    PlatformMechanismConfig,
    assemble_baseline_components,
    initialize_baseline,
    load_baseline_experiment_config,
    run_baseline_condition,
)
from opinion_model.shared import (
    DEFAULT_COMPONENTS,
    RandomStreams,
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
            initialization=BaselineInitializationConfig(
                network_m=1,
                leader_share=0.25,
                ordinary_mean_alpha=2.0,
                ordinary_concentration=4.0,
                leader_positive_a=18.0,
                leader_positive_b=2.0,
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

    def test_orientation_cases_share_network_ordinary_beliefs_and_leaders(self) -> None:
        streams = RandomStreams(self.config.simulation.seed)
        positive = initialize_baseline(
            self.config,
            "positive",
            streams.initialization(),
        )
        streams = RandomStreams(self.config.simulation.seed)
        negative = initialize_baseline(
            self.config,
            "negative",
            streams.initialization(),
        )

        self.assertEqual(positive.state.network, negative.state.network)
        self.assertEqual(positive.leader_ids, negative.leader_ids)
        ordinary_ids = set(positive.state.agents) - positive.leader_ids
        for agent_id in ordinary_ids:
            self.assertEqual(
                positive.state.agents[agent_id],
                negative.state.agents[agent_id],
            )

    def test_positive_and_negative_leader_beliefs_are_mirrored(self) -> None:
        positive = initialize_baseline(
            self.config,
            "positive",
            RandomStreams(self.config.simulation.seed).initialization(),
        )
        negative = initialize_baseline(
            self.config,
            "negative",
            RandomStreams(self.config.simulation.seed).initialization(),
        )

        for leader_id in positive.leader_ids:
            positive_belief = positive.state.agents[leader_id].belief
            negative_belief = negative.state.agents[leader_id].belief
            self.assertEqual(positive_belief.a, negative_belief.b)
            self.assertEqual(positive_belief.b, negative_belief.a)

    def test_balanced_assignments_are_exact_mirrors(self) -> None:
        balanced_positive = initialize_baseline(
            self.config,
            "balanced_positive",
            RandomStreams(self.config.simulation.seed).initialization(),
        )
        balanced_negative = initialize_baseline(
            self.config,
            "balanced_negative",
            RandomStreams(self.config.simulation.seed).initialization(),
        )

        self.assertEqual(
            balanced_positive.positive_leader_ids,
            balanced_negative.negative_leader_ids,
        )
        self.assertEqual(
            balanced_positive.negative_leader_ids,
            balanced_negative.positive_leader_ids,
        )

    def test_condition_runner_records_initial_and_final_rounds(self) -> None:
        run = run_baseline_condition(
            self.config,
            "positive",
            extremism_threshold=0.8,
        )

        self.assertEqual(run.round_metrics["round"].tolist(), [0, 1])
        self.assertEqual(
            set(run.round_metrics["orientation"]),
            {"positive"},
        )

    def test_repository_baseline_configuration_loads(self) -> None:
        experiment = load_baseline_experiment_config(
            PROJECT_ROOT / "configs" / "baseline.toml"
        )

        self.assertEqual(experiment.baseline.simulation.agent_count, 500)
        self.assertEqual(experiment.baseline.simulation.rounds, 50)
        self.assertEqual(
            experiment.baseline.initialization.leader_count(500),
            15,
        )
        self.assertEqual(len(experiment.seeds), 10)


if __name__ == "__main__":
    unittest.main()
