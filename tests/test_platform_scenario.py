from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from math import log
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.platform import PlatformMessageSelection, PlatformNetworkUpdate
from opinion_model.scenarios.platform import (
    FixedPlatformInitializer,
    PlatformConfig,
    PlatformExperimentConfig,
    PlatformInitializationConfig,
    PlatformMechanismConfig,
    assemble_platform_components,
    initialize_platform,
    load_platform_experiment_config,
    platform_diagnostic_frames,
    run_platform_condition,
    run_platform_experiment,
)
from opinion_model.shared import (
    DEFAULT_COMPONENTS,
    RandomStreams,
    SimulationConfig,
    simulation_frames,
)


class PlatformScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PlatformConfig(
            simulation=SimulationConfig(
                agent_count=8,
                rounds=3,
                seed=20260910,
                base_origination_probability=1.0,
                evidence_weight=0.1,
                consumption_capacity=3,
                exclude_self_messages=True,
            ),
            initialization=PlatformInitializationConfig(
                network_m=2,
                ordinary_mean_alpha=2.0,
                ordinary_concentration=4.0,
            ),
            platform=PlatformMechanismConfig(
                out_of_network_availability_probability=1.0,
                formation_midpoint_probability=0.25,
                formation_degree_log_odds_strength=log(3.0),
                formation_alignment_log_odds_strength=log(3.0),
                dissolution_midpoint_probability=0.25,
                dissolution_degree_log_odds_strength=log(3.0),
                dissolution_alignment_log_odds_strength=log(3.0),
            ),
        )
        self.initialization = initialize_platform(
            self.config,
            RandomStreams(self.config.simulation.seed).initialization(),
        )
        self.components = assemble_platform_components(
            self.config,
            initializer=FixedPlatformInitializer(self.initialization.state),
        )
        self.run = run_platform_condition(
            self.config,
            extremism_threshold=0.8,
        )

    def test_assembly_uses_platform_mechanisms_and_ordinary_shared_rules(self) -> None:
        self.assertIsInstance(
            self.components.message_selection,
            PlatformMessageSelection,
        )
        self.assertIsInstance(
            self.components.network_update,
            PlatformNetworkUpdate,
        )
        self.assertIs(
            self.components.message_origination,
            DEFAULT_COMPONENTS.message_origination,
        )
        self.assertIs(
            self.components.message_aggregation,
            DEFAULT_COMPONENTS.message_aggregation,
        )
        self.assertIs(
            self.components.opinion_update,
            DEFAULT_COMPONENTS.opinion_update,
        )

    def test_initialization_is_reproducible_and_sparse(self) -> None:
        repeated = initialize_platform(
            self.config,
            RandomStreams(self.config.simulation.seed).initialization(),
        )
        self.assertEqual(self.initialization, repeated)
        edge_count = sum(
            len(producers)
            for producers in self.initialization.state.network.neighbors_by_agent.values()
        )
        self.assertEqual(edge_count, 12)

    def test_repository_initialization_matches_main_and_null_fingerprint(self) -> None:
        experiment = load_platform_experiment_config(
            PROJECT_ROOT / "configs" / "platform.toml"
        )
        initialized = initialize_platform(
            experiment.platform_case,
            RandomStreams(experiment.seeds[0]).initialization(),
        ).state
        expected_beliefs = (
            (2.23495899895898, 1.7650410010410198),
            (1.6287346819762714, 2.3712653180237284),
            (2.1989388107033268, 1.8010611892966732),
        )
        for agent_id, (expected_a, expected_b) in enumerate(expected_beliefs):
            belief = initialized.agents[agent_id].belief
            self.assertAlmostEqual(belief.a, expected_a)
            self.assertAlmostEqual(belief.b, expected_b)
        self.assertEqual(
            initialized.network.eligible_producers(0)[:6],
            (1, 2, 3, 5, 7, 11),
        )
        self.assertEqual(
            sum(
                len(producers)
                for producers in initialized.network.neighbors_by_agent.values()
            ),
            1491,
        )

    def test_diagnostic_traces_reconstruct_selection_and_network_updates(self) -> None:
        frames = platform_diagnostic_frames(
            self.run.simulation_result,
            self.config,
            last_round=3,
        )
        selection = frames["selection_decisions"]
        network = frames["network_decisions"]
        transitions = frames["transitions"]

        self.assertEqual(len(selection), 3 * 8 * 7)
        self.assertTrue(selection["available"].all())
        self.assertTrue(selection["capacity_binding"].all())
        self.assertTrue(
            selection.groupby(["round", "consumer_id"])["retained"]
            .sum()
            .eq(3)
            .all()
        )
        self.assertEqual(
            set(selection.loc[selection["retained"], "channel"]),
            {"tie", "out_of_network"},
        )
        self.assertEqual(len(network), 3 * 8 * 3)
        self.assertTrue(network["probability"].between(0.0, 1.0).all())
        self.assertTrue(network["random_draw"].between(0.0, 1.0).all())
        self.assertTrue(
            network["accepted"].eq(
                network["random_draw"] < network["probability"]
            ).all()
        )
        self.assertTrue(
            (transitions["a_after"]
             == transitions["a_before"] + transitions["weighted_support"])
            .all()
        )
        self.assertTrue(
            (transitions["b_after"]
             == transitions["b_before"] + transitions["weighted_oppose"])
            .all()
        )

    def test_round_metrics_reconstruct_edge_changes(self) -> None:
        metrics = self.run.round_metrics.sort_values("round")
        for previous, current in zip(
            metrics.iloc[:-1].itertuples(),
            metrics.iloc[1:].itertuples(),
        ):
            self.assertEqual(
                current.edge_count,
                previous.edge_count
                + current.accepted_addition_count
                - current.accepted_removal_count,
            )
        self.assertTrue(
            metrics.loc[metrics["round"] > 0, "capacity_binding_rate"].eq(1.0).all()
        )

    def test_same_seed_and_agent_order_are_reproducible(self) -> None:
        repeated = run_platform_condition(
            self.config,
            extremism_threshold=0.8,
        )
        reversed_run = run_platform_condition(
            self.config,
            extremism_threshold=0.8,
            agent_order=tuple(reversed(range(self.config.simulation.agent_count))),
        )
        expected_frames = simulation_frames(self.run.simulation_result)
        repeated_frames = simulation_frames(repeated.simulation_result)
        reversed_frames = simulation_frames(reversed_run.simulation_result)
        sort_keys = {
            "states": ["round", "agent_id"],
            "origination": ["round", "agent_id"],
            "messages": ["round", "producer_id"],
            "exposures": ["round", "consumer_id", "producer_id"],
            "aggregates": ["round", "consumer_id"],
            "network": ["round", "consumer_id", "producer_id"],
        }
        for name, keys in sort_keys.items():
            pd.testing.assert_frame_equal(expected_frames[name], repeated_frames[name])
            expected = expected_frames[name].sort_values(keys).reset_index(drop=True)
            observed = reversed_frames[name].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, observed)
        pd.testing.assert_frame_equal(
            self.run.round_metrics,
            repeated.round_metrics,
        )
        pd.testing.assert_frame_equal(
            self.run.round_metrics,
            reversed_run.round_metrics,
        )

    def test_experiment_runs_once_per_seed(self) -> None:
        experiment = PlatformExperimentConfig(
            experiment_id="platform-test",
            status="test",
            seeds=(101, 102),
            extremism_threshold=0.8,
            platform_case=replace(
                self.config,
                simulation=replace(self.config.simulation, rounds=1),
            ),
        )
        metrics = run_platform_experiment(experiment)
        self.assertEqual(set(metrics["seed"]), {101, 102})
        self.assertEqual(set(metrics["condition"]), {"platform"})
        self.assertEqual(
            metrics.groupby("seed")["condition"].nunique().tolist(),
            [1, 1],
        )

    def test_configuration_rejects_opinion_leaders_and_loads_repository_values(
        self,
    ) -> None:
        source = (PROJECT_ROOT / "configs" / "platform.toml").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(
                source + "\n[opinion_leader]\nenabled = true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "opinion_leader"):
                load_platform_experiment_config(path)

        experiment = load_platform_experiment_config(
            PROJECT_ROOT / "configs" / "platform.toml"
        )
        self.assertEqual(experiment.experiment_id, "platform")
        self.assertEqual(experiment.platform_case.simulation.agent_count, 500)
        self.assertEqual(experiment.platform_case.simulation.rounds, 50)
        self.assertEqual(experiment.platform_case.simulation.interest_decay, 0.02)
        self.assertEqual(
            experiment.platform_case.simulation.consumption_capacity,
            10,
        )
        self.assertEqual(experiment.platform_case.initialization.network_m, 3)
        self.assertEqual(
            experiment.platform_case.platform.out_of_network_availability_probability,
            0.04,
        )
        self.assertEqual(len(experiment.seeds), 10)


if __name__ == "__main__":
    unittest.main()
