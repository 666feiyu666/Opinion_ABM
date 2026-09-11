from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.scenarios.null import (
    FixedNullInitializer,
    NullConfig,
    NullExperimentConfig,
    NullInitializationConfig,
    assemble_null_components,
    initialize_null,
    load_null_experiment_config,
    null_diagnostic_frames,
    null_transition_ledger,
    run_null_condition,
    run_null_experiment,
)
from opinion_model.shared import (
    DEFAULT_COMPONENTS,
    RandomStreams,
    SimulationConfig,
    run_simulation,
    simulation_frames,
)


class NullScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NullConfig(
            simulation=SimulationConfig(
                agent_count=8,
                rounds=3,
                seed=20260910,
                base_origination_probability=1.0,
                evidence_weight=0.1,
                consumption_capacity=7,
            ),
            initialization=NullInitializationConfig(
                network_m=2,
                ordinary_mean_alpha=2.0,
                ordinary_concentration=4.0,
            ),
        )
        self.initialization = initialize_null(
            self.config,
            RandomStreams(self.config.simulation.seed).initialization(),
        )
        self.components = assemble_null_components(
            self.config,
            initializer=FixedNullInitializer(self.initialization.state),
        )
        self.run = run_null_condition(
            self.config,
            extremism_threshold=0.8,
        )

    def test_assembly_uses_only_shared_ordinary_mechanisms(self) -> None:
        for name in (
            "message_origination",
            "message_selection",
            "message_aggregation",
            "opinion_update",
            "network_update",
        ):
            self.assertIs(
                getattr(self.components, name),
                getattr(DEFAULT_COMPONENTS, name),
            )

    def test_initialization_is_reproducible_and_sparse(self) -> None:
        repeated = initialize_null(
            self.config,
            RandomStreams(self.config.simulation.seed).initialization(),
        )
        self.assertEqual(self.initialization.state, repeated.state)
        edge_count = sum(
            len(producers)
            for producers in self.initialization.state.network.neighbors_by_agent.values()
        )
        self.assertEqual(edge_count, 12)

    def test_repository_initialization_matches_main_reference_fingerprint(self) -> None:
        experiment = load_null_experiment_config(
            PROJECT_ROOT / "configs" / "null.toml"
        )
        initialized = initialize_null(
            experiment.null,
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

    def test_network_is_static_and_every_exposure_uses_a_tie(self) -> None:
        initial_network = self.run.simulation_result.initial_state.network
        ties = {
            (consumer_id, producer_id)
            for consumer_id, producer_ids in initial_network.neighbors_by_agent.items()
            for producer_id in producer_ids
        }
        for round_result in self.run.simulation_result.rounds:
            self.assertEqual(round_result.next_state.network, initial_network)
            for exposure in round_result.events.exposures:
                pair = (exposure.consumer_id, exposure.message.producer_id)
                self.assertIn(pair, ties)
                self.assertNotEqual(*pair)

    def test_evidence_is_homogeneous_and_all_tied_messages_are_retained(self) -> None:
        for round_result in self.run.simulation_result.rounds:
            messages_by_producer = {
                outcome.agent_id
                for outcome in round_result.events.origination_outcomes
                if outcome.message is not None
            }
            expected_exposures = sum(
                len(
                    set(round_result.snapshot.network.eligible_producers(agent_id))
                    & messages_by_producer
                )
                for agent_id in round_result.snapshot.agents
            )
            self.assertEqual(len(round_result.events.exposures), expected_exposures)
            for evidence in round_result.events.evidence_by_agent.values():
                self.assertLessEqual(
                    evidence.total_messages,
                    self.config.simulation.consumption_capacity,
                )
                self.assertAlmostEqual(
                    evidence.weighted_support,
                    self.config.simulation.evidence_weight * evidence.n_support,
                )
                self.assertAlmostEqual(
                    evidence.weighted_oppose,
                    self.config.simulation.evidence_weight * evidence.n_oppose,
                )

    def test_transition_ledger_reconstructs_every_beta_update(self) -> None:
        ledger = null_transition_ledger(
            self.run.simulation_result,
            last_round=3,
        )
        self.assertEqual(
            len(ledger),
            self.config.simulation.agent_count * self.config.simulation.rounds,
        )
        self.assertTrue(
            np.allclose(
                ledger["a_after"],
                ledger["a_before"] + ledger["weighted_support"],
            )
        )
        self.assertTrue(
            np.allclose(
                ledger["b_after"],
                ledger["b_before"] + ledger["weighted_oppose"],
            )
        )

    def test_diagnostic_frames_retain_only_requested_rounds(self) -> None:
        frames = null_diagnostic_frames(
            self.run.simulation_result,
            last_round=2,
        )
        self.assertEqual(set(frames), {
            "states",
            "origination",
            "messages",
            "exposures",
            "aggregates",
            "network",
            "transitions",
        })
        for frame in frames.values():
            self.assertLessEqual(int(frame["round"].max()), 2)
        self.assertIn(0, set(frames["states"]["round"]))
        self.assertNotIn(0, set(frames["transitions"]["round"]))

    def test_condition_runner_records_only_null_and_all_rounds(self) -> None:
        metrics = self.run.round_metrics
        self.assertEqual(metrics["round"].tolist(), [0, 1, 2, 3])
        self.assertEqual(set(metrics["condition"]), {"null"})
        self.assertNotIn("orientation", metrics)
        self.assertNotIn("leader_mean_signed_belief", metrics)

    def test_two_seed_experiment_runs_once_per_seed(self) -> None:
        experiment = NullExperimentConfig(
            experiment_id="test-null",
            status="test",
            seeds=(101, 102),
            extremism_threshold=0.8,
            null=replace(
                self.config,
                simulation=replace(self.config.simulation, rounds=1),
            ),
        )
        metrics = run_null_experiment(experiment)
        self.assertEqual(set(metrics["seed"]), {101, 102})
        self.assertEqual(len(metrics), 4)
        self.assertEqual(
            metrics.groupby("seed")["condition"].nunique().tolist(),
            [1, 1],
        )

    def test_same_seed_condition_is_reproducible(self) -> None:
        repeated = run_null_condition(
            self.config,
            extremism_threshold=0.8,
        )
        expected = simulation_frames(self.run.simulation_result)
        observed = simulation_frames(repeated.simulation_result)
        for name in expected:
            pd.testing.assert_frame_equal(expected[name], observed[name])

    def test_agent_iteration_order_does_not_change_null_results(self) -> None:
        reverse_order = tuple(reversed(range(self.config.simulation.agent_count)))
        reversed_result = run_simulation(
            self.config.simulation,
            self.components,
            agent_order=reverse_order,
        )
        expected = simulation_frames(self.run.simulation_result)
        observed = simulation_frames(reversed_result)
        sort_keys = {
            "states": ["round", "agent_id"],
            "origination": ["round", "agent_id"],
            "messages": ["round", "producer_id"],
            "exposures": ["round", "consumer_id", "producer_id"],
            "aggregates": ["round", "consumer_id"],
            "network": ["round", "consumer_id", "producer_id"],
        }
        for name, keys in sort_keys.items():
            expected_frame = expected[name].sort_values(keys).reset_index(drop=True)
            observed_frame = observed[name].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(expected_frame, observed_frame)

    def test_configuration_rejects_focal_mechanism_sections(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_count - 1"):
            replace(
                self.config,
                simulation=replace(
                    self.config.simulation,
                    consumption_capacity=6,
                ),
            )

        source = (PROJECT_ROOT / "configs" / "null.toml").read_text(
            encoding="utf-8"
        )
        with TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "invalid-null.toml"
            invalid_path.write_text(
                source + "\n[platform]\nenabled = false\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "cannot contain focal mechanism sections",
            ):
                load_null_experiment_config(invalid_path)

    def test_repository_configuration_loads_matched_values(self) -> None:
        experiment = load_null_experiment_config(
            PROJECT_ROOT / "configs" / "null.toml"
        )
        self.assertEqual(experiment.experiment_id, "null")
        self.assertEqual(experiment.null.simulation.agent_count, 500)
        self.assertEqual(experiment.null.simulation.rounds, 50)
        self.assertEqual(experiment.null.simulation.interest_decay, 0.02)
        self.assertEqual(experiment.null.simulation.consumption_capacity, 499)
        self.assertEqual(
            experiment.null.simulation.base_origination_probability,
            0.10,
        )
        self.assertEqual(experiment.null.initialization.network_m, 3)
        self.assertEqual(len(experiment.seeds), 10)


if __name__ == "__main__":
    unittest.main()
