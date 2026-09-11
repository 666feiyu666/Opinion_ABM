from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from math import log
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.opleader import (
    OpinionLeaderMessageAggregation,
    OpinionLeaderMessageOrigination,
    select_messages as select_opleader_messages,
)
from opinion_model.scenarios.opleader import (
    FixedOpleaderInitializer,
    OpleaderConfig,
    OpleaderExperimentConfig,
    OpleaderInitializationConfig,
    OpinionLeaderMechanismConfig,
    assemble_opleader_components,
    final_condition_metrics,
    initialize_opleader,
    load_opleader_experiment_config,
    opleader_diagnostic_frames,
    opleader_transition_ledger,
    run_opleader_condition,
    run_opleader_experiment,
    summarize_final_conditions,
)
from opinion_model.shared import (
    DEFAULT_COMPONENTS,
    RandomStreams,
    SimulationConfig,
    run_simulation,
    simulation_frames,
)


class OpleaderScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OpleaderConfig(
            simulation=SimulationConfig(
                agent_count=7,
                rounds=3,
                seed=20260910,
                base_origination_probability=0.3,
                interest_decay=0.02,
                evidence_weight=0.1,
                consumption_capacity=6,
            ),
            initialization=OpleaderInitializationConfig(
                network_m=2,
                leader_share=0.3,
                ordinary_mean_alpha=2.0,
                ordinary_concentration=4.0,
                leader_positive_a=18.0,
                leader_positive_b=2.0,
            ),
            opinion_leader=OpinionLeaderMechanismConfig(
                leader_log_odds_advantage=log(3.0),
                leader_evidence_multiplier=1.5,
            ),
        )
        self.initialization = initialize_opleader(
            self.config,
            "positive",
            RandomStreams(self.config.simulation.seed).initialization(),
        )
        self.components = assemble_opleader_components(
            self.config,
            initializer=FixedOpleaderInitializer(self.initialization.state),
            leader_ids=self.initialization.leader_ids,
        )
        self.run = run_opleader_condition(
            self.config,
            "positive",
            extremism_threshold=0.8,
        )

    def test_assembly_uses_opleader_mechanisms_without_platform(self) -> None:
        self.assertIsInstance(
            self.components.message_origination,
            OpinionLeaderMessageOrigination,
        )
        self.assertIs(
            self.components.message_selection,
            select_opleader_messages,
        )
        self.assertIsInstance(
            self.components.message_aggregation,
            OpinionLeaderMessageAggregation,
        )
        self.assertIs(
            self.components.network_update,
            DEFAULT_COMPONENTS.network_update,
        )

    def test_initialization_is_reproducible_and_selects_highest_reach(self) -> None:
        repeated = initialize_opleader(
            self.config,
            "positive",
            RandomStreams(self.config.simulation.seed).initialization(),
        )
        self.assertEqual(self.initialization, repeated)
        network = self.initialization.state.network
        in_degree = {agent_id: 0 for agent_id in network.neighbors_by_agent}
        for producers in network.neighbors_by_agent.values():
            for producer_id in producers:
                in_degree[producer_id] += 1
        expected = set(
            sorted(in_degree, key=lambda value: (-in_degree[value], value))[:2]
        )
        self.assertEqual(set(self.initialization.leader_ids), expected)

    def test_repository_initialization_matches_main_reference_fingerprint(self) -> None:
        experiment = load_opleader_experiment_config(
            PROJECT_ROOT / "configs" / "opleader.toml"
        )
        initialized = initialize_opleader(
            experiment.opleader,
            "positive",
            RandomStreams(experiment.seeds[0]).initialization(),
        )
        self.assertEqual(
            sorted(initialized.leader_ids),
            [0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 17, 20, 26, 28, 140],
        )
        self.assertEqual(
            initialized.state.network.eligible_producers(0)[:6],
            (1, 2, 3, 5, 7, 11),
        )
        self.assertEqual(
            sum(
                len(producers)
                for producers in initialized.state.network.neighbors_by_agent.values()
            ),
            1491,
        )
        belief = initialized.state.agents[2].belief
        self.assertAlmostEqual(belief.a, 2.1989388107033268)
        self.assertAlmostEqual(belief.b, 1.8010611892966732)

    def test_orientations_share_topology_leaders_and_ordinary_beliefs(self) -> None:
        initialized = {
            orientation: initialize_opleader(
                self.config,
                orientation,
                RandomStreams(self.config.simulation.seed).initialization(),
            )
            for orientation in (
                "positive",
                "negative",
                "balanced_positive",
                "balanced_negative",
            )
        }
        first = initialized["positive"]
        for value in initialized.values():
            self.assertEqual(value.state.network, first.state.network)
            self.assertEqual(value.leader_ids, first.leader_ids)
            for agent_id in set(value.state.agents) - set(value.leader_ids):
                self.assertEqual(
                    value.state.agents[agent_id],
                    first.state.agents[agent_id],
                )
        leader_count = len(first.leader_ids)
        self.assertEqual(len(first.positive_leader_ids), leader_count)
        self.assertEqual(len(initialized["negative"].positive_leader_ids), 0)
        self.assertEqual(
            len(initialized["balanced_positive"].positive_leader_ids),
            (leader_count + 1) // 2,
        )
        self.assertEqual(
            len(initialized["balanced_negative"].positive_leader_ids),
            leader_count // 2,
        )

    def test_network_is_static_and_every_exposure_uses_a_directed_tie(self) -> None:
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

    def test_role_dependent_origination_and_source_weighting_are_active(self) -> None:
        for round_result in self.run.simulation_result.rounds:
            probabilities = {
                outcome.agent_id: outcome.origination_probability
                for outcome in round_result.events.origination_outcomes
            }
            self.assertGreater(
                min(probabilities[value] for value in self.initialization.leader_ids),
                max(
                    probability
                    for agent_id, probability in probabilities.items()
                    if agent_id not in self.initialization.leader_ids
                ),
            )
        ledger = opleader_transition_ledger(
            self.run.simulation_result,
            self.run.initialization,
            last_round=3,
        )
        expected_support = self.config.simulation.evidence_weight * (
            ledger["ordinary_support_messages"]
            + self.config.opinion_leader.leader_evidence_multiplier
            * ledger["leader_support_messages"]
        )
        expected_oppose = self.config.simulation.evidence_weight * (
            ledger["ordinary_oppose_messages"]
            + self.config.opinion_leader.leader_evidence_multiplier
            * ledger["leader_oppose_messages"]
        )
        self.assertTrue(np.allclose(ledger["weighted_support"], expected_support))
        self.assertTrue(np.allclose(ledger["weighted_oppose"], expected_oppose))

    def test_transition_ledger_reconstructs_every_beta_update(self) -> None:
        ledger = opleader_transition_ledger(
            self.run.simulation_result,
            self.run.initialization,
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

    def test_diagnostic_frames_retain_roles_and_requested_rounds(self) -> None:
        frames = opleader_diagnostic_frames(
            self.run.simulation_result,
            self.run.initialization,
            last_round=2,
        )
        self.assertEqual(
            set(frames),
            {
                "states",
                "origination",
                "messages",
                "exposures",
                "aggregates",
                "network",
                "transitions",
            },
        )
        for frame in frames.values():
            if not frame.empty:
                self.assertLessEqual(int(frame["round"].max()), 2)
        self.assertIn("role", frames["states"])
        self.assertIn("producer_role", frames["messages"])
        self.assertIn("consumer_role", frames["exposures"])
        self.assertIn("leader_support_messages", frames["transitions"])

    def test_experiment_runs_four_orientations_per_seed(self) -> None:
        experiment = OpleaderExperimentConfig(
            experiment_id="test",
            status="test",
            seeds=(101, 102),
            orientations=(
                "positive",
                "negative",
                "balanced_positive",
                "balanced_negative",
            ),
            extremism_threshold=0.8,
            opleader=replace(
                self.config,
                simulation=replace(self.config.simulation, rounds=1),
            ),
        )
        metrics = run_opleader_experiment(experiment)
        self.assertEqual(len(metrics), 2 * 4 * 2)
        self.assertEqual(set(metrics["seed"]), {101, 102})
        self.assertEqual(set(metrics["orientation"]), set(experiment.orientations))
        final = final_condition_metrics(metrics)
        self.assertEqual(len(final), 2 * 3)
        self.assertEqual(set(final["condition"]), {"positive", "negative", "balanced"})
        summary = summarize_final_conditions(final)
        self.assertEqual(set(summary["condition"]), {"positive", "negative", "balanced"})

    def test_same_seed_and_agent_order_are_reproducible(self) -> None:
        repeated = run_opleader_condition(
            self.config,
            "positive",
            extremism_threshold=0.8,
        )
        expected = simulation_frames(self.run.simulation_result)
        observed = simulation_frames(repeated.simulation_result)
        for name in expected:
            pd.testing.assert_frame_equal(expected[name], observed[name])

        reverse = simulation_frames(
            run_simulation(
                self.config.simulation,
                self.components,
                agent_order=tuple(reversed(range(self.config.simulation.agent_count))),
            )
        )
        sort_keys = {
            "states": ["round", "agent_id"],
            "origination": ["round", "agent_id"],
            "messages": ["round", "producer_id"],
            "exposures": ["round", "consumer_id", "producer_id"],
            "aggregates": ["round", "consumer_id"],
            "network": ["round", "consumer_id", "producer_id"],
        }
        for name, keys in sort_keys.items():
            left = expected[name].sort_values(keys).reset_index(drop=True)
            right = reverse[name].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right)

    def test_configuration_rejects_platform_and_binding_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_count - 1"):
            replace(
                self.config,
                simulation=replace(
                    self.config.simulation,
                    consumption_capacity=5,
                ),
            )

        source = (PROJECT_ROOT / "configs" / "opleader.toml").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(source + "\n[platform]\nenabled = true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "platform"):
                load_opleader_experiment_config(path)

    def test_repository_configuration_loads_matched_values(self) -> None:
        experiment = load_opleader_experiment_config(
            PROJECT_ROOT / "configs" / "opleader.toml"
        )
        self.assertEqual(experiment.experiment_id, "opleader")
        self.assertEqual(experiment.opleader.simulation.agent_count, 500)
        self.assertEqual(experiment.opleader.simulation.rounds, 50)
        self.assertEqual(experiment.opleader.simulation.interest_decay, 0.02)
        self.assertEqual(experiment.opleader.simulation.consumption_capacity, 499)
        self.assertEqual(experiment.opleader.initialization.network_m, 3)
        self.assertEqual(
            experiment.opleader.initialization.leader_count(500),
            15,
        )
        self.assertEqual(len(experiment.seeds), 10)
        self.assertEqual(len(experiment.orientations), 4)


if __name__ == "__main__":
    unittest.main()
