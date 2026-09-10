from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from math import log
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.shared import (
    DEFAULT_COMPONENTS,
    SimulationConfig,
    run_simulation,
    simulation_frames,
)
from opinion_model.opleader import (
    OpinionLeaderMessageAggregation,
    OpinionLeaderMessageOrigination,
    select_messages as select_opleader_messages,
)


class OpinionLeaderIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.leader_ids = frozenset({0, 1})
        self.config = SimulationConfig(
            agent_count=11,
            rounds=3,
            seed=20260909,
            base_origination_probability=0.2,
            consumption_capacity=10,
        )
        self.components = replace(
            DEFAULT_COMPONENTS,
            message_origination=OpinionLeaderMessageOrigination(
                leader_ids=self.leader_ids,
                interest_decay=0.03,
                leader_log_odds_advantage=log(4.0),
            ),
            message_selection=select_opleader_messages,
            message_aggregation=OpinionLeaderMessageAggregation(
                leader_ids=self.leader_ids,
                leader_evidence_multiplier=4.0,
            ),
        )

    def test_connected_opleader_run_uses_role_dependent_origination(self) -> None:
        result = run_simulation(self.config, self.components)

        self.assertEqual(result.final_state.round_index, self.config.rounds)
        for round_result in result.rounds:
            probabilities = {
                outcome.agent_id: outcome.origination_probability
                for outcome in round_result.events.origination_outcomes
            }
            leader_probabilities = [
                probabilities[agent_id] for agent_id in self.leader_ids
            ]
            ordinary_probabilities = [
                probability
                for agent_id, probability in probabilities.items()
                if agent_id not in self.leader_ids
            ]
            self.assertGreater(
                min(leader_probabilities),
                max(ordinary_probabilities),
            )

    def test_same_seed_is_reproducible(self) -> None:
        first = simulation_frames(run_simulation(self.config, self.components))
        second = simulation_frames(run_simulation(self.config, self.components))

        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_agent_iteration_order_does_not_change_results(self) -> None:
        forward = simulation_frames(run_simulation(self.config, self.components))
        reverse = simulation_frames(
            run_simulation(
                self.config,
                self.components,
                agent_order=tuple(reversed(range(self.config.agent_count))),
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
            expected = forward[name].sort_values(keys).reset_index(drop=True)
            observed = reverse[name].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, observed)

    def test_neutral_parameters_reproduce_the_shared_defaults(self) -> None:
        config = replace(self.config, base_origination_probability=1.0)
        neutral_components = replace(
            DEFAULT_COMPONENTS,
            message_origination=OpinionLeaderMessageOrigination(
                leader_ids=self.leader_ids,
                interest_decay=0.0,
                leader_log_odds_advantage=0.0,
            ),
            message_selection=select_opleader_messages,
            message_aggregation=OpinionLeaderMessageAggregation(
                leader_ids=self.leader_ids,
                leader_evidence_multiplier=1.0,
            ),
        )
        expected = simulation_frames(run_simulation(config, DEFAULT_COMPONENTS))
        observed = simulation_frames(run_simulation(config, neutral_components))

        for name in expected:
            pd.testing.assert_frame_equal(expected[name], observed[name])


if __name__ == "__main__":
    unittest.main()
