from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from math import log
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.baseline import (
    BASELINE_COMPONENTS,
    SimulationConfig,
    initialize_baseline,
    run_simulation,
    simulation_frames,
)
from opinion_model.core import NetworkState, WorldState
from opinion_model.platform import (
    PlatformMessageSelection,
    PlatformNetworkUpdate,
    network_decision_frame,
)


def initialize_sparse_ring(config, rng) -> WorldState:
    baseline = initialize_baseline(config, rng)
    agent_ids = tuple(baseline.agents)
    network = NetworkState(
        {
            consumer_id: (
                (consumer_id + 1) % config.agent_count,
                (consumer_id + 2) % config.agent_count,
            )
            for consumer_id in agent_ids
        }
    )
    return WorldState(0, baseline.agents, network)


class PlatformIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig(
            agent_count=12,
            rounds=3,
            consumption_capacity=10,
            seed=20260909,
        )
        self.network_rule = PlatformNetworkUpdate(
            formation_midpoint_probability=0.1,
            formation_degree_log_odds_strength=log(3.0),
            formation_alignment_log_odds_strength=log(3.0),
            dissolution_midpoint_probability=0.1,
            dissolution_degree_log_odds_strength=log(3.0),
            dissolution_alignment_log_odds_strength=log(3.0),
        )
        self.components = replace(
            BASELINE_COMPONENTS,
            initializer=initialize_sparse_ring,
            message_selection=PlatformMessageSelection(0.5),
            network_update=self.network_rule,
        )

    def test_connected_platform_run_records_reconstructable_decisions(self) -> None:
        result = run_simulation(self.config, self.components)
        decisions = network_decision_frame(result, self.network_rule)
        self.assertFalse(decisions.empty)
        self.assertTrue(decisions["probability"].between(0.0, 1.0).all())
        self.assertTrue(set(decisions["action"]).issubset({"tie", "untie"}))
        self.assertEqual(result.final_state.round_index, 3)

    def test_same_seed_is_reproducible(self) -> None:
        first = run_simulation(self.config, self.components)
        second = run_simulation(self.config, self.components)
        for name, first_frame in simulation_frames(first).items():
            pd.testing.assert_frame_equal(first_frame, simulation_frames(second)[name])
        pd.testing.assert_frame_equal(
            network_decision_frame(first, self.network_rule),
            network_decision_frame(second, self.network_rule),
        )

    def test_agent_iteration_order_does_not_change_results(self) -> None:
        forward = run_simulation(self.config, self.components)
        reverse = run_simulation(
            self.config,
            self.components,
            agent_order=tuple(reversed(range(self.config.agent_count))),
        )
        sort_keys = {
            "states": ["round", "agent_id"],
            "origination": ["round", "agent_id"],
            "messages": ["round", "producer_id"],
            "exposures": ["round", "consumer_id", "producer_id"],
            "aggregates": ["round", "consumer_id"],
            "network": ["round", "consumer_id", "producer_id"],
        }
        forward_frames = simulation_frames(forward)
        reverse_frames = simulation_frames(reverse)
        for name, keys in sort_keys.items():
            expected = forward_frames[name].sort_values(keys).reset_index(drop=True)
            observed = reverse_frames[name].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, observed)


if __name__ == "__main__":
    unittest.main()
