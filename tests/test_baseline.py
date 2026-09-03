from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.baseline import (
    BASELINE_COMPONENTS,
    SimulationConfig,
    aggregate_messages,
    initialize_baseline,
    propose_opinion_update,
    propose_static_network,
    run_simulation,
    select_messages,
    simulation_frames,
)
from opinion_model.core import (
    AgentState,
    AggregationContext,
    BetaBelief,
    Message,
    MessageEvidence,
    NetworkUpdateContext,
    RoundEvents,
    SelectionContext,
)


class BaselineRuleTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()

    def test_default_configuration_matches_confirmed_toy_boundary(self):
        self.assertEqual(self.config.agent_count, 11)
        self.assertEqual(self.config.rounds, 10)
        self.assertEqual(self.config.consumption_capacity, 10)
        self.assertTrue(self.config.exclude_self_messages)

    def test_hand_calculable_self_exclusion_and_update(self):
        messages = tuple(
            Message(
                message_id=f"r1:a{producer_id}",
                round_index=1,
                producer_id=producer_id,
                stance=1 if producer_id < 8 else -1,
            )
            for producer_id in range(11)
        )
        initial = initialize_baseline(
            self.config,
            np.random.default_rng(0),
        )
        selection_context = SelectionContext(1, 10, True)
        support_batch = select_messages(
            0,
            messages,
            initial.network,
            selection_context,
            np.random.default_rng(1),
        )
        oppose_batch = select_messages(
            8,
            messages,
            initial.network,
            selection_context,
            np.random.default_rng(2),
        )
        support_evidence = aggregate_messages(
            support_batch,
            AggregationContext(0.1),
        )
        oppose_evidence = aggregate_messages(
            oppose_batch,
            AggregationContext(0.1),
        )
        prior = AgentState(BetaBelief(2.0, 2.0))
        support_after = propose_opinion_update(prior, support_evidence)
        oppose_after = propose_opinion_update(prior, oppose_evidence)

        self.assertEqual((support_evidence.n_support, support_evidence.n_oppose), (7, 3))
        self.assertEqual((oppose_evidence.n_support, oppose_evidence.n_oppose), (8, 2))
        self.assertAlmostEqual(support_after.belief.a, 2.7)
        self.assertAlmostEqual(support_after.belief.b, 2.3)
        self.assertAlmostEqual(oppose_after.belief.a, 2.8)
        self.assertAlmostEqual(oppose_after.belief.b, 2.2)

    def test_static_network_rule_returns_same_network(self):
        snapshot = initialize_baseline(self.config, np.random.default_rng(0))
        events = RoundEvents((), (), {})
        proposed = propose_static_network(
            snapshot.network,
            snapshot,
            events,
            NetworkUpdateContext(1),
            np.random.default_rng(1),
        )
        self.assertIs(proposed, snapshot.network)


class BaselineSimulationTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        self.result = run_simulation(self.config)
        self.frames = simulation_frames(self.result)

    def test_default_run_preserves_parent_notebook_trajectory(self):
        messages = self.frames["messages"]
        support_counts = (
            messages.assign(is_support=messages["stance"].eq(1).astype(int))
            .groupby("round")["is_support"]
            .sum()
            .tolist()
        )
        self.assertEqual(support_counts, [7, 6, 9, 9, 9, 8, 9, 9, 10, 9])
        final_states = self.frames["states"].query("round == 10")
        self.assertAlmostEqual(final_states["signed_mean"].mean(), 0.3896103896103896)

    def test_event_counts_and_null_exposure(self):
        production = self.frames["production"]
        messages = self.frames["messages"]
        exposures = self.frames["exposures"]
        aggregates = self.frames["aggregates"]
        states = self.frames["states"]

        self.assertEqual(len(production), 110)
        self.assertTrue(production["did_post"].all())
        self.assertEqual(len(messages), 110)
        self.assertEqual(len(exposures), 1_100)
        self.assertEqual(len(aggregates), 110)
        self.assertEqual(len(states), 121)
        self.assertTrue((exposures["consumer_id"] != exposures["producer_id"]).all())
        self.assertTrue(
            exposures.groupby(["round", "consumer_id"]).size().eq(10).all()
        )
        self.assertTrue(
            exposures.groupby(["round", "producer_id"]).size().eq(10).all()
        )

    def test_aggregates_reconstruct_every_transition(self):
        aggregates = self.frames["aggregates"]
        self.assertTrue(
            np.allclose(
                aggregates["a_after"],
                aggregates["a_before"] + aggregates["weighted_support"],
            )
        )
        self.assertTrue(
            np.allclose(
                aggregates["b_after"],
                aggregates["b_before"] + aggregates["weighted_oppose"],
            )
        )
        self.assertTrue(aggregates["consumed_total"].eq(10).all())

    def test_zero_evidence_weight_preserves_beliefs(self):
        config = replace(self.config, evidence_weight=0.0)
        states = simulation_frames(run_simulation(config))["states"]
        self.assertTrue(np.allclose(states["a"], 2.0))
        self.assertTrue(np.allclose(states["b"], 2.0))

    def test_zero_post_probability_records_opportunities_without_messages(self):
        config = replace(self.config, rounds=2, post_probability=0.0)
        frames = simulation_frames(run_simulation(config))
        self.assertEqual(len(frames["production"]), 22)
        self.assertFalse(frames["production"]["did_post"].any())
        self.assertTrue(frames["messages"].empty)
        self.assertTrue(frames["exposures"].empty)
        self.assertIn("stance", frames["messages"].columns)
        self.assertIn("consumer_id", frames["exposures"].columns)
        self.assertTrue(frames["aggregates"]["consumed_total"].eq(0).all())
        self.assertTrue(np.allclose(frames["states"]["a"], 2.0))
        self.assertTrue(np.allclose(frames["states"]["b"], 2.0))

    def test_same_seed_is_reproducible(self):
        repeated = simulation_frames(run_simulation(self.config))
        for name in self.frames:
            pd.testing.assert_frame_equal(self.frames[name], repeated[name])

    def test_agent_iteration_order_does_not_change_results(self):
        reverse_order = tuple(reversed(range(self.config.agent_count)))
        reversed_frames = simulation_frames(
            run_simulation(self.config, agent_order=reverse_order)
        )
        sort_keys = {
            "states": ["round", "agent_id"],
            "production": ["round", "agent_id"],
            "messages": ["round", "producer_id"],
            "exposures": ["round", "consumer_id", "producer_id"],
            "aggregates": ["round", "consumer_id"],
            "network": ["round", "consumer_id", "producer_id"],
        }
        for name, keys in sort_keys.items():
            expected = self.frames[name].sort_values(keys).reset_index(drop=True)
            observed = reversed_frames[name].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, observed)

    def test_components_are_replaceable_without_scheduler_flags(self):
        def ignore_messages(exposures, context):
            del exposures, context
            return MessageEvidence(0, 0, 0.0, 0.0)

        components = replace(
            BASELINE_COMPONENTS,
            message_aggregation=ignore_messages,
        )
        config = replace(self.config, rounds=2)
        states = simulation_frames(run_simulation(config, components))["states"]

        self.assertTrue(np.allclose(states["a"], 2.0))
        self.assertTrue(np.allclose(states["b"], 2.0))


if __name__ == "__main__":
    unittest.main()
