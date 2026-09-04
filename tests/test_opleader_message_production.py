from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.baseline import (
    belief_from_mean_concentration,
    support_probability as baseline_support_probability,
)
from opinion_model.core import (
    AgentState,
    MessageProduction,
    ProductionContext,
)
from opinion_model.opleader import (
    LeaderMessageProduction,
    ScheduledPressMessageProduction,
    confidence_sensitive_support_probability,
    posterior_mean_support_probability,
    produce_press_message,
)


class CountingRng:
    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        if not self.values:
            raise AssertionError("Unexpected random draw.")
        return self.values.pop(0)


def state_from_mean_concentration(mean: float, concentration: float) -> AgentState:
    return AgentState(belief_from_mean_concentration(mean, concentration))


class StanceProbabilityTests(unittest.TestCase):
    def test_posterior_mean_rule_returns_beta_mean(self) -> None:
        state = state_from_mean_concentration(0.6, 4.0)
        self.assertAlmostEqual(posterior_mean_support_probability(state), 0.6)

    def test_posterior_mean_rule_is_invariant_to_concentration(self) -> None:
        low = state_from_mean_concentration(0.6, 4.0)
        high = state_from_mean_concentration(0.6, 40.0)
        self.assertAlmostEqual(
            posterior_mean_support_probability(low),
            posterior_mean_support_probability(high),
        )

    def test_confidence_sensitive_rule_reuses_baseline_rule(self) -> None:
        state = state_from_mean_concentration(0.6, 4.0)
        self.assertIs(
            confidence_sensitive_support_probability,
            baseline_support_probability,
        )
        self.assertAlmostEqual(
            confidence_sensitive_support_probability(state),
            baseline_support_probability(state),
        )

    def test_confidence_sensitive_rule_responds_to_concentration(self) -> None:
        low = state_from_mean_concentration(0.6, 4.0)
        high = state_from_mean_concentration(0.6, 40.0)
        self.assertGreater(
            confidence_sensitive_support_probability(high),
            confidence_sensitive_support_probability(low),
        )

    def test_symmetric_beliefs_give_half_support_probability(self) -> None:
        for concentration in (4.0, 40.0):
            state = state_from_mean_concentration(0.5, concentration)
            with self.subTest(concentration=concentration):
                self.assertAlmostEqual(posterior_mean_support_probability(state), 0.5)
                self.assertAlmostEqual(
                    confidence_sensitive_support_probability(state),
                    0.5,
                )


class LeaderMessageProductionTests(unittest.TestCase):
    LEADER_ID = 1
    ORDINARY_ID = 2

    def setUp(self) -> None:
        self.state = state_from_mean_concentration(0.6, 4.0)
        self.producer = LeaderMessageProduction(
            leader_ids=frozenset({self.LEADER_ID}),
            support_probability_rule=posterior_mean_support_probability,
        )

    def test_implements_message_production_protocol(self) -> None:
        self.assertIsInstance(self.producer, MessageProduction)

    def test_ordinary_agent_is_silent_without_random_draws(self) -> None:
        posting_rng = CountingRng()
        stance_rng = CountingRng()
        outcome = self.producer(
            self.ORDINARY_ID,
            self.state,
            ProductionContext(1, 1.0),
            posting_rng,
            stance_rng,
        )

        self.assertFalse(outcome.did_post)
        self.assertEqual(outcome.post_probability, 0.0)
        self.assertIsNone(outcome.message)
        self.assertEqual(posting_rng.calls, 0)
        self.assertEqual(stance_rng.calls, 0)

    def test_zero_post_probability_produces_silence(self) -> None:
        posting_rng = CountingRng(0.0)
        stance_rng = CountingRng()
        outcome = self.producer(
            self.LEADER_ID,
            self.state,
            ProductionContext(1, 0.0),
            posting_rng,
            stance_rng,
        )

        self.assertFalse(outcome.did_post)
        self.assertIsNone(outcome.message)
        self.assertEqual(posting_rng.calls, 1)
        self.assertEqual(stance_rng.calls, 0)

    def test_unit_post_probability_produces_one_support_message(self) -> None:
        posting_rng = CountingRng(0.99)
        stance_rng = CountingRng(0.59)
        outcome = self.producer(
            self.LEADER_ID,
            self.state,
            ProductionContext(3, 1.0),
            posting_rng,
            stance_rng,
        )

        self.assertTrue(outcome.did_post)
        self.assertEqual(outcome.post_probability, 1.0)
        self.assertAlmostEqual(outcome.support_probability, 0.6)
        self.assertEqual(outcome.message.stance, 1)
        self.assertEqual(outcome.message.message_id, "r3:a1")
        self.assertEqual(posting_rng.calls, 1)
        self.assertEqual(stance_rng.calls, 1)

    def test_stance_draw_can_produce_opposition(self) -> None:
        outcome = self.producer(
            self.LEADER_ID,
            self.state,
            ProductionContext(1, 1.0),
            CountingRng(0.0),
            CountingRng(0.9),
        )
        self.assertEqual(outcome.message.stance, -1)

    def test_production_does_not_mutate_agent_state(self) -> None:
        before = self.state
        self.producer(
            self.LEADER_ID,
            self.state,
            ProductionContext(1, 1.0),
            CountingRng(0.0),
            CountingRng(0.0),
        )
        self.assertIs(self.state, before)

    def test_fixed_seed_is_reproducible(self) -> None:
        def run(seed: int) -> tuple[tuple[bool, int | None], ...]:
            posting_rng = np.random.default_rng(seed)
            stance_rng = np.random.default_rng(seed + 1)
            return tuple(
                (
                    outcome.did_post,
                    outcome.message.stance if outcome.message is not None else None,
                )
                for round_index in range(1, 21)
                for outcome in (
                    self.producer(
                        self.LEADER_ID,
                        self.state,
                        ProductionContext(round_index, 0.4),
                        posting_rng,
                        stance_rng,
                    ),
                )
            )

        self.assertEqual(run(7), run(7))

    def test_invalid_support_probability_is_rejected(self) -> None:
        producer = LeaderMessageProduction(
            leader_ids=frozenset({self.LEADER_ID}),
            support_probability_rule=lambda state: float("inf"),
        )
        with self.assertRaisesRegex(ValueError, "finite value in"):
            producer(
                self.LEADER_ID,
                self.state,
                ProductionContext(1, 1.0),
                CountingRng(0.0),
                CountingRng(0.0),
            )

    def test_invalid_leader_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            LeaderMessageProduction(leader_ids=frozenset({True}))


class PressMessageProductionTests(unittest.TestCase):
    PRESS_ID = 10

    def test_direct_press_production_is_deterministic(self) -> None:
        first = produce_press_message(self.PRESS_ID, 2, 1)
        second = produce_press_message(self.PRESS_ID, 2, 1)

        self.assertEqual(first, second)
        self.assertTrue(first.did_post)
        self.assertEqual(first.post_probability, 1.0)
        self.assertEqual(first.support_probability, 1.0)
        self.assertEqual(first.message.message_id, "r2:press10")
        self.assertEqual(first.message.stance, 1)

    def test_negative_press_stance_is_degenerate_opposition(self) -> None:
        outcome = produce_press_message(self.PRESS_ID, 1, -1)
        self.assertEqual(outcome.support_probability, 0.0)
        self.assertEqual(outcome.message.stance, -1)

    def test_invalid_direct_press_stance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stance must be"):
            produce_press_message(self.PRESS_ID, 1, 0)

    def test_schedule_produces_one_configured_message_per_round(self) -> None:
        producer = ScheduledPressMessageProduction(
            press_id=self.PRESS_ID,
            stance_by_round={1: 1, 2: 1, 3: -1},
        )
        outcomes = tuple(producer(round_index) for round_index in (1, 2, 3))

        self.assertTrue(all(outcome.did_post for outcome in outcomes))
        self.assertEqual(
            [outcome.message.stance for outcome in outcomes],
            [1, 1, -1],
        )

    def test_unspecified_schedule_round_is_rejected(self) -> None:
        producer = ScheduledPressMessageProduction(
            press_id=self.PRESS_ID,
            stance_by_round={1: 1},
        )
        with self.assertRaisesRegex(ValueError, "No press stance configured"):
            producer(2)

    def test_invalid_schedule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stances must be"):
            ScheduledPressMessageProduction(
                press_id=self.PRESS_ID,
                stance_by_round={1: 0},
            )


if __name__ == "__main__":
    unittest.main()
