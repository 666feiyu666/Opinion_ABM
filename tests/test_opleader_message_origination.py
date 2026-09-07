from __future__ import annotations

import sys
import unittest
from math import exp, log
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.baseline import beta_tail_support_probability
from opinion_model.core import (
    AgentState,
    BetaBelief,
    MessageOrigination,
    OriginationContext,
)
from opinion_model.opleader import (
    OpinionLeaderMessageOrigination,
    origination_probability,
)


class CountingRng:
    """Minimal deterministic random source that records draw counts."""

    def __init__(self, *values: float):
        self._values = iter(values)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return next(self._values)


class OriginationProbabilityTests(unittest.TestCase):
    def test_hand_calculable_leader_odds_multiplier(self) -> None:
        ordinary = origination_probability(
            base_origination_probability=0.2,
            round_index=1,
            is_leader=False,
            interest_decay=0.0,
            leader_log_odds_advantage=log(4.0),
        )
        leader = origination_probability(
            base_origination_probability=0.2,
            round_index=1,
            is_leader=True,
            interest_decay=0.0,
            leader_log_odds_advantage=log(4.0),
        )

        self.assertAlmostEqual(ordinary, 0.2)
        self.assertAlmostEqual(leader, 0.5)

    def test_interest_decay_reduces_probability_for_both_roles(self) -> None:
        for is_leader in (False, True):
            probabilities = [
                origination_probability(
                    base_origination_probability=0.1,
                    round_index=round_index,
                    is_leader=is_leader,
                    interest_decay=0.08,
                    leader_log_odds_advantage=log(3.0),
                )
                for round_index in range(1, 8)
            ]
            self.assertTrue(
                all(
                    later < earlier
                    for earlier, later in zip(probabilities, probabilities[1:])
                )
            )

    def test_zero_decay_is_constant_across_rounds(self) -> None:
        probabilities = [
            origination_probability(
                base_origination_probability=0.1,
                round_index=round_index,
                is_leader=True,
                interest_decay=0.0,
                leader_log_odds_advantage=log(2.0),
            )
            for round_index in (1, 10, 50)
        ]
        self.assertTrue(np.allclose(probabilities, probabilities[0]))

    def test_zero_leader_advantage_makes_roles_equal(self) -> None:
        ordinary = origination_probability(
            base_origination_probability=0.1,
            round_index=12,
            is_leader=False,
            interest_decay=0.03,
            leader_log_odds_advantage=0.0,
        )
        leader = origination_probability(
            base_origination_probability=0.1,
            round_index=12,
            is_leader=True,
            interest_decay=0.03,
            leader_log_odds_advantage=0.0,
        )
        self.assertAlmostEqual(ordinary, leader)

    def test_leader_advantage_has_constant_odds_ratio(self) -> None:
        advantage = log(3.0)
        ordinary = origination_probability(
            base_origination_probability=0.1,
            round_index=20,
            is_leader=False,
            interest_decay=0.04,
            leader_log_odds_advantage=advantage,
        )
        leader = origination_probability(
            base_origination_probability=0.1,
            round_index=20,
            is_leader=True,
            interest_decay=0.04,
            leader_log_odds_advantage=advantage,
        )
        ordinary_odds = ordinary / (1.0 - ordinary)
        leader_odds = leader / (1.0 - leader)
        self.assertAlmostEqual(leader_odds / ordinary_odds, exp(advantage))

    def test_invalid_parameters_are_rejected(self) -> None:
        valid = {
            "base_origination_probability": 0.1,
            "round_index": 1,
            "is_leader": False,
            "interest_decay": 0.0,
            "leader_log_odds_advantage": 0.0,
        }
        invalid_changes = (
            {"base_origination_probability": 0.0},
            {"base_origination_probability": 1.0},
            {"base_origination_probability": float("nan")},
            {"round_index": 0},
            {"round_index": True},
            {"is_leader": 1},
            {"interest_decay": -0.01},
            {"interest_decay": float("inf")},
            {"leader_log_odds_advantage": -0.01},
            {"leader_log_odds_advantage": float("inf")},
        )
        for change in invalid_changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                origination_probability(**(valid | change))


class BetaTailStanceProbabilityTests(unittest.TestCase):
    def test_symmetric_beliefs_give_half_support_probability(self) -> None:
        for shape in (1.0, 2.0, 20.0):
            state = AgentState(BetaBelief(shape, shape))
            self.assertAlmostEqual(beta_tail_support_probability(state), 0.5)

    def test_greater_concentration_strengthens_a_favored_stance(self) -> None:
        low = AgentState(BetaBelief(6.0, 4.0))
        high = AgentState(BetaBelief(60.0, 40.0))
        self.assertGreater(
            beta_tail_support_probability(high),
            beta_tail_support_probability(low),
        )

    def test_mirrored_beliefs_have_complementary_probabilities(self) -> None:
        support = beta_tail_support_probability(AgentState(BetaBelief(7.0, 3.0)))
        oppose = beta_tail_support_probability(AgentState(BetaBelief(3.0, 7.0)))
        self.assertAlmostEqual(support + oppose, 1.0)


class OpinionLeaderMessageOriginationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = OpinionLeaderMessageOrigination(
            leader_ids=frozenset({1}),
            interest_decay=0.03,
            leader_log_odds_advantage=log(4.0),
        )
        self.state = AgentState(BetaBelief(3.0, 2.0))

    def test_implements_message_origination_protocol(self) -> None:
        self.assertIsInstance(self.rule, MessageOrigination)

    def test_ordinary_agents_are_eligible_to_originate(self) -> None:
        origination_rng = CountingRng(0.0)
        stance_rng = CountingRng(0.0)
        outcome = self.rule(
            0,
            self.state,
            OriginationContext(1, 0.2),
            origination_rng,
            stance_rng,
        )

        self.assertTrue(outcome.did_originate)
        self.assertAlmostEqual(outcome.origination_probability, 0.2)
        self.assertIsNotNone(outcome.message)
        self.assertEqual(origination_rng.calls, 1)
        self.assertEqual(stance_rng.calls, 1)

    def test_leaders_have_greater_origination_probability(self) -> None:
        ordinary = self.rule(
            0,
            self.state,
            OriginationContext(1, 0.2),
            CountingRng(0.99),
            CountingRng(),
        )
        leader = self.rule(
            1,
            self.state,
            OriginationContext(1, 0.2),
            CountingRng(0.99),
            CountingRng(),
        )
        self.assertAlmostEqual(ordinary.origination_probability, 0.2)
        self.assertAlmostEqual(leader.origination_probability, 0.5)

    def test_failed_origination_does_not_draw_a_stance(self) -> None:
        origination_rng = CountingRng(0.99)
        stance_rng = CountingRng()
        outcome = self.rule(
            0,
            self.state,
            OriginationContext(1, 0.2),
            origination_rng,
            stance_rng,
        )

        self.assertFalse(outcome.did_originate)
        self.assertIsNone(outcome.message)
        self.assertEqual(origination_rng.calls, 1)
        self.assertEqual(stance_rng.calls, 0)

    def test_successful_origination_creates_exactly_one_belief_derived_message(self) -> None:
        outcome = self.rule(
            1,
            self.state,
            OriginationContext(3, 0.2),
            CountingRng(0.0),
            CountingRng(0.0),
        )

        self.assertTrue(outcome.did_originate)
        self.assertEqual(outcome.message.message_id, "r3:a1")
        self.assertEqual(outcome.message.producer_id, 1)
        self.assertEqual(outcome.message.stance, 1)
        self.assertAlmostEqual(
            outcome.support_probability,
            beta_tail_support_probability(self.state),
        )

    def test_stance_draw_can_create_an_opposition_message(self) -> None:
        outcome = self.rule(
            1,
            self.state,
            OriginationContext(2, 0.2),
            CountingRng(0.0),
            CountingRng(0.999999),
        )
        self.assertEqual(outcome.message.stance, -1)

    def test_origination_does_not_mutate_agent_state(self) -> None:
        before = self.state
        self.rule(
            1,
            self.state,
            OriginationContext(1, 0.2),
            CountingRng(0.0),
            CountingRng(0.0),
        )
        self.assertEqual(self.state, before)

    def test_fixed_random_streams_are_reproducible(self) -> None:
        def run(seed: int) -> list[tuple[bool, int | None]]:
            origination_rng = np.random.default_rng(seed)
            stance_rng = np.random.default_rng(seed + 1)
            return [
                (
                    outcome.did_originate,
                    None if outcome.message is None else outcome.message.stance,
                )
                for round_index in range(1, 11)
                for outcome in (
                    self.rule(
                        1,
                        self.state,
                        OriginationContext(round_index, 0.2),
                        origination_rng,
                        stance_rng,
                    ),
                )
            ]

        self.assertEqual(run(17), run(17))

    def test_invalid_rule_configuration_is_rejected(self) -> None:
        invalid_arguments = (
            {"leader_ids": frozenset({True})},
            {"leader_ids": frozenset({-1})},
            {"interest_decay": -0.01},
            {"interest_decay": float("inf")},
            {"leader_log_odds_advantage": -0.01},
            {"leader_log_odds_advantage": float("inf")},
        )
        defaults = {
            "leader_ids": frozenset({1}),
            "interest_decay": 0.0,
            "leader_log_odds_advantage": 0.0,
        }
        for change in invalid_arguments:
            with self.subTest(change=change), self.assertRaises(ValueError):
                OpinionLeaderMessageOrigination(**(defaults | change))


if __name__ == "__main__":
    unittest.main()
