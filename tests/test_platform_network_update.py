from __future__ import annotations

import sys
import unittest
from math import log
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.core import (
    AgentState,
    BetaBelief,
    Exposure,
    Message,
    NetworkState,
    NetworkUpdate,
    NetworkUpdateContext,
    RoundEvents,
    WorldState,
)
from opinion_model.platform import (
    PlatformNetworkUpdate,
    belief_message_alignment,
    tie_dissolution_probability,
    tie_formation_probability,
)


class ControlledRng:
    def __init__(self, *values: float):
        self._values = iter(values)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return next(self._values)


class PlatformNetworkProbabilityTests(unittest.TestCase):
    def test_beta_tail_alignment_uses_direction_and_concentration(self) -> None:
        low = AgentState(BetaBelief(6.0, 4.0))
        high = AgentState(BetaBelief(60.0, 40.0))
        low_support = belief_message_alignment(low, 1)
        high_support = belief_message_alignment(high, 1)
        self.assertGreater(high_support, low_support)
        self.assertAlmostEqual(
            belief_message_alignment(high, -1),
            -high_support,
        )

    def test_formation_decreases_with_degree_and_increases_with_alignment(self) -> None:
        common = {
            "midpoint_probability": 0.1,
            "agent_count": 11,
            "degree_log_odds_strength": log(3.0),
            "alignment_log_odds_strength": log(3.0),
        }
        by_degree = [
            tie_formation_probability(
                **common,
                following_degree=degree,
                alignment=0.0,
            )
            for degree in (0, 5, 10)
        ]
        by_alignment = [
            tie_formation_probability(
                **common,
                following_degree=5,
                alignment=alignment,
            )
            for alignment in (-1.0, 0.0, 1.0)
        ]
        self.assertTrue(by_degree[0] > by_degree[1] > by_degree[2])
        self.assertTrue(by_alignment[0] < by_alignment[1] < by_alignment[2])
        self.assertAlmostEqual(by_degree[1], 0.1)

    def test_dissolution_increases_with_degree_and_decreases_with_alignment(self) -> None:
        common = {
            "midpoint_probability": 0.1,
            "agent_count": 11,
            "degree_log_odds_strength": log(3.0),
            "alignment_log_odds_strength": log(3.0),
        }
        by_degree = [
            tie_dissolution_probability(
                **common,
                following_degree=degree,
                alignment=0.0,
            )
            for degree in (0, 5, 10)
        ]
        by_alignment = [
            tie_dissolution_probability(
                **common,
                following_degree=5,
                alignment=alignment,
            )
            for alignment in (-1.0, 0.0, 1.0)
        ]
        self.assertTrue(by_degree[0] < by_degree[1] < by_degree[2])
        self.assertTrue(by_alignment[0] > by_alignment[1] > by_alignment[2])
        self.assertAlmostEqual(by_degree[1], 0.1)

    def test_invalid_probability_inputs_are_rejected(self) -> None:
        valid = {
            "midpoint_probability": 0.1,
            "following_degree": 2,
            "agent_count": 5,
            "alignment": 0.0,
            "degree_log_odds_strength": 1.0,
            "alignment_log_odds_strength": 1.0,
        }
        for change in (
            {"midpoint_probability": 0.0},
            {"midpoint_probability": 1.0},
            {"following_degree": 5},
            {"agent_count": 1},
            {"alignment": 1.1},
            {"degree_log_odds_strength": 0.0},
            {"alignment_log_odds_strength": -1.0},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                tie_formation_probability(**(valid | change))


class PlatformNetworkUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = PlatformNetworkUpdate(
            formation_midpoint_probability=0.1,
            formation_degree_log_odds_strength=log(3.0),
            formation_alignment_log_odds_strength=log(3.0),
            dissolution_midpoint_probability=0.1,
            dissolution_degree_log_odds_strength=log(3.0),
            dissolution_alignment_log_odds_strength=log(3.0),
        )
        agents = {
            0: AgentState(BetaBelief(8.0, 2.0)),
            1: AgentState(BetaBelief(2.0, 8.0)),
            2: AgentState(BetaBelief(5.0, 5.0)),
        }
        self.network = NetworkState({0: (1,), 1: (), 2: ()})
        self.snapshot = WorldState(0, agents, self.network)
        self.context = NetworkUpdateContext(1)

    @staticmethod
    def exposure(consumer_id: int, producer_id: int, stance: int = 1) -> Exposure:
        return Exposure(
            round_index=1,
            consumer_id=consumer_id,
            message=Message(
                message_id=f"r1:a{producer_id}",
                round_index=1,
                producer_id=producer_id,
                stance=stance,
            ),
        )

    @staticmethod
    def events(*exposures: Exposure) -> RoundEvents:
        return RoundEvents((), tuple(exposures), {})

    def test_implements_network_update_protocol(self) -> None:
        self.assertIsInstance(self.rule, NetworkUpdate)

    def test_exposures_are_classified_from_the_start_of_round_network(self) -> None:
        opportunities = self.rule.opportunities(
            self.network,
            self.snapshot,
            self.events(
                self.exposure(0, 1, -1),
                self.exposure(0, 2, 1),
            ),
            self.context,
        )
        self.assertEqual(
            [(item.producer_id, item.action) for item in opportunities],
            [(1, "untie"), (2, "tie")],
        )
        self.assertTrue(all(item.following_degree == 1 for item in opportunities))

    def test_accepted_additions_and_removals_commit_together(self) -> None:
        before = self.network
        proposed = self.rule(
            self.network,
            self.snapshot,
            self.events(
                self.exposure(0, 1, -1),
                self.exposure(0, 2, 1),
            ),
            self.context,
            ControlledRng(0.0, 0.0),
        )
        self.assertEqual(proposed.eligible_producers(0), (2,))
        self.assertIs(self.snapshot.network, before)
        self.assertEqual(before.eligible_producers(0), (1,))

    def test_no_exposure_produces_no_change_and_no_draws(self) -> None:
        rng = ControlledRng()
        proposed = self.rule(
            self.network,
            self.snapshot,
            self.events(),
            self.context,
            rng,
        )
        self.assertEqual(proposed, self.network)
        self.assertEqual(rng.calls, 0)

    def test_isolated_agent_can_form_a_tie(self) -> None:
        proposed = self.rule(
            self.network,
            self.snapshot,
            self.events(self.exposure(1, 2, 1)),
            self.context,
            ControlledRng(0.0),
        )
        self.assertEqual(proposed.eligible_producers(1), (2,))

    def test_duplicate_same_stance_exposure_is_evaluated_once(self) -> None:
        first = self.exposure(0, 2, 1)
        duplicate = Exposure(
            round_index=1,
            consumer_id=0,
            message=Message("r1:a2:duplicate", 1, 2, 1),
        )
        opportunities = self.rule.opportunities(
            self.network,
            self.snapshot,
            self.events(first, duplicate),
            self.context,
        )
        self.assertEqual(len(opportunities), 1)

    def test_conflicting_messages_from_one_producer_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-message origination boundary"):
            self.rule.opportunities(
                self.network,
                self.snapshot,
                self.events(
                    self.exposure(0, 2, 1),
                    self.exposure(0, 2, -1),
                ),
                self.context,
            )

    def test_invalid_rule_configuration_is_rejected(self) -> None:
        arguments = {
            "formation_midpoint_probability": 0.1,
            "formation_degree_log_odds_strength": 1.0,
            "formation_alignment_log_odds_strength": 1.0,
            "dissolution_midpoint_probability": 0.1,
            "dissolution_degree_log_odds_strength": 1.0,
            "dissolution_alignment_log_odds_strength": 1.0,
        }
        for change in (
            {"formation_midpoint_probability": 0.0},
            {"formation_degree_log_odds_strength": 0.0},
            {"dissolution_midpoint_probability": 1.0},
            {"dissolution_alignment_log_odds_strength": float("inf")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                PlatformNetworkUpdate(**(arguments | change))


if __name__ == "__main__":
    unittest.main()
