from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.baseline import aggregate_messages, propose_opinion_update
from opinion_model.core import (
    AgentState,
    AggregationContext,
    BetaBelief,
    Exposure,
    Message,
    MessageAggregation,
    MessageEvidence,
)
from opinion_model.opleader import OriginatorKind, SourceWeightedAggregation


class SourceWeightedAggregationTests(unittest.TestCase):
    PRESS_ID = 10
    LEADER_ID = 1
    CONSUMER_ID = 0

    def setUp(self) -> None:
        self.aggregator = SourceWeightedAggregation(
            originator_kind_by_id={
                self.PRESS_ID: OriginatorKind.PRESS,
                self.LEADER_ID: OriginatorKind.LEADER,
            },
            source_weight_by_kind={
                OriginatorKind.PRESS: 1.0,
                OriginatorKind.LEADER: 3.0,
            },
        )

    def exposure(self, producer_id: int, stance: int, label: str) -> Exposure:
        return Exposure(
            round_index=1,
            consumer_id=self.CONSUMER_ID,
            message=Message(
                message_id=f"r1:{label}",
                round_index=1,
                producer_id=producer_id,
                stance=stance,
            ),
        )

    def test_implements_existing_message_aggregation_protocol(self) -> None:
        self.assertIsInstance(self.aggregator, MessageAggregation)

    def test_empty_exposure_produces_zero_evidence(self) -> None:
        evidence = self.aggregator((), AggregationContext(0.5))
        self.assertEqual(evidence, MessageEvidence(0, 0, 0.0, 0.0))

    def test_press_and_leader_receive_distinct_source_weights(self) -> None:
        exposures = (
            self.exposure(self.PRESS_ID, 1, "press"),
            self.exposure(self.LEADER_ID, -1, "leader"),
        )
        evidence = self.aggregator(exposures, AggregationContext(0.5))

        self.assertEqual(evidence.n_support, 1)
        self.assertEqual(evidence.n_oppose, 1)
        self.assertAlmostEqual(evidence.weighted_support, 0.5)
        self.assertAlmostEqual(evidence.weighted_oppose, 1.5)

    def test_weighted_evidence_updates_existing_beta_state(self) -> None:
        exposures = (
            self.exposure(self.PRESS_ID, 1, "press"),
            self.exposure(self.LEADER_ID, -1, "leader"),
        )
        evidence = self.aggregator(exposures, AggregationContext(0.5))
        updated = propose_opinion_update(
            AgentState(BetaBelief(2.0, 2.0)),
            evidence,
        )

        self.assertAlmostEqual(updated.belief.a, 2.5)
        self.assertAlmostEqual(updated.belief.b, 3.5)
        self.assertAlmostEqual(updated.belief.mean, 2.5 / 6.0)

    def test_repeated_leader_messages_accumulate_weight(self) -> None:
        exposures = (
            self.exposure(self.LEADER_ID, 1, "leader-1"),
            self.exposure(self.LEADER_ID, 1, "leader-2"),
        )
        evidence = self.aggregator(exposures, AggregationContext(0.5))

        self.assertEqual(evidence, MessageEvidence(2, 0, 3.0, 0.0))

    def test_aggregation_is_independent_of_exposure_order(self) -> None:
        exposures = (
            self.exposure(self.PRESS_ID, 1, "press"),
            self.exposure(self.LEADER_ID, -1, "leader"),
        )
        context = AggregationContext(0.5)
        self.assertEqual(
            self.aggregator(exposures, context),
            self.aggregator(tuple(reversed(exposures)), context),
        )

    def test_zero_base_weight_preserves_raw_counts_only(self) -> None:
        exposures = (self.exposure(self.LEADER_ID, 1, "leader"),)
        evidence = self.aggregator(exposures, AggregationContext(0.0))

        self.assertEqual(evidence, MessageEvidence(1, 0, 0.0, 0.0))

    def test_unit_source_weights_reproduce_baseline_aggregation(self) -> None:
        aggregator = SourceWeightedAggregation(
            originator_kind_by_id={
                self.PRESS_ID: OriginatorKind.PRESS,
                self.LEADER_ID: OriginatorKind.LEADER,
            },
            source_weight_by_kind={
                OriginatorKind.PRESS: 1.0,
                OriginatorKind.LEADER: 1.0,
            },
        )
        exposures = (
            self.exposure(self.PRESS_ID, 1, "press"),
            self.exposure(self.LEADER_ID, -1, "leader"),
        )
        context = AggregationContext(0.25)

        self.assertEqual(
            aggregator(exposures, context),
            aggregate_messages(exposures, context),
        )

    def test_unknown_originator_is_rejected(self) -> None:
        exposures = (self.exposure(99, 1, "unknown"),)
        with self.assertRaisesRegex(ValueError, "No originator kind registered"):
            self.aggregator(exposures, AggregationContext(1.0))

    def test_invalid_source_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            SourceWeightedAggregation(
                originator_kind_by_id={self.PRESS_ID: OriginatorKind.PRESS},
                source_weight_by_kind={
                    OriginatorKind.PRESS: 1.0,
                    OriginatorKind.LEADER: float("inf"),
                },
            )

    def test_missing_source_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing source weights for: leader"):
            SourceWeightedAggregation(
                originator_kind_by_id={self.PRESS_ID: OriginatorKind.PRESS},
                source_weight_by_kind={OriginatorKind.PRESS: 1.0},
            )


if __name__ == "__main__":
    unittest.main()
