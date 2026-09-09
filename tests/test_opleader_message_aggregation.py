from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
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
from opinion_model.opleader import OpinionLeaderMessageAggregation


class OpinionLeaderMessageAggregationTests(unittest.TestCase):
    CONSUMER_ID = 0
    OTHER_CONSUMER_ID = 9
    LEADER_IDS = frozenset({1, 2})
    ORDINARY_ID = 3

    def setUp(self) -> None:
        self.aggregator = OpinionLeaderMessageAggregation(
            leader_ids=self.LEADER_IDS,
            leader_evidence_multiplier=3.0,
        )

    def exposure(
        self,
        producer_id: int,
        stance: int,
        label: str,
        *,
        consumer_id: int | None = None,
    ) -> Exposure:
        resolved_consumer_id = (
            self.CONSUMER_ID if consumer_id is None else consumer_id
        )
        return Exposure(
            round_index=1,
            consumer_id=resolved_consumer_id,
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

    def test_ordinary_and_leader_messages_receive_distinct_weights(self) -> None:
        exposures = (
            self.exposure(self.ORDINARY_ID, 1, "ordinary-support"),
            self.exposure(1, -1, "leader-oppose"),
        )
        evidence = self.aggregator(exposures, AggregationContext(0.5))

        self.assertEqual(evidence.n_support, 1)
        self.assertEqual(evidence.n_oppose, 1)
        self.assertAlmostEqual(evidence.weighted_support, 0.5)
        self.assertAlmostEqual(evidence.weighted_oppose, 1.5)

    def test_weighted_evidence_updates_existing_beta_state(self) -> None:
        exposures = (
            self.exposure(self.ORDINARY_ID, 1, "ordinary-support"),
            self.exposure(1, -1, "leader-oppose"),
        )
        evidence = self.aggregator(exposures, AggregationContext(0.5))
        updated = propose_opinion_update(
            AgentState(BetaBelief(2.0, 2.0)),
            evidence,
        )

        self.assertAlmostEqual(updated.belief.a, 2.5)
        self.assertAlmostEqual(updated.belief.b, 3.5)
        self.assertAlmostEqual(updated.belief.mean, 2.5 / 6.0)
        self.assertAlmostEqual(updated.belief.concentration, 6.0)

    def test_multiple_leader_messages_accumulate_weight(self) -> None:
        exposures = (
            self.exposure(1, 1, "leader-1"),
            self.exposure(2, 1, "leader-2"),
        )
        evidence = self.aggregator(exposures, AggregationContext(0.5))

        self.assertEqual(evidence, MessageEvidence(2, 0, 3.0, 0.0))

    def test_aggregation_is_independent_of_exposure_order(self) -> None:
        exposures = (
            self.exposure(self.ORDINARY_ID, 1, "ordinary-support"),
            self.exposure(1, -1, "leader-oppose"),
        )
        context = AggregationContext(0.5)
        self.assertEqual(
            self.aggregator(exposures, context),
            self.aggregator(tuple(reversed(exposures)), context),
        )

    def test_zero_base_weight_preserves_raw_counts_only(self) -> None:
        exposures = (self.exposure(1, 1, "leader-support"),)
        evidence = self.aggregator(exposures, AggregationContext(0.0))

        self.assertEqual(evidence, MessageEvidence(1, 0, 0.0, 0.0))

    def test_zero_leader_multiplier_preserves_raw_count_only(self) -> None:
        aggregator = OpinionLeaderMessageAggregation(
            leader_ids=self.LEADER_IDS,
            leader_evidence_multiplier=0.0,
        )
        exposures = (self.exposure(1, -1, "leader-oppose"),)

        evidence = aggregator(exposures, AggregationContext(0.5))

        self.assertEqual(evidence, MessageEvidence(0, 1, 0.0, 0.0))

    def test_unit_multiplier_reproduces_baseline_aggregation(self) -> None:
        aggregator = OpinionLeaderMessageAggregation(
            leader_ids=self.LEADER_IDS,
            leader_evidence_multiplier=1.0,
        )
        exposures = (
            self.exposure(self.ORDINARY_ID, 1, "ordinary-support"),
            self.exposure(1, -1, "leader-oppose"),
        )
        context = AggregationContext(0.25)

        self.assertEqual(
            aggregator(exposures, context),
            aggregate_messages(exposures, context),
        )

    def test_recipient_identity_does_not_change_evidence(self) -> None:
        first = (
            self.exposure(self.ORDINARY_ID, 1, "ordinary-support"),
            self.exposure(1, -1, "leader-oppose"),
        )
        second = tuple(
            self.exposure(
                exposure.message.producer_id,
                exposure.message.stance,
                exposure.message.message_id.split(":", maxsplit=1)[1],
                consumer_id=self.OTHER_CONSUMER_ID,
            )
            for exposure in first
        )
        context = AggregationContext(0.5)

        self.assertEqual(
            self.aggregator(first, context),
            self.aggregator(second, context),
        )

    def test_exposures_for_multiple_recipients_are_rejected(self) -> None:
        exposures = (
            self.exposure(self.ORDINARY_ID, 1, "ordinary-support"),
            self.exposure(
                1,
                -1,
                "leader-oppose",
                consumer_id=self.OTHER_CONSUMER_ID,
            ),
        )

        with self.assertRaisesRegex(ValueError, "multiple recipients"):
            self.aggregator(exposures, AggregationContext(0.5))

    def test_invalid_leader_ids_are_rejected(self) -> None:
        for leader_ids in (frozenset({True}), frozenset({-1}), frozenset({1.5})):
            with self.subTest(leader_ids=leader_ids):
                with self.assertRaisesRegex(
                    ValueError,
                    "Leader IDs must be non-negative integers",
                ):
                    OpinionLeaderMessageAggregation(
                        leader_ids=leader_ids,
                        leader_evidence_multiplier=1.0,
                    )

    def test_invalid_leader_multiplier_is_rejected(self) -> None:
        for multiplier in (-0.01, float("inf"), float("nan")):
            with self.subTest(multiplier=multiplier):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and non-negative",
                ):
                    OpinionLeaderMessageAggregation(
                        leader_ids=self.LEADER_IDS,
                        leader_evidence_multiplier=multiplier,
                    )

    def test_configuration_is_normalized_and_immutable(self) -> None:
        supplied_ids = {1}
        aggregator = OpinionLeaderMessageAggregation(
            leader_ids=supplied_ids,
            leader_evidence_multiplier=2,
        )
        supplied_ids.add(2)

        self.assertEqual(aggregator.leader_ids, frozenset({1}))
        self.assertEqual(aggregator.leader_evidence_multiplier, 2.0)
        with self.assertRaises(FrozenInstanceError):
            aggregator.leader_evidence_multiplier = 3.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
