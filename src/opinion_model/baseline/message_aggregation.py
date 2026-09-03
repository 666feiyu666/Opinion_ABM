"""Homogeneous multi-message aggregation for the null baseline."""

from __future__ import annotations

from opinion_model.core import AggregationContext, Exposure, MessageEvidence


def aggregate_messages(
    exposures: tuple[Exposure, ...],
    context: AggregationContext,
) -> MessageEvidence:
    n_support = sum(exposure.message.stance == 1 for exposure in exposures)
    n_oppose = sum(exposure.message.stance == -1 for exposure in exposures)
    return MessageEvidence(
        n_support=n_support,
        n_oppose=n_oppose,
        weighted_support=context.evidence_weight * n_support,
        weighted_oppose=context.evidence_weight * n_oppose,
    )
