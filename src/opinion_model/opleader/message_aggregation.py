"""Source-dependent evidence aggregation for the opinion-leader case."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from opinion_model.core import AggregationContext, Exposure, MessageEvidence


@dataclass(frozen=True)
class OpinionLeaderMessageAggregation:
    """Aggregate messages using one relative opinion-leader multiplier.

    Ordinary-agent messages receive the common base evidence weight. Messages
    from agents in ``leader_ids`` receive that base weight multiplied by
    ``leader_evidence_multiplier``. Raw message counts remain unweighted so
    observations can distinguish exposure from influence.

    Recipient role is deliberately absent. Given the same prior and exposures,
    recipients therefore receive identical evidence regardless of their own
    role.
    """

    leader_ids: frozenset[int]
    leader_evidence_multiplier: float

    def __post_init__(self) -> None:
        normalized_ids = frozenset(self.leader_ids)
        for leader_id in normalized_ids:
            if (
                isinstance(leader_id, bool)
                or not isinstance(leader_id, int)
                or leader_id < 0
            ):
                raise ValueError("Leader IDs must be non-negative integers.")

        multiplier = float(self.leader_evidence_multiplier)
        if not isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(
                "leader_evidence_multiplier must be finite and non-negative."
            )

        object.__setattr__(self, "leader_ids", normalized_ids)
        object.__setattr__(self, "leader_evidence_multiplier", multiplier)

    def __call__(
        self,
        exposures: tuple[Exposure, ...],
        context: AggregationContext,
    ) -> MessageEvidence:
        """Return raw counts and source-weighted Beta evidence."""
        recipient_ids = {exposure.consumer_id for exposure in exposures}
        if len(recipient_ids) > 1:
            raise ValueError(
                "One aggregation call cannot combine exposures for multiple recipients."
            )

        n_support = 0
        n_oppose = 0
        weighted_support = 0.0
        weighted_oppose = 0.0

        for exposure in exposures:
            source_multiplier = (
                self.leader_evidence_multiplier
                if exposure.message.producer_id in self.leader_ids
                else 1.0
            )
            effective_weight = context.evidence_weight * source_multiplier
            if exposure.message.stance == 1:
                n_support += 1
                weighted_support += effective_weight
            else:
                n_oppose += 1
                weighted_oppose += effective_weight

        return MessageEvidence(
            n_support=n_support,
            n_oppose=n_oppose,
            weighted_support=weighted_support,
            weighted_oppose=weighted_oppose,
        )
