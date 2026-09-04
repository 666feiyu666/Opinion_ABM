"""Source-dependent evidence aggregation for the opinion-leader case."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType

from opinion_model.core import AggregationContext, Exposure, MessageEvidence


class OriginatorKind(StrEnum):
    """Message-originator roles retained by the first mechanism probe."""

    PRESS = "press"
    LEADER = "leader"


class RecipientKind(StrEnum):
    """Message-recipient roles retained by the relation-aware probe."""

    LEADER = "leader"
    ORDINARY = "ordinary"


@dataclass(frozen=True)
class SourceWeightedAggregation:
    """Source-only null: aggregate using weights determined by originator kind.

    Each message contributes ``context.evidence_weight * source_weight`` to
    the Beta evidence associated with its stance. Raw message counts remain
    unweighted so observations can distinguish exposure from influence.

    Recipient role is deliberately absent. Given the same prior and exposures,
    leaders and ordinary recipients therefore update identically under this
    benchmark.
    """

    originator_kind_by_id: Mapping[int, OriginatorKind | str]
    source_weight_by_kind: Mapping[OriginatorKind | str, float]

    def __post_init__(self) -> None:
        normalized_originators: dict[int, OriginatorKind] = {}
        for originator_id, kind in self.originator_kind_by_id.items():
            if (
                isinstance(originator_id, bool)
                or not isinstance(originator_id, int)
                or originator_id < 0
            ):
                raise ValueError("Originator IDs must be non-negative integers.")
            normalized_originators[originator_id] = OriginatorKind(kind)

        normalized_weights: dict[OriginatorKind, float] = {}
        for kind, weight in self.source_weight_by_kind.items():
            normalized_kind = OriginatorKind(kind)
            normalized_weight = float(weight)
            if not isfinite(normalized_weight) or normalized_weight < 0.0:
                raise ValueError("Source weights must be finite and non-negative.")
            normalized_weights[normalized_kind] = normalized_weight

        missing = set(OriginatorKind) - set(normalized_weights)
        if missing:
            missing_names = ", ".join(sorted(kind.value for kind in missing))
            raise ValueError(f"Missing source weights for: {missing_names}.")

        object.__setattr__(
            self,
            "originator_kind_by_id",
            MappingProxyType(dict(sorted(normalized_originators.items()))),
        )
        object.__setattr__(
            self,
            "source_weight_by_kind",
            MappingProxyType(dict(sorted(normalized_weights.items()))),
        )

    def __call__(
        self,
        exposures: tuple[Exposure, ...],
        context: AggregationContext,
    ) -> MessageEvidence:
        """Return raw exposure counts and source-weighted Beta evidence."""
        n_support = 0
        n_oppose = 0
        weighted_support = 0.0
        weighted_oppose = 0.0

        for exposure in exposures:
            originator_id = exposure.message.producer_id
            try:
                kind = self.originator_kind_by_id[originator_id]
            except KeyError as error:
                raise ValueError(
                    f"No originator kind registered for producer {originator_id}."
                ) from error

            effective_weight = (
                context.evidence_weight * self.source_weight_by_kind[kind]
            )
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


@dataclass(frozen=True)
class SourceRecipientWeightedAggregation:
    """Aggregate using an explicit source-role by recipient-role matrix.

    Each message contributes
    ``context.evidence_weight * relation_weight[(source_kind, recipient_kind)]``
    to the Beta evidence associated with its stance. The four supported
    relations are press-to-leader, press-to-ordinary, leader-to-leader, and
    leader-to-ordinary. Requiring the complete matrix keeps those relations
    visible even when some weights are intentionally equal.

    This class changes evidence interpretation only. Differences caused by
    role-specific Beta priors remain an independent initialization mechanism.
    """

    originator_kind_by_id: Mapping[int, OriginatorKind | str]
    recipient_kind_by_id: Mapping[int, RecipientKind | str]
    relation_weight_by_kinds: Mapping[
        tuple[OriginatorKind | str, RecipientKind | str],
        float,
    ]

    def __post_init__(self) -> None:
        normalized_originators: dict[int, OriginatorKind] = {}
        for originator_id, kind in self.originator_kind_by_id.items():
            if (
                isinstance(originator_id, bool)
                or not isinstance(originator_id, int)
                or originator_id < 0
            ):
                raise ValueError("Originator IDs must be non-negative integers.")
            normalized_originators[originator_id] = OriginatorKind(kind)

        normalized_recipients: dict[int, RecipientKind] = {}
        for recipient_id, kind in self.recipient_kind_by_id.items():
            if (
                isinstance(recipient_id, bool)
                or not isinstance(recipient_id, int)
                or recipient_id < 0
            ):
                raise ValueError("Recipient IDs must be non-negative integers.")
            normalized_recipients[recipient_id] = RecipientKind(kind)

        normalized_weights: dict[tuple[OriginatorKind, RecipientKind], float] = {}
        for relation, weight in self.relation_weight_by_kinds.items():
            if not isinstance(relation, tuple) or len(relation) != 2:
                raise ValueError(
                    "Relation keys must be (originator kind, recipient kind) tuples."
                )
            originator_kind, recipient_kind = relation
            normalized_relation = (
                OriginatorKind(originator_kind),
                RecipientKind(recipient_kind),
            )
            normalized_weight = float(weight)
            if not isfinite(normalized_weight) or normalized_weight < 0.0:
                raise ValueError("Relation weights must be finite and non-negative.")
            normalized_weights[normalized_relation] = normalized_weight

        expected_relations = {
            (originator_kind, recipient_kind)
            for originator_kind in OriginatorKind
            for recipient_kind in RecipientKind
        }
        missing = expected_relations - set(normalized_weights)
        if missing:
            missing_names = ", ".join(
                f"{originator_kind.value}->{recipient_kind.value}"
                for originator_kind, recipient_kind in sorted(missing)
            )
            raise ValueError(f"Missing relation weights for: {missing_names}.")

        object.__setattr__(
            self,
            "originator_kind_by_id",
            MappingProxyType(dict(sorted(normalized_originators.items()))),
        )
        object.__setattr__(
            self,
            "recipient_kind_by_id",
            MappingProxyType(dict(sorted(normalized_recipients.items()))),
        )
        object.__setattr__(
            self,
            "relation_weight_by_kinds",
            MappingProxyType(dict(sorted(normalized_weights.items()))),
        )

    def __call__(
        self,
        exposures: tuple[Exposure, ...],
        context: AggregationContext,
    ) -> MessageEvidence:
        """Return raw counts and source-recipient-weighted Beta evidence."""
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
            originator_id = exposure.message.producer_id
            recipient_id = exposure.consumer_id
            try:
                originator_kind = self.originator_kind_by_id[originator_id]
            except KeyError as error:
                raise ValueError(
                    f"No originator kind registered for producer {originator_id}."
                ) from error
            try:
                recipient_kind = self.recipient_kind_by_id[recipient_id]
            except KeyError as error:
                raise ValueError(
                    f"No recipient kind registered for consumer {recipient_id}."
                ) from error

            effective_weight = context.evidence_weight * self.relation_weight_by_kinds[
                (originator_kind, recipient_kind)
            ]
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
