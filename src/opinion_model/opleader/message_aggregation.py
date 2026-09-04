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


@dataclass(frozen=True)
class SourceWeightedAggregation:
    """Aggregate messages using weights determined by originator kind.

    Each message contributes ``context.evidence_weight * source_weight`` to
    the Beta evidence associated with its stance. Raw message counts remain
    unweighted so observations can distinguish exposure from influence.
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
