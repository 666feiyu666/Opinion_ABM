"""Role- and tie-dependent message selection for the opinion-leader case."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import numpy as np

from opinion_model.core import Exposure, Message, NetworkState, SelectionContext
from opinion_model.opleader.message_aggregation import OriginatorKind, RecipientKind


@dataclass(frozen=True)
class OpinionLeaderMessageSelection:
    """Select press and opinion-leader messages under the first case design.

    Press messages bypass the adaptive-agent network. Each is delivered through
    an independent Bernoulli draw whose probability depends on whether the
    recipient is a leader or an ordinary agent. Leader messages are delivered
    deterministically only to ordinary agents with an incoming tie from that
    leader in ``NetworkState``. Leader-to-leader delivery is inactive.

    Selection never inspects message stance. Attention competition is also
    outside this V1 mechanism: ``SelectionContext.capacity`` must therefore be
    large enough to retain every message selected by access and network rules.
    """

    originator_kind_by_id: Mapping[int, OriginatorKind | str]
    recipient_kind_by_id: Mapping[int, RecipientKind | str]
    press_delivery_probability_by_recipient_kind: Mapping[
        RecipientKind | str,
        float,
    ]

    def __post_init__(self) -> None:
        normalized_originators: dict[int, OriginatorKind] = {}
        for originator_id, kind in self.originator_kind_by_id.items():
            self._validate_id(originator_id, "Originator")
            normalized_originators[originator_id] = OriginatorKind(kind)

        normalized_recipients: dict[int, RecipientKind] = {}
        for recipient_id, kind in self.recipient_kind_by_id.items():
            self._validate_id(recipient_id, "Recipient")
            normalized_recipients[recipient_id] = RecipientKind(kind)

        for shared_id in set(normalized_originators) & set(normalized_recipients):
            originator_kind = normalized_originators[shared_id]
            recipient_kind = normalized_recipients[shared_id]
            if originator_kind is OriginatorKind.PRESS:
                raise ValueError(
                    f"Press originator {shared_id} cannot also be an "
                    "adaptive recipient."
                )
            if recipient_kind is not RecipientKind.LEADER:
                raise ValueError(
                    f"Leader originator {shared_id} must have recipient kind 'leader'."
                )

        normalized_probabilities: dict[RecipientKind, float] = {}
        for kind, probability in (
            self.press_delivery_probability_by_recipient_kind.items()
        ):
            normalized_kind = RecipientKind(kind)
            normalized_probability = float(probability)
            if (
                not isfinite(normalized_probability)
                or not 0.0 <= normalized_probability <= 1.0
            ):
                raise ValueError(
                    "Press delivery probabilities must be finite values in [0, 1]."
                )
            normalized_probabilities[normalized_kind] = normalized_probability

        missing = set(RecipientKind) - set(normalized_probabilities)
        if missing:
            missing_names = ", ".join(sorted(kind.value for kind in missing))
            raise ValueError(
                f"Missing press delivery probabilities for: {missing_names}."
            )

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
            "press_delivery_probability_by_recipient_kind",
            MappingProxyType(dict(sorted(normalized_probabilities.items()))),
        )

    @staticmethod
    def _validate_id(value: int, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} IDs must be non-negative integers.")

    def __call__(
        self,
        consumer_id: int,
        message_pool: tuple[Message, ...],
        network: NetworkState,
        context: SelectionContext,
        rng: np.random.Generator,
    ) -> tuple[Exposure, ...]:
        """Return stance-blind exposures for one registered adaptive recipient."""
        try:
            recipient_kind = self.recipient_kind_by_id[consumer_id]
        except KeyError as error:
            raise ValueError(
                f"No recipient kind registered for consumer {consumer_id}."
            ) from error
        if consumer_id not in network.neighbors_by_agent:
            raise ValueError(
                f"Consumer {consumer_id} is not registered in the network."
            )

        eligible_leaders = set(network.eligible_producers(consumer_id))
        selected: list[Message] = []
        for message in sorted(
            message_pool,
            key=lambda candidate: (candidate.producer_id, candidate.message_id),
        ):
            if message.round_index != context.round_index:
                raise ValueError(
                    f"Message {message.message_id!r} is from round "
                    f"{message.round_index}, not selection round {context.round_index}."
                )
            try:
                originator_kind = self.originator_kind_by_id[message.producer_id]
            except KeyError as error:
                raise ValueError(
                    "No originator kind registered for producer "
                    f"{message.producer_id}."
                ) from error

            if context.exclude_self_messages and message.producer_id == consumer_id:
                continue

            if originator_kind is OriginatorKind.PRESS:
                probability = (
                    self.press_delivery_probability_by_recipient_kind[recipient_kind]
                )
                if rng.random() < probability:
                    selected.append(message)
            elif (
                recipient_kind is RecipientKind.ORDINARY
                and message.producer_id in eligible_leaders
            ):
                selected.append(message)

        if len(selected) > context.capacity:
            raise ValueError(
                "SelectionContext.capacity is binding, but attention competition "
                "is omitted from OpinionLeaderMessageSelection V1. Increase capacity "
                "or add an explicit attention-selection mechanism."
            )

        return tuple(
            Exposure(
                round_index=context.round_index,
                consumer_id=consumer_id,
                message=message,
            )
            for message in selected
        )
