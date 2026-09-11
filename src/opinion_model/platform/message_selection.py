"""Uniform platform-mediated message selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from opinion_model.core import Exposure, Message, NetworkState, SelectionContext


SelectionChannel = Literal["tie", "out_of_network"]


@dataclass(frozen=True)
class MessageSelectionDecision:
    """One non-self message--consumer availability and retention decision."""

    round_index: int
    consumer_id: int
    message_id: str
    producer_id: int
    message_stance: int
    channel: SelectionChannel
    availability_probability: float
    availability_draw: float | None
    available: bool
    candidate_pool_size: int
    capacity_binding: bool
    retained: bool


@dataclass(frozen=True)
class PlatformSelectionTrace:
    """Auditable platform-selection result returned by the canonical rule."""

    exposures: tuple[Exposure, ...]
    decisions: tuple[MessageSelectionDecision, ...]


@dataclass(frozen=True)
class PlatformMessageSelection:
    """Select tied and probabilistically available out-of-network messages.

    Messages from followed producers are always candidates. Every message from
    an untied producer receives an independent Bernoulli availability draw.
    All candidates then compete uniformly for the same processing capacity.
    """

    out_of_network_availability_probability: float

    def __post_init__(self) -> None:
        probability = float(self.out_of_network_availability_probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "out_of_network_availability_probability must lie in [0, 1]."
            )
        object.__setattr__(
            self,
            "out_of_network_availability_probability",
            probability,
        )

    def __call__(
        self,
        consumer_id: int,
        message_pool: tuple[Message, ...],
        network: NetworkState,
        context: SelectionContext,
        rng: np.random.Generator,
    ) -> tuple[Exposure, ...]:
        """Return capacity-limited exposures without inspecting message stance."""
        return self.select_with_trace(
            consumer_id,
            message_pool,
            network,
            context,
            rng,
        ).exposures

    def select_with_trace(
        self,
        consumer_id: int,
        message_pool: tuple[Message, ...],
        network: NetworkState,
        context: SelectionContext,
        rng: np.random.Generator,
    ) -> PlatformSelectionTrace:
        """Return exposures plus the availability and capacity decisions."""
        if consumer_id not in network.neighbors_by_agent:
            raise ValueError(f"Consumer {consumer_id} is not registered in the network.")

        network_agent_ids = set(network.neighbors_by_agent)
        tied_producers = set(network.eligible_producers(consumer_id))
        candidates: list[Message] = []
        provisional: list[
            tuple[Message, SelectionChannel, float, float | None, bool]
        ] = []

        for message in sorted(
            message_pool,
            key=lambda candidate: (candidate.producer_id, candidate.message_id),
        ):
            if message.round_index != context.round_index:
                raise ValueError(
                    f"Message {message.message_id!r} is from round "
                    f"{message.round_index}, not selection round {context.round_index}."
                )
            if message.producer_id not in network_agent_ids:
                raise ValueError(
                    f"Producer {message.producer_id} is not registered in the network."
                )
            if context.exclude_self_messages and message.producer_id == consumer_id:
                continue

            if message.producer_id in tied_producers:
                candidates.append(message)
                provisional.append((message, "tie", 1.0, None, True))
                continue

            draw = float(rng.random())
            available = draw < self.out_of_network_availability_probability
            if available:
                candidates.append(message)
            provisional.append(
                (
                    message,
                    "out_of_network",
                    self.out_of_network_availability_probability,
                    draw,
                    available,
                )
            )

        capacity_binding = len(candidates) > context.capacity
        if len(candidates) > context.capacity:
            selected_indices = sorted(
                rng.choice(
                    len(candidates),
                    size=context.capacity,
                    replace=False,
                ).tolist()
            )
            selected = tuple(candidates[index] for index in selected_indices)
        else:
            selected = tuple(candidates)

        exposures = tuple(
            Exposure(
                round_index=context.round_index,
                consumer_id=consumer_id,
                message=message,
            )
            for message in selected
        )
        retained_ids = {exposure.message.message_id for exposure in exposures}
        decisions = tuple(
            MessageSelectionDecision(
                round_index=context.round_index,
                consumer_id=consumer_id,
                message_id=message.message_id,
                producer_id=message.producer_id,
                message_stance=message.stance,
                channel=channel,
                availability_probability=availability_probability,
                availability_draw=availability_draw,
                available=available,
                candidate_pool_size=len(candidates),
                capacity_binding=capacity_binding,
                retained=message.message_id in retained_ids,
            )
            for (
                message,
                channel,
                availability_probability,
                availability_draw,
                available,
            ) in provisional
        )
        return PlatformSelectionTrace(exposures=exposures, decisions=decisions)
