"""Uniform platform-mediated message selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from opinion_model.core import Exposure, Message, NetworkState, SelectionContext


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
        if consumer_id not in network.neighbors_by_agent:
            raise ValueError(f"Consumer {consumer_id} is not registered in the network.")

        network_agent_ids = set(network.neighbors_by_agent)
        tied_producers = set(network.eligible_producers(consumer_id))
        candidates: list[Message] = []

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
                continue

            if rng.random() < self.out_of_network_availability_probability:
                candidates.append(message)

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

        return tuple(
            Exposure(
                round_index=context.round_index,
                consumer_id=consumer_id,
                message=message,
            )
            for message in selected
        )
