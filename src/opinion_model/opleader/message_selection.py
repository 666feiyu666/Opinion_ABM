"""Deterministic tie-bound message selection for the opinion-leader case."""

from __future__ import annotations

import numpy as np

from opinion_model.core import Exposure, Message, NetworkState, SelectionContext


def select_messages(
    consumer_id: int,
    message_pool: tuple[Message, ...],
    network: NetworkState,
    context: SelectionContext,
    rng: np.random.Generator,
) -> tuple[Exposure, ...]:
    """Deliver every current-round message from a regular social contact.

    Delivery is deterministic, role-independent, and stance-independent. The
    original message is retained in each exposure so downstream aggregation can
    recover the producer identity and stance. Attention competition is outside
    this mechanism, so a binding capacity is rejected rather than applied.
    """
    del rng

    if consumer_id not in network.neighbors_by_agent:
        raise ValueError(f"Consumer {consumer_id} is not registered in the network.")

    network_agent_ids = set(network.neighbors_by_agent)
    eligible_producers = set(network.eligible_producers(consumer_id))
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
        if message.producer_id not in network_agent_ids:
            raise ValueError(
                f"Producer {message.producer_id} is not registered in the network."
            )
        if context.exclude_self_messages and message.producer_id == consumer_id:
            continue
        if message.producer_id in eligible_producers:
            selected.append(message)

    if len(selected) > context.capacity:
        raise ValueError(
            "SelectionContext.capacity is binding, but attention competition "
            "is omitted from opleader message selection. Increase capacity or "
            "add an explicit attention-selection mechanism."
        )

    return tuple(
        Exposure(
            round_index=context.round_index,
            consumer_id=consumer_id,
            message=message,
        )
        for message in selected
    )
