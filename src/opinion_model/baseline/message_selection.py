"""Neutral message-selection rule for the coupled null baseline."""

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
    """Select uniformly from eligible messages without inspecting stance."""
    eligible_producers = set(network.eligible_producers(consumer_id))
    eligible = tuple(
        sorted(
            (
                message
                for message in message_pool
                if message.producer_id in eligible_producers
                and not (
                    context.exclude_self_messages
                    and message.producer_id == consumer_id
                )
            ),
            key=lambda message: (message.producer_id, message.message_id),
        )
    )
    if len(eligible) > context.capacity:
        selected_indices = sorted(
            rng.choice(len(eligible), size=context.capacity, replace=False).tolist()
        )
        selected = tuple(eligible[index] for index in selected_indices)
    else:
        selected = eligible
    return tuple(
        Exposure(
            round_index=context.round_index,
            consumer_id=consumer_id,
            message=message,
        )
        for message in selected
    )
