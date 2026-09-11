"""Component assembly for the opinion-leader-only scenario."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from opinion_model.opleader import select_messages as select_opleader_messages
from opinion_model.scenarios.opleader.config import OpleaderConfig
from opinion_model.scenarios.opleader.initialization import (
    OpleaderInitializer,
    ValidatedOpleaderInitializer,
    _normalize_leader_ids,
)
from opinion_model.shared import DEFAULT_COMPONENTS, ModelComponents


def assemble_opleader_components(
    config: OpleaderConfig,
    *,
    initializer: OpleaderInitializer,
    leader_ids: Iterable[int],
) -> ModelComponents:
    """Compose opinion-leader mechanisms without platform behavior."""
    if not isinstance(config, OpleaderConfig):
        raise TypeError("config must be an OpleaderConfig.")

    normalized_leader_ids = _normalize_leader_ids(
        leader_ids,
        config.simulation.agent_count,
    )
    validated_initializer = ValidatedOpleaderInitializer(
        initializer=initializer,
        leader_ids=normalized_leader_ids,
    )
    return replace(
        DEFAULT_COMPONENTS,
        initializer=validated_initializer,
        message_origination=(
            config.opinion_leader.build_message_origination(normalized_leader_ids)
        ),
        message_selection=select_opleader_messages,
        message_aggregation=(
            config.opinion_leader.build_message_aggregation(normalized_leader_ids)
        ),
    )


__all__ = ["assemble_opleader_components"]
