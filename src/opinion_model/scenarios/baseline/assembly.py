"""Component assembly for the integrated baseline scenario."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from opinion_model.scenarios.baseline.config import BaselineConfig
from opinion_model.scenarios.baseline.initialization import (
    BaselineInitializer,
    ValidatedBaselineInitializer,
    _normalize_leader_ids,
)
from opinion_model.shared import DEFAULT_COMPONENTS, ModelComponents


def assemble_baseline_components(
    config: BaselineConfig,
    *,
    initializer: BaselineInitializer,
    leader_ids: Iterable[int],
) -> ModelComponents:
    """Combine implemented mechanisms without choosing experimental values.

    The same normalized leader set is supplied to origination and aggregation.
    Platform selection replaces the tie-only opinion-leader selector because
    the integrated baseline includes platform-mediated reach and attention.
    """
    if not isinstance(config, BaselineConfig):
        raise TypeError("config must be a BaselineConfig.")

    normalized_leader_ids = _normalize_leader_ids(
        leader_ids,
        config.simulation.agent_count,
    )
    validated_initializer = ValidatedBaselineInitializer(
        initializer=initializer,
        leader_ids=normalized_leader_ids,
    )

    return replace(
        DEFAULT_COMPONENTS,
        initializer=validated_initializer,
        message_origination=(
            config.opinion_leader.build_message_origination(normalized_leader_ids)
        ),
        message_selection=config.platform.build_message_selection(),
        message_aggregation=(
            config.opinion_leader.build_message_aggregation(normalized_leader_ids)
        ),
        network_update=config.platform.build_network_update(),
    )


__all__ = ["assemble_baseline_components"]
