"""Initialization boundary for the integrated baseline scenario.

The concrete sparse-network, belief, and leader-orientation algorithm remains
open for experiment design. This module currently validates any supplied
initializer so scenario components cannot refer to missing leader IDs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from opinion_model.core import WorldState
from opinion_model.shared import SimulationConfig


BaselineInitializer = Callable[
    [SimulationConfig, np.random.Generator],
    WorldState,
]


def _normalize_leader_ids(
    leader_ids: Iterable[int],
    agent_count: int,
) -> frozenset[int]:
    normalized = frozenset(leader_ids)
    for leader_id in normalized:
        if (
            isinstance(leader_id, bool)
            or not isinstance(leader_id, int)
            or not 0 <= leader_id < agent_count
        ):
            raise ValueError(
                "Every leader ID must be a non-negative integer below agent_count."
            )
    return normalized


@dataclass(frozen=True)
class ValidatedBaselineInitializer:
    """Validate a future baseline initializer at the shared-scheduler boundary."""

    initializer: BaselineInitializer
    leader_ids: frozenset[int]

    def __post_init__(self) -> None:
        if not callable(self.initializer):
            raise TypeError("initializer must be callable.")
        object.__setattr__(self, "leader_ids", frozenset(self.leader_ids))

    def __call__(
        self,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> WorldState:
        normalized_leader_ids = _normalize_leader_ids(
            self.leader_ids,
            config.agent_count,
        )
        state = self.initializer(config, rng)
        if not isinstance(state, WorldState):
            raise TypeError("A baseline initializer must return a WorldState.")
        if state.round_index != 0:
            raise ValueError("A baseline initializer must return the round-zero state.")
        if len(state.agents) != config.agent_count:
            raise ValueError(
                "Initialized agent count must equal SimulationConfig.agent_count."
            )
        missing_leaders = normalized_leader_ids - set(state.agents)
        if missing_leaders:
            raise ValueError(
                f"Initialized state is missing leader IDs {sorted(missing_leaders)}."
            )
        return state


__all__ = ["BaselineInitializer", "ValidatedBaselineInitializer"]
