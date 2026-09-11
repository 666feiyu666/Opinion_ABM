"""Matched sparse-network and ordinary-belief initialization for null."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from opinion_model.core import WorldState
from opinion_model.scenarios.matched_initialization import (
    directed_ba_network,
    matched_initialization_streams,
    ordinary_agent_states,
)
from opinion_model.scenarios.null.config import NullConfig
from opinion_model.shared import SimulationConfig


NullInitializer = Callable[
    [SimulationConfig, np.random.Generator],
    WorldState,
]


@dataclass(frozen=True)
class NullInitialization:
    """One resolved round-zero world containing only ordinary agents."""

    state: WorldState


@dataclass(frozen=True)
class FixedNullInitializer:
    """Return one already resolved state when the shared scheduler initializes."""

    state: WorldState

    def __call__(
        self,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> WorldState:
        del rng
        if len(self.state.agents) != config.agent_count:
            raise ValueError(
                "Fixed null state and SimulationConfig agent counts differ."
            )
        return self.state


@dataclass(frozen=True)
class ValidatedNullInitializer:
    """Validate a null initializer at the shared-scheduler boundary."""

    initializer: NullInitializer

    def __post_init__(self) -> None:
        if not callable(self.initializer):
            raise TypeError("initializer must be callable.")

    def __call__(
        self,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> WorldState:
        state = self.initializer(config, rng)
        if not isinstance(state, WorldState):
            raise TypeError("A null initializer must return a WorldState.")
        if state.round_index != 0:
            raise ValueError("A null initializer must return the round-zero state.")
        if len(state.agents) != config.agent_count:
            raise ValueError(
                "Initialized agent count must equal SimulationConfig.agent_count."
            )
        return state


def initialize_null(
    config: NullConfig,
    rng: np.random.Generator,
) -> NullInitialization:
    """Create the matched null world before any focal mechanism is applied."""
    initialization = config.initialization
    simulation = config.simulation
    streams = matched_initialization_streams(rng)
    network = directed_ba_network(
        simulation.agent_count,
        initialization.network_m,
        streams.topology_seed,
        streams.edge_direction,
    )
    agents = ordinary_agent_states(
        simulation.agent_count,
        initialization.ordinary_mean_alpha,
        initialization.ordinary_concentration,
        streams.ordinary_belief,
    )
    return NullInitialization(
        state=WorldState(round_index=0, agents=agents, network=network)
    )


__all__ = [
    "FixedNullInitializer",
    "NullInitialization",
    "NullInitializer",
    "ValidatedNullInitializer",
    "initialize_null",
]
