"""Matched directed-network and belief initialization for opleader."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from opinion_model.core import (
    AgentState,
    BetaBelief,
    WorldState,
)
from opinion_model.scenarios.matched_initialization import (
    directed_ba_network,
    leader_orientations,
    leaders_by_in_degree,
    matched_initialization_streams,
    normalize_leader_ids as _normalize_leader_ids,
    ordinary_agent_states,
)
from opinion_model.scenarios.opleader.config import (
    OpleaderConfig,
    OpleaderOrientation,
)
from opinion_model.shared import SimulationConfig


OpleaderInitializer = Callable[
    [SimulationConfig, np.random.Generator],
    WorldState,
]


@dataclass(frozen=True)
class OpleaderInitialization:
    """One resolved round-zero world and its opinion-leader assignments."""

    state: WorldState
    leader_ids: frozenset[int]
    positive_leader_ids: frozenset[int]
    negative_leader_ids: frozenset[int]
    orientation: OpleaderOrientation

    def __post_init__(self) -> None:
        agent_ids = set(self.state.agents)
        if not self.leader_ids <= agent_ids:
            raise ValueError("Every leader must exist in the initialized state.")
        if self.positive_leader_ids & self.negative_leader_ids:
            raise ValueError("Positive and negative leader sets cannot overlap.")
        if self.positive_leader_ids | self.negative_leader_ids != self.leader_ids:
            raise ValueError("Every leader must receive exactly one orientation.")


@dataclass(frozen=True)
class FixedOpleaderInitializer:
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
                "Fixed opleader state and SimulationConfig agent counts differ."
            )
        return self.state


@dataclass(frozen=True)
class ValidatedOpleaderInitializer:
    """Validate an opleader initializer at the shared-scheduler boundary."""

    initializer: OpleaderInitializer
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
            raise TypeError("An opleader initializer must return a WorldState.")
        if state.round_index != 0:
            raise ValueError("An opleader initializer must return round zero.")
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


def initialize_opleader(
    config: OpleaderConfig,
    orientation: OpleaderOrientation,
    rng: np.random.Generator,
) -> OpleaderInitialization:
    """Create one matched initial world for an opinion-leader-only run."""
    initialization = config.initialization
    simulation = config.simulation
    streams = matched_initialization_streams(rng)
    network = directed_ba_network(
        simulation.agent_count,
        initialization.network_m,
        streams.topology_seed,
        streams.edge_direction,
    )
    leader_ids = leaders_by_in_degree(
        network,
        initialization.leader_count(simulation.agent_count),
    )
    positive_ids, negative_ids = leader_orientations(
        leader_ids,
        orientation,
        streams.leader_assignment,
    )

    agents = ordinary_agent_states(
        simulation.agent_count,
        initialization.ordinary_mean_alpha,
        initialization.ordinary_concentration,
        streams.ordinary_belief,
    )
    positive_belief = BetaBelief(
        initialization.leader_positive_a,
        initialization.leader_positive_b,
    )
    negative_belief = BetaBelief(
        initialization.leader_positive_b,
        initialization.leader_positive_a,
    )
    for agent_id in positive_ids:
        agents[agent_id] = AgentState(belief=positive_belief)
    for agent_id in negative_ids:
        agents[agent_id] = AgentState(belief=negative_belief)

    state = WorldState(round_index=0, agents=agents, network=network)
    return OpleaderInitialization(
        state=state,
        leader_ids=frozenset(leader_ids),
        positive_leader_ids=positive_ids,
        negative_leader_ids=negative_ids,
        orientation=orientation,
    )


__all__ = [
    "FixedOpleaderInitializer",
    "OpleaderInitialization",
    "OpleaderInitializer",
    "ValidatedOpleaderInitializer",
    "initialize_opleader",
]
