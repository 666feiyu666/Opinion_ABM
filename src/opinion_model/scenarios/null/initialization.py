"""Matched sparse-network and ordinary-belief initialization for null."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from opinion_model.core import AgentState, BetaBelief, NetworkState, WorldState
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


def _initialization_streams(
    rng: np.random.Generator,
) -> tuple[int, np.random.Generator, np.random.Generator]:
    # Draw four child seeds exactly as main@1d8eaac does. The fourth seed is
    # reserved for leader assignment there and intentionally unused in null.
    seeds = rng.integers(0, 2**32, size=4, dtype=np.uint32)
    topology_seed = int(seeds[0])
    return (
        topology_seed,
        np.random.default_rng(int(seeds[1])),
        np.random.default_rng(int(seeds[2])),
    )


def _directed_ba_network(
    agent_count: int,
    network_m: int,
    topology_seed: int,
    edge_direction_rng: np.random.Generator,
) -> NetworkState:
    graph = nx.barabasi_albert_graph(
        agent_count,
        network_m,
        seed=topology_seed,
    )
    followed_by_agent = {agent_id: set() for agent_id in graph.nodes}
    for endpoint_a, endpoint_b in sorted(graph.edges):
        if edge_direction_rng.random() < 0.5:
            consumer_id, producer_id = endpoint_a, endpoint_b
        else:
            consumer_id, producer_id = endpoint_b, endpoint_a
        followed_by_agent[consumer_id].add(producer_id)
    return NetworkState(
        {
            agent_id: tuple(sorted(producer_ids))
            for agent_id, producer_ids in followed_by_agent.items()
        }
    )


def initialize_null(
    config: NullConfig,
    rng: np.random.Generator,
) -> NullInitialization:
    """Create the matched null world before any focal mechanism is applied."""
    initialization = config.initialization
    simulation = config.simulation
    topology_seed, edge_direction_rng, belief_rng = _initialization_streams(rng)
    network = _directed_ba_network(
        simulation.agent_count,
        initialization.network_m,
        topology_seed,
        edge_direction_rng,
    )
    ordinary_means = belief_rng.beta(
        initialization.ordinary_mean_alpha,
        initialization.ordinary_mean_alpha,
        size=simulation.agent_count,
    )
    agents = {
        agent_id: AgentState(
            belief=BetaBelief(
                a=float(mean * initialization.ordinary_concentration),
                b=float((1.0 - mean) * initialization.ordinary_concentration),
            )
        )
        for agent_id, mean in enumerate(ordinary_means)
    }
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
