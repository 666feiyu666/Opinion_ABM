"""Matched sparse-network and heterogeneous-belief baseline initialization."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from opinion_model.core import (
    AgentState,
    BetaBelief,
    NetworkState,
    WorldState,
)
from opinion_model.scenarios.baseline.config import (
    BaselineConfig,
    BaselineOrientation,
)
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
class BaselineInitialization:
    """One resolved round-zero world and its opinion-leader assignments."""

    state: WorldState
    leader_ids: frozenset[int]
    positive_leader_ids: frozenset[int]
    negative_leader_ids: frozenset[int]
    orientation: BaselineOrientation

    def __post_init__(self) -> None:
        agent_ids = set(self.state.agents)
        if not self.leader_ids <= agent_ids:
            raise ValueError("Every leader must exist in the initialized state.")
        if self.positive_leader_ids & self.negative_leader_ids:
            raise ValueError("Positive and negative leader sets cannot overlap.")
        if self.positive_leader_ids | self.negative_leader_ids != self.leader_ids:
            raise ValueError("Every leader must receive exactly one orientation.")


@dataclass(frozen=True)
class FixedBaselineInitializer:
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
                "Fixed baseline state and SimulationConfig agent counts differ."
            )
        return self.state


@dataclass(frozen=True)
class ValidatedBaselineInitializer:
    """Validate a baseline initializer at the shared-scheduler boundary."""

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


def _initialization_streams(
    rng: np.random.Generator,
) -> tuple[int, np.random.Generator, np.random.Generator, np.random.Generator]:
    seeds = rng.integers(0, 2**32, size=4, dtype=np.uint32)
    topology_seed = int(seeds[0])
    return (
        topology_seed,
        np.random.default_rng(int(seeds[1])),
        np.random.default_rng(int(seeds[2])),
        np.random.default_rng(int(seeds[3])),
    )


def _directed_ba_network(
    agent_count: int,
    network_m: int,
    topology_seed: int,
    orientation_rng: np.random.Generator,
) -> NetworkState:
    graph = nx.barabasi_albert_graph(
        agent_count,
        network_m,
        seed=topology_seed,
    )
    followed_by_agent = {agent_id: set() for agent_id in graph.nodes}
    for endpoint_a, endpoint_b in sorted(graph.edges):
        if orientation_rng.random() < 0.5:
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


def _leaders_by_in_degree(
    network: NetworkState,
    leader_count: int,
) -> tuple[int, ...]:
    in_degree = {agent_id: 0 for agent_id in network.neighbors_by_agent}
    for followed_ids in network.neighbors_by_agent.values():
        for producer_id in followed_ids:
            in_degree[producer_id] += 1
    ranked = sorted(in_degree, key=lambda agent_id: (-in_degree[agent_id], agent_id))
    return tuple(ranked[:leader_count])


def _leader_orientations(
    leader_ids: tuple[int, ...],
    orientation: BaselineOrientation,
    assignment_rng: np.random.Generator,
) -> tuple[frozenset[int], frozenset[int]]:
    ordered = tuple(
        int(agent_id)
        for agent_id in assignment_rng.permutation(np.asarray(leader_ids))
    )
    if orientation == "positive":
        positive_ids = frozenset(ordered)
    elif orientation == "negative":
        positive_ids = frozenset()
    elif orientation == "balanced_positive":
        positive_ids = frozenset(ordered[: (len(ordered) + 1) // 2])
    elif orientation == "balanced_negative":
        positive_ids = frozenset(ordered[(len(ordered) + 1) // 2 :])
    else:
        raise ValueError(f"Unknown baseline orientation: {orientation!r}")
    negative_ids = frozenset(leader_ids) - positive_ids
    return positive_ids, negative_ids


def initialize_baseline(
    config: BaselineConfig,
    orientation: BaselineOrientation,
    rng: np.random.Generator,
) -> BaselineInitialization:
    """Create one matched initial world for an integrated orientation case."""
    initialization = config.initialization
    simulation = config.simulation
    topology_seed, orientation_rng, belief_rng, assignment_rng = (
        _initialization_streams(rng)
    )
    network = _directed_ba_network(
        simulation.agent_count,
        initialization.network_m,
        topology_seed,
        orientation_rng,
    )
    leader_ids = _leaders_by_in_degree(
        network,
        initialization.leader_count(simulation.agent_count),
    )
    positive_ids, negative_ids = _leader_orientations(
        leader_ids,
        orientation,
        assignment_rng,
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
    return BaselineInitialization(
        state=state,
        leader_ids=frozenset(leader_ids),
        positive_leader_ids=positive_ids,
        negative_leader_ids=negative_ids,
        orientation=orientation,
    )


__all__ = [
    "BaselineInitialization",
    "BaselineInitializer",
    "FixedBaselineInitializer",
    "ValidatedBaselineInitializer",
    "initialize_baseline",
]
