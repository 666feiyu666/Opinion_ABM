"""Canonical seed-matched initialization shared by all comparison scenarios."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from opinion_model.core import AgentState, BetaBelief, NetworkState


@dataclass(frozen=True)
class MatchedInitializationStreams:
    """Independent streams whose meanings stay fixed across scenarios."""

    topology_seed: int
    edge_direction: np.random.Generator
    ordinary_belief: np.random.Generator
    leader_assignment: np.random.Generator


def matched_initialization_streams(
    rng: np.random.Generator,
) -> MatchedInitializationStreams:
    """Split one initialization stream identically for every scenario."""
    seeds = rng.integers(0, 2**32, size=4, dtype=np.uint32)
    return MatchedInitializationStreams(
        topology_seed=int(seeds[0]),
        edge_direction=np.random.default_rng(int(seeds[1])),
        ordinary_belief=np.random.default_rng(int(seeds[2])),
        leader_assignment=np.random.default_rng(int(seeds[3])),
    )


def directed_ba_network(
    agent_count: int,
    network_m: int,
    topology_seed: int,
    edge_direction_rng: np.random.Generator,
) -> NetworkState:
    """Create one directed information-access edge per undirected BA edge."""
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


def ordinary_agent_states(
    agent_count: int,
    mean_alpha: float,
    concentration: float,
    belief_rng: np.random.Generator,
) -> dict[int, AgentState]:
    """Draw the canonical ordinary-agent beliefs for one matched seed."""
    ordinary_means = belief_rng.beta(
        mean_alpha,
        mean_alpha,
        size=agent_count,
    )
    return {
        agent_id: AgentState(
            belief=BetaBelief(
                a=float(mean * concentration),
                b=float((1.0 - mean) * concentration),
            )
        )
        for agent_id, mean in enumerate(ordinary_means)
    }


def normalize_leader_ids(
    leader_ids: Iterable[int],
    agent_count: int,
) -> frozenset[int]:
    """Validate and freeze leader IDs at scenario assembly boundaries."""
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


def leaders_by_in_degree(
    network: NetworkState,
    leader_count: int,
) -> tuple[int, ...]:
    """Select the requested number of producers with greatest initial reach."""
    in_degree = {agent_id: 0 for agent_id in network.neighbors_by_agent}
    for followed_ids in network.neighbors_by_agent.values():
        for producer_id in followed_ids:
            in_degree[producer_id] += 1
    ranked = sorted(
        in_degree,
        key=lambda agent_id: (-in_degree[agent_id], agent_id),
    )
    return tuple(ranked[:leader_count])


def leader_orientations(
    leader_ids: tuple[int, ...],
    orientation: str,
    assignment_rng: np.random.Generator,
) -> tuple[frozenset[int], frozenset[int]]:
    """Assign matched positive and negative leader sets for one orientation."""
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
        raise ValueError(f"Unknown leader orientation: {orientation!r}")
    return positive_ids, frozenset(leader_ids) - positive_ids


__all__ = [
    "MatchedInitializationStreams",
    "directed_ba_network",
    "leader_orientations",
    "leaders_by_in_degree",
    "matched_initialization_streams",
    "normalize_leader_ids",
    "ordinary_agent_states",
]
