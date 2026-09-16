"""Read-only role/channel observations and initialization provenance."""

from hashlib import sha256
import json

import pandas as pd
import networkx as nx
from math import ceil

STRUCTURE_DEFINITIONS = {
    "timing": "round 0 and post-update states at 10, 30, 50, 75, 100, analysis round and final round",
    "structural_top_in_degree_share": "sum of current top ceil(0.03*N) in-degrees / edge count; NaN for zero edges; ties by ascending agent ID; distinct from fixed leaders",
    "undirected_clustering": "mean local clustering on simple undirected projection, including zero-degree and degree-one nodes as zero",
    "largest_weak_component_size": "node count of largest weakly connected component, including isolates",
    "largest_weak_component_fraction": "largest weak component node count / population",
    "unobserved_rounds": "structural fields are missing outside checkpoints, not zero",
}


def network_structure(network):
    """Post-round structure; top 3% uses ceil and ascending ID to break ties."""
    graph = nx.DiGraph()
    graph.add_nodes_from(network.neighbors_by_agent)
    graph.add_edges_from((i, j) for i, js in network.neighbors_by_agent.items() for j in js)
    n, edges = len(graph), graph.number_of_edges()
    count = max(1, ceil(n * 0.03))
    ranked = sorted(graph, key=lambda i: (-graph.in_degree(i), i))
    largest = max((len(c) for c in nx.weakly_connected_components(graph)), default=0)
    return {
        "structural_top_count": count,
        "structural_top_in_degree_share": sum(graph.in_degree(i) for i in ranked[:count]) / edges if edges else float("nan"),
        "undirected_clustering": nx.average_clustering(graph.to_undirected()) if n else float("nan"),
        "largest_weak_component_size": largest,
        "largest_weak_component_fraction": largest / n if n else float("nan"),
    }


def structural_metrics(run, frame, analysis_round):
    """Observe retained states without drawing randomness or altering dynamics."""
    checkpoints = {0, 10, 30, 50, 75, 100, analysis_round, run.simulation_result.config.rounds}
    states = [run.initialization.state, *(step.next_state for step in run.simulation_result.rounds)]
    rows = [{"round": state.round_index, **network_structure(state.network)}
            for state in states if state.round_index in checkpoints]
    return frame.merge(pd.DataFrame(rows), on="round", how="left", validate="one_to_one")


def initialization_record(run):
    state = run.initialization.state
    leaders = sorted(getattr(run.initialization, "leader_ids", ()))
    edges = sorted((i, j) for i, js in state.network.neighbors_by_agent.items() for j in js)
    degree = {i: 0 for i in state.agents}
    for _, producer in edges:
        degree[producer] += 1
    digest = lambda value: sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {
        "initial_network_sha256": digest(edges),
        "initial_beliefs_sha256": digest([(i, a.belief.a, a.belief.b) for i, a in sorted(state.agents.items())]),
        "leader_ids": leaders,
        "positive_leader_ids": sorted(getattr(run.initialization, "positive_leader_ids", ())),
        "negative_leader_ids": sorted(getattr(run.initialization, "negative_leader_ids", ())),
        "leader_initial_in_degree": {str(i): degree[i] for i in leaders},
        "realized_leader_count": len(leaders),
        "realized_leader_share": len(leaders) / len(state.agents),
        "initial_edge_count": len(edges),
    }


def role_channel_metrics(run):
    """Classify exposure using the network snapshot before that round's update."""
    leaders = getattr(run.initialization, "leader_ids", None)
    if leaders is None:
        return run.round_metrics.copy()
    columns = [f"{role}_{channel}_exposure_count"
               for role in ("leader", "ordinary") for channel in ("tied", "out_of_network")]
    rows = [{"round": 0, **dict.fromkeys(columns, 0)}]
    for step in run.simulation_result.rounds:
        row = {"round": step.next_state.round_index, **dict.fromkeys(columns, 0)}
        for exposure in step.events.exposures:
            producer = exposure.message.producer_id
            role = "leader" if producer in leaders else "ordinary"
            tied = producer in step.snapshot.network.neighbors_by_agent[exposure.consumer_id]
            channel = "tied" if tied else "out_of_network"
            row[f"{role}_{channel}_exposure_count"] += 1
        rows.append(row)
    frame = run.round_metrics.merge(pd.DataFrame(rows), on="round", validate="one_to_one")
    if not (frame[columns].sum(axis=1) == frame.exposure_count).all():
        raise AssertionError("Role/channel exposure counts do not sum to total.")
    return frame
