"""Read-only role/channel observations and initialization provenance."""

from hashlib import sha256
import json

import pandas as pd


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
