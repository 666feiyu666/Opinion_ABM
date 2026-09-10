"""Round-level diagnostic measures for exploratory baseline runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opinion_model.core import WorldState
from opinion_model.scenarios.baseline.config import BaselineOrientation
from opinion_model.shared import SimulationResult


def _content_balance(support: int, oppose: int) -> float:
    total = support + oppose
    return float((support - oppose) / total) if total else float("nan")


def _state_measures(
    state: WorldState,
    leader_ids: frozenset[int],
    extremism_threshold: float,
) -> dict[str, float | int]:
    signed_beliefs = {
        agent_id: agent.belief.signed_mean
        for agent_id, agent in state.agents.items()
    }
    all_values = np.fromiter(signed_beliefs.values(), dtype=float)
    leader_values = np.asarray(
        [signed_beliefs[agent_id] for agent_id in sorted(leader_ids)],
        dtype=float,
    )
    ordinary_values = np.asarray(
        [
            value
            for agent_id, value in signed_beliefs.items()
            if agent_id not in leader_ids
        ],
        dtype=float,
    )

    same_side = 0
    comparable_edges = 0
    edge_count = 0
    for consumer_id, producer_ids in state.network.neighbors_by_agent.items():
        consumer_value = signed_beliefs[consumer_id]
        for producer_id in producer_ids:
            edge_count += 1
            producer_value = signed_beliefs[producer_id]
            product = consumer_value * producer_value
            if product == 0.0:
                continue
            comparable_edges += 1
            same_side += int(product > 0.0)

    homophily = (
        float(same_side / comparable_edges)
        if comparable_edges
        else float("nan")
    )
    return {
        "mean_signed_belief": float(all_values.mean()),
        "mean_absolute_belief": float(np.abs(all_values).mean()),
        "extremist_ratio": float(
            (np.abs(all_values) >= extremism_threshold).mean()
        ),
        "ordinary_mean_signed_belief": float(ordinary_values.mean()),
        "leader_mean_signed_belief": float(leader_values.mean()),
        "edge_count": edge_count,
        "mean_following_degree": float(edge_count / len(state.agents)),
        "homophily_ratio": homophily,
    }


def baseline_round_metrics(
    result: SimulationResult,
    *,
    seed: int,
    orientation: BaselineOrientation,
    leader_ids: frozenset[int],
    extremism_threshold: float,
) -> pd.DataFrame:
    """Return synchronized state, content, exposure, and network diagnostics."""
    rows: list[dict[str, float | int | str]] = []
    cumulative_support = 0
    cumulative_oppose = 0
    initial_row: dict[str, float | int | str] = {
        "seed": seed,
        "orientation": orientation,
        "round": 0,
        "message_count": 0,
        "exposure_count": 0,
        "round_content_balance": float("nan"),
        "cumulative_content_balance": float("nan"),
    }
    initial_row.update(
        _state_measures(result.initial_state, leader_ids, extremism_threshold)
    )
    rows.append(initial_row)

    for round_result in result.rounds:
        messages = tuple(
            outcome.message
            for outcome in round_result.events.origination_outcomes
            if outcome.message is not None
        )
        support = sum(message.stance == 1 for message in messages)
        oppose = len(messages) - support
        cumulative_support += support
        cumulative_oppose += oppose
        row: dict[str, float | int | str] = {
            "seed": seed,
            "orientation": orientation,
            "round": round_result.next_state.round_index,
            "message_count": len(messages),
            "exposure_count": len(round_result.events.exposures),
            "round_content_balance": _content_balance(support, oppose),
            "cumulative_content_balance": _content_balance(
                cumulative_support,
                cumulative_oppose,
            ),
        }
        row.update(
            _state_measures(
                round_result.next_state,
                leader_ids,
                extremism_threshold,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["baseline_round_metrics"]
