"""Round metrics and close-reading tables for null simulations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opinion_model.core import WorldState
from opinion_model.shared import SimulationResult, simulation_frames


def _content_balance(support: int, oppose: int) -> float:
    total = support + oppose
    return float((support - oppose) / total) if total else float("nan")


def _state_measures(
    state: WorldState,
    extremism_threshold: float,
) -> dict[str, float | int]:
    signed_beliefs = {
        agent_id: agent.belief.signed_mean
        for agent_id, agent in state.agents.items()
    }
    values = np.fromiter(signed_beliefs.values(), dtype=float)
    same_side = 0
    comparable_edges = 0
    edge_count = 0
    for consumer_id, producer_ids in state.network.neighbors_by_agent.items():
        consumer_value = signed_beliefs[consumer_id]
        for producer_id in producer_ids:
            edge_count += 1
            product = consumer_value * signed_beliefs[producer_id]
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
        "mean_signed_belief": float(values.mean()),
        "mean_absolute_belief": float(np.abs(values).mean()),
        "extremist_ratio": float(
            (np.abs(values) >= extremism_threshold).mean()
        ),
        "edge_count": edge_count,
        "mean_following_degree": float(edge_count / len(state.agents)),
        "homophily_ratio": homophily,
    }


def null_round_metrics(
    result: SimulationResult,
    *,
    seed: int,
    extremism_threshold: float,
) -> pd.DataFrame:
    """Return synchronized state, content, exposure, and network diagnostics."""
    rows: list[dict[str, float | int | str]] = []
    cumulative_support = 0
    cumulative_oppose = 0
    initial_row: dict[str, float | int | str] = {
        "seed": seed,
        "condition": "null",
        "round": 0,
        "message_count": 0,
        "exposure_count": 0,
        "round_content_balance": float("nan"),
        "cumulative_content_balance": float("nan"),
    }
    initial_row.update(
        _state_measures(result.initial_state, extremism_threshold)
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
            "condition": "null",
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
                extremism_threshold,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def null_transition_ledger(
    result: SimulationResult,
    *,
    last_round: int = 5,
) -> pd.DataFrame:
    """Return one reconstructable start-to-end row per agent and round."""
    if last_round < 1:
        raise ValueError("last_round must be at least one.")
    frames = simulation_frames(result)
    originations = frames["origination"].loc[
        lambda frame: frame["round"] <= last_round
    ].copy()
    messages = frames["messages"][
        ["round", "producer_id", "stance"]
    ].rename(columns={"stance": "originated_stance"})
    originations = originations.merge(
        messages,
        how="left",
        left_on=["round", "agent_id"],
        right_on=["round", "producer_id"],
        validate="one_to_one",
    ).drop(columns="producer_id")
    aggregates = frames["aggregates"].loc[
        lambda frame: frame["round"] <= last_round
    ].rename(
        columns={
            "consumer_id": "agent_id",
            "consumed_total": "received_message_count",
        }
    )
    ledger = aggregates.merge(
        originations,
        how="left",
        on=["round", "agent_id"],
        validate="one_to_one",
    )
    columns = [
        "round",
        "agent_id",
        "a_before",
        "b_before",
        "origination_probability",
        "p_support_at_origination",
        "did_originate",
        "originated_stance",
        "received_message_count",
        "n_support",
        "n_oppose",
        "weighted_support",
        "weighted_oppose",
        "a_after",
        "b_after",
        "signed_mean_before",
        "signed_mean_after",
    ]
    return ledger[columns].sort_values(
        ["round", "agent_id"],
        ignore_index=True,
    )


def null_diagnostic_frames(
    result: SimulationResult,
    *,
    last_round: int = 5,
) -> dict[str, pd.DataFrame]:
    """Return all detailed shared tables through the requested round."""
    if last_round < 1:
        raise ValueError("last_round must be at least one.")
    frames = {
        name: frame.loc[frame["round"] <= last_round].reset_index(drop=True)
        for name, frame in simulation_frames(result).items()
    }
    frames["transitions"] = null_transition_ledger(
        result,
        last_round=last_round,
    )
    return frames


__all__ = [
    "null_diagnostic_frames",
    "null_round_metrics",
    "null_transition_ledger",
]
