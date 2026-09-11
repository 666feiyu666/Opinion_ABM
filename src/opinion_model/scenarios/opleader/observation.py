"""Round metrics and close-reading tables for opleader simulations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opinion_model.core import WorldState
from opinion_model.scenarios.opleader.config import OpleaderOrientation
from opinion_model.scenarios.opleader.initialization import OpleaderInitialization
from opinion_model.shared import SimulationResult, simulation_frames


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


def opleader_round_metrics(
    result: SimulationResult,
    *,
    seed: int,
    orientation: OpleaderOrientation,
    leader_ids: frozenset[int],
    extremism_threshold: float,
) -> pd.DataFrame:
    """Return synchronized belief, content, role, exposure, and network metrics."""
    rows: list[dict[str, float | int | str]] = []
    cumulative_support = 0
    cumulative_oppose = 0
    initial_row: dict[str, float | int | str] = {
        "seed": seed,
        "orientation": orientation,
        "round": 0,
        "message_count": 0,
        "exposure_count": 0,
        "leader_origination_rate": float("nan"),
        "ordinary_origination_rate": float("nan"),
        "leader_exposure_share": float("nan"),
        "round_content_balance": float("nan"),
        "cumulative_content_balance": float("nan"),
    }
    initial_row.update(
        _state_measures(result.initial_state, leader_ids, extremism_threshold)
    )
    rows.append(initial_row)

    for round_result in result.rounds:
        outcomes = round_result.events.origination_outcomes
        leader_outcomes = [
            outcome for outcome in outcomes if outcome.agent_id in leader_ids
        ]
        ordinary_outcomes = [
            outcome for outcome in outcomes if outcome.agent_id not in leader_ids
        ]
        messages = tuple(
            outcome.message for outcome in outcomes if outcome.message is not None
        )
        support = sum(message.stance == 1 for message in messages)
        oppose = len(messages) - support
        cumulative_support += support
        cumulative_oppose += oppose
        exposures = round_result.events.exposures
        leader_exposures = sum(
            exposure.message.producer_id in leader_ids for exposure in exposures
        )
        row: dict[str, float | int | str] = {
            "seed": seed,
            "orientation": orientation,
            "round": round_result.next_state.round_index,
            "message_count": len(messages),
            "exposure_count": len(exposures),
            "leader_origination_rate": float(
                np.mean([outcome.did_originate for outcome in leader_outcomes])
            ),
            "ordinary_origination_rate": float(
                np.mean([outcome.did_originate for outcome in ordinary_outcomes])
            ),
            "leader_exposure_share": (
                float(leader_exposures / len(exposures))
                if exposures
                else float("nan")
            ),
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


def _role(agent_id: int, initialization: OpleaderInitialization) -> str:
    return "leader" if agent_id in initialization.leader_ids else "ordinary"


def _leader_orientation(
    agent_id: int,
    initialization: OpleaderInitialization,
) -> str:
    if agent_id in initialization.positive_leader_ids:
        return "positive"
    if agent_id in initialization.negative_leader_ids:
        return "negative"
    return "ordinary"


def _add_agent_roles(
    frame: pd.DataFrame,
    id_column: str,
    initialization: OpleaderInitialization,
    *,
    prefix: str = "",
) -> pd.DataFrame:
    result = frame.copy()
    result[f"{prefix}role"] = result[id_column].map(
        lambda value: _role(int(value), initialization)
    )
    result[f"{prefix}orientation"] = result[id_column].map(
        lambda value: _leader_orientation(int(value), initialization)
    )
    return result


def _source_count_table(
    exposures: pd.DataFrame,
    initialization: OpleaderInitialization,
) -> pd.DataFrame:
    columns = [
        "round",
        "agent_id",
        "ordinary_support_messages",
        "ordinary_oppose_messages",
        "leader_support_messages",
        "leader_oppose_messages",
    ]
    if exposures.empty:
        return pd.DataFrame(columns=columns)
    counted = exposures[["round", "consumer_id", "producer_id", "stance"]].copy()
    counted["source_role"] = counted["producer_id"].map(
        lambda value: _role(int(value), initialization)
    )
    counted["stance_name"] = counted["stance"].map(
        {1: "support", -1: "oppose"}
    )
    counted["category"] = counted["source_role"] + "_" + counted["stance_name"] + "_messages"
    table = (
        counted.groupby(["round", "consumer_id", "category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"consumer_id": "agent_id"})
    )
    for column in columns[2:]:
        if column not in table:
            table[column] = 0
    return table[columns]


def opleader_transition_ledger(
    result: SimulationResult,
    initialization: OpleaderInitialization,
    *,
    last_round: int = 5,
) -> pd.DataFrame:
    """Return one source-reconstructable transition per agent and round."""
    if last_round < 1:
        raise ValueError("last_round must be at least one.")
    frames = simulation_frames(result)
    originations = frames["origination"].loc[
        lambda frame: frame["round"] <= last_round
    ].copy()
    messages = frames["messages"][["round", "producer_id", "stance"]].rename(
        columns={"stance": "originated_stance"}
    )
    originations = originations.merge(
        messages,
        how="left",
        left_on=["round", "agent_id"],
        right_on=["round", "producer_id"],
        validate="one_to_one",
    ).drop(columns="producer_id")
    originations = _add_agent_roles(
        originations,
        "agent_id",
        initialization,
    )
    aggregates = frames["aggregates"].loc[
        lambda frame: frame["round"] <= last_round
    ].rename(
        columns={
            "consumer_id": "agent_id",
            "consumed_total": "received_message_count",
        }
    )
    source_counts = _source_count_table(
        frames["exposures"].loc[
            lambda frame: frame["round"] <= last_round
        ],
        initialization,
    )
    ledger = (
        aggregates.merge(
            originations,
            how="left",
            on=["round", "agent_id"],
            validate="one_to_one",
        )
        .merge(
            source_counts,
            how="left",
            on=["round", "agent_id"],
            validate="one_to_one",
        )
    )
    count_columns = [
        "ordinary_support_messages",
        "ordinary_oppose_messages",
        "leader_support_messages",
        "leader_oppose_messages",
    ]
    for column in count_columns:
        ledger[column] = (
            pd.to_numeric(ledger[column], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    columns = [
        "round",
        "agent_id",
        "role",
        "orientation",
        "a_before",
        "b_before",
        "origination_probability",
        "p_support_at_origination",
        "did_originate",
        "originated_stance",
        "received_message_count",
        *count_columns,
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


def opleader_diagnostic_frames(
    result: SimulationResult,
    initialization: OpleaderInitialization,
    *,
    last_round: int = 5,
) -> dict[str, pd.DataFrame]:
    """Return role-enriched shared tables through the requested round."""
    if last_round < 1:
        raise ValueError("last_round must be at least one.")
    frames = {
        name: frame.loc[frame["round"] <= last_round].reset_index(drop=True)
        for name, frame in simulation_frames(result).items()
    }
    frames["states"] = _add_agent_roles(
        frames["states"], "agent_id", initialization
    )
    frames["origination"] = _add_agent_roles(
        frames["origination"], "agent_id", initialization
    )
    frames["messages"] = _add_agent_roles(
        frames["messages"], "producer_id", initialization, prefix="producer_"
    )
    frames["exposures"] = _add_agent_roles(
        frames["exposures"], "consumer_id", initialization, prefix="consumer_"
    )
    frames["exposures"] = _add_agent_roles(
        frames["exposures"], "producer_id", initialization, prefix="producer_"
    )
    frames["aggregates"] = _add_agent_roles(
        frames["aggregates"], "consumer_id", initialization, prefix="consumer_"
    )
    frames["network"] = _add_agent_roles(
        frames["network"], "consumer_id", initialization, prefix="consumer_"
    )
    frames["network"] = _add_agent_roles(
        frames["network"], "producer_id", initialization, prefix="producer_"
    )
    frames["transitions"] = opleader_transition_ledger(
        result,
        initialization,
        last_round=last_round,
    )
    return frames


__all__ = [
    "opleader_diagnostic_frames",
    "opleader_round_metrics",
    "opleader_transition_ledger",
]
