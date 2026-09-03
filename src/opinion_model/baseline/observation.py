"""Read-only conversion of baseline events and states to analysis tables."""

from __future__ import annotations

import pandas as pd

from opinion_model.baseline.message_production import support_probability
from opinion_model.baseline.simulation import SimulationResult
from opinion_model.core import WorldState


def _state_rows(state: WorldState) -> list[dict[str, float | int]]:
    rows = []
    for agent_id, agent_state in state.agents.items():
        belief = agent_state.belief
        rows.append(
            {
                "round": state.round_index,
                "agent_id": agent_id,
                "a": belief.a,
                "b": belief.b,
                "mean": belief.mean,
                "signed_mean": belief.signed_mean,
                "concentration": belief.concentration,
                "p_support_message": support_probability(agent_state),
            }
        )
    return rows


def simulation_frames(result: SimulationResult) -> dict[str, pd.DataFrame]:
    """Return reconstructable state, production, message, exposure, and update tables."""
    state_rows = _state_rows(result.initial_state)
    production_rows = []
    message_rows = []
    exposure_rows = []
    aggregate_rows = []
    network_rows = []

    world_states = [result.initial_state]
    for round_result in result.rounds:
        snapshot = round_result.snapshot
        events = round_result.events
        next_state = round_result.next_state
        state_rows.extend(_state_rows(next_state))
        world_states.append(next_state)

        for outcome in events.production_outcomes:
            production_rows.append(
                {
                    "round": outcome.round_index,
                    "agent_id": outcome.agent_id,
                    "did_post": outcome.did_post,
                    "post_probability": outcome.post_probability,
                    "p_support_at_production": outcome.support_probability,
                    "message_id": (
                        outcome.message.message_id
                        if outcome.message is not None
                        else None
                    ),
                }
            )
            if outcome.message is not None:
                message_rows.append(
                    {
                        "round": outcome.message.round_index,
                        "message_id": outcome.message.message_id,
                        "producer_id": outcome.message.producer_id,
                        "stance": outcome.message.stance,
                        "p_support_at_production": outcome.support_probability,
                    }
                )

        for exposure in events.exposures:
            exposure_rows.append(
                {
                    "round": exposure.round_index,
                    "consumer_id": exposure.consumer_id,
                    "message_id": exposure.message.message_id,
                    "producer_id": exposure.message.producer_id,
                    "stance": exposure.message.stance,
                }
            )

        for consumer_id, evidence in events.evidence_by_agent.items():
            before = snapshot.agents[consumer_id].belief
            after = next_state.agents[consumer_id].belief
            aggregate_rows.append(
                {
                    "round": next_state.round_index,
                    "consumer_id": consumer_id,
                    "n_support": evidence.n_support,
                    "n_oppose": evidence.n_oppose,
                    "consumed_total": evidence.total_messages,
                    "weighted_support": evidence.weighted_support,
                    "weighted_oppose": evidence.weighted_oppose,
                    "a_before": before.a,
                    "b_before": before.b,
                    "a_after": after.a,
                    "b_after": after.b,
                    "signed_mean_before": before.signed_mean,
                    "signed_mean_after": after.signed_mean,
                }
            )

    for state in world_states:
        for consumer_id, producers in state.network.neighbors_by_agent.items():
            for producer_id in producers:
                network_rows.append(
                    {
                        "round": state.round_index,
                        "consumer_id": consumer_id,
                        "producer_id": producer_id,
                    }
                )

    return {
        "states": pd.DataFrame(
            state_rows,
            columns=[
                "round",
                "agent_id",
                "a",
                "b",
                "mean",
                "signed_mean",
                "concentration",
                "p_support_message",
            ],
        ),
        "production": pd.DataFrame(
            production_rows,
            columns=[
                "round",
                "agent_id",
                "did_post",
                "post_probability",
                "p_support_at_production",
                "message_id",
            ],
        ),
        "messages": pd.DataFrame(
            message_rows,
            columns=[
                "round",
                "message_id",
                "producer_id",
                "stance",
                "p_support_at_production",
            ],
        ),
        "exposures": pd.DataFrame(
            exposure_rows,
            columns=[
                "round",
                "consumer_id",
                "message_id",
                "producer_id",
                "stance",
            ],
        ),
        "aggregates": pd.DataFrame(
            aggregate_rows,
            columns=[
                "round",
                "consumer_id",
                "n_support",
                "n_oppose",
                "consumed_total",
                "weighted_support",
                "weighted_oppose",
                "a_before",
                "b_before",
                "a_after",
                "b_after",
                "signed_mean_before",
                "signed_mean_after",
            ],
        ),
        "network": pd.DataFrame(
            network_rows,
            columns=["round", "consumer_id", "producer_id"],
        ),
    }
