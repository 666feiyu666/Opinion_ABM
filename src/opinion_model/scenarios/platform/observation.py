"""Round metrics and reconstructable decision tables for platform runs."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from opinion_model.core import NetworkUpdateContext, SelectionContext, WorldState
from opinion_model.platform import (
    PlatformMessageSelection,
    PlatformNetworkUpdate,
    PlatformNetworkUpdateTrace,
    PlatformSelectionTrace,
)
from opinion_model.scenarios.platform.config import PlatformConfig
from opinion_model.shared import RandomStreams, SimulationResult, simulation_frames


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
    isolated_agent_count = 0
    for consumer_id, producer_ids in state.network.neighbors_by_agent.items():
        isolated_agent_count += int(not producer_ids)
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
        "isolated_agent_count": isolated_agent_count,
        "homophily_ratio": homophily,
    }


def _message_pool(round_result) -> tuple:
    return tuple(
        outcome.message
        for outcome in round_result.events.origination_outcomes
        if outcome.message is not None
    )


def _observed_exposures_by_consumer(round_result) -> dict[int, tuple]:
    observed: dict[int, list] = defaultdict(list)
    for exposure in round_result.events.exposures:
        observed[exposure.consumer_id].append(exposure)
    return {
        consumer_id: tuple(
            sorted(
                exposures,
                key=lambda item: (
                    item.message.producer_id,
                    item.message.message_id,
                ),
            )
        )
        for consumer_id, exposures in observed.items()
    }


def _selection_traces_for_round(
    result: SimulationResult,
    round_result,
    rule: PlatformMessageSelection,
) -> dict[int, PlatformSelectionTrace]:
    streams = RandomStreams(result.config.seed)
    context = SelectionContext(
        round_index=round_result.next_state.round_index,
        capacity=result.config.consumption_capacity,
        exclude_self_messages=result.config.exclude_self_messages,
    )
    messages = _message_pool(round_result)
    observed = _observed_exposures_by_consumer(round_result)
    traces = {}
    for consumer_id in sorted(round_result.snapshot.agents):
        trace = rule.select_with_trace(
            consumer_id,
            messages,
            round_result.snapshot.network,
            context,
            streams.selection(context.round_index, consumer_id),
        )
        if trace.exposures != observed.get(consumer_id, ()):
            raise AssertionError(
                "Replayed platform selection does not match recorded exposures."
            )
        traces[consumer_id] = trace
    return traces


def _network_trace_for_round(
    result: SimulationResult,
    round_result,
    rule: PlatformNetworkUpdate,
) -> PlatformNetworkUpdateTrace:
    round_index = round_result.next_state.round_index
    trace = rule.propose_with_trace(
        round_result.snapshot.network,
        round_result.snapshot,
        round_result.events,
        NetworkUpdateContext(round_index),
        RandomStreams(result.config.seed).network(round_index),
    )
    if trace.next_network != round_result.next_state.network:
        raise AssertionError(
            "Replayed platform network update does not match the recorded network."
        )
    return trace


def platform_selection_decision_frame(
    result: SimulationResult,
    rule: PlatformMessageSelection,
    *,
    last_round: int | None = None,
) -> pd.DataFrame:
    """Return every eligible availability and capacity decision."""
    rows = []
    for round_result in result.rounds:
        round_index = round_result.next_state.round_index
        if last_round is not None and round_index > last_round:
            continue
        traces = _selection_traces_for_round(result, round_result, rule)
        for trace in traces.values():
            for decision in trace.decisions:
                rows.append(
                    {
                        "round": decision.round_index,
                        "consumer_id": decision.consumer_id,
                        "message_id": decision.message_id,
                        "producer_id": decision.producer_id,
                        "message_stance": decision.message_stance,
                        "channel": decision.channel,
                        "availability_probability": (
                            decision.availability_probability
                        ),
                        "availability_draw": decision.availability_draw,
                        "available": decision.available,
                        "candidate_pool_size": decision.candidate_pool_size,
                        "capacity_binding": decision.capacity_binding,
                        "retained": decision.retained,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "round",
            "consumer_id",
            "message_id",
            "producer_id",
            "message_stance",
            "channel",
            "availability_probability",
            "availability_draw",
            "available",
            "candidate_pool_size",
            "capacity_binding",
            "retained",
        ],
    )


def platform_network_decision_frame(
    result: SimulationResult,
    rule: PlatformNetworkUpdate,
    *,
    last_round: int | None = None,
) -> pd.DataFrame:
    """Return probabilities, random draws, and outcomes for network decisions."""
    rows = []
    for round_result in result.rounds:
        round_index = round_result.next_state.round_index
        if last_round is not None and round_index > last_round:
            continue
        trace = _network_trace_for_round(result, round_result, rule)
        for decision in trace.decisions:
            opportunity = decision.opportunity
            rows.append(
                {
                    "round": opportunity.round_index,
                    "consumer_id": opportunity.consumer_id,
                    "producer_id": opportunity.producer_id,
                    "action": opportunity.action,
                    "following_degree": opportunity.following_degree,
                    "normalized_degree": opportunity.normalized_degree,
                    "support_probability": opportunity.support_probability,
                    "message_stance": opportunity.message_stance,
                    "alignment": opportunity.alignment,
                    "probability": opportunity.probability,
                    "random_draw": decision.random_draw,
                    "accepted": decision.accepted,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "round",
            "consumer_id",
            "producer_id",
            "action",
            "following_degree",
            "normalized_degree",
            "support_probability",
            "message_stance",
            "alignment",
            "probability",
            "random_draw",
            "accepted",
        ],
    )


def platform_round_metrics(
    result: SimulationResult,
    config: PlatformConfig,
    *,
    extremism_threshold: float,
) -> pd.DataFrame:
    """Return synchronized belief, reach, attention, and network diagnostics."""
    selection_rule = config.platform.build_message_selection()
    network_rule = config.platform.build_network_update()
    rows: list[dict[str, float | int | str]] = []
    cumulative_support = 0
    cumulative_oppose = 0
    initial_row: dict[str, float | int | str] = {
        "seed": config.simulation.seed,
        "condition": "platform",
        "round": 0,
        "message_count": 0,
        "exposure_count": 0,
        "tied_exposure_count": 0,
        "out_of_network_exposure_count": 0,
        "capacity_binding_consumer_count": 0,
        "capacity_binding_rate": 0.0,
        "round_content_balance": float("nan"),
        "cumulative_content_balance": float("nan"),
        "round_exposure_balance": float("nan"),
        "formation_opportunity_count": 0,
        "dissolution_opportunity_count": 0,
        "accepted_addition_count": 0,
        "accepted_removal_count": 0,
    }
    initial_row.update(
        _state_measures(result.initial_state, extremism_threshold)
    )
    rows.append(initial_row)

    for round_result in result.rounds:
        messages = _message_pool(round_result)
        support = sum(message.stance == 1 for message in messages)
        oppose = len(messages) - support
        cumulative_support += support
        cumulative_oppose += oppose

        selection_traces = _selection_traces_for_round(
            result,
            round_result,
            selection_rule,
        )
        selection_decisions = tuple(
            decision
            for trace in selection_traces.values()
            for decision in trace.decisions
        )
        retained = tuple(
            decision for decision in selection_decisions if decision.retained
        )
        tied_exposures = sum(decision.channel == "tie" for decision in retained)
        out_exposures = len(retained) - tied_exposures
        exposed_support = sum(
            decision.message_stance == 1 for decision in retained
        )
        exposed_oppose = len(retained) - exposed_support
        binding_consumers = sum(
            trace.decisions[0].capacity_binding if trace.decisions else False
            for trace in selection_traces.values()
        )

        network_trace = _network_trace_for_round(
            result,
            round_result,
            network_rule,
        )
        formation = tuple(
            decision
            for decision in network_trace.decisions
            if decision.opportunity.action == "tie"
        )
        dissolution = tuple(
            decision
            for decision in network_trace.decisions
            if decision.opportunity.action == "untie"
        )
        row: dict[str, float | int | str] = {
            "seed": config.simulation.seed,
            "condition": "platform",
            "round": round_result.next_state.round_index,
            "message_count": len(messages),
            "exposure_count": len(retained),
            "tied_exposure_count": tied_exposures,
            "out_of_network_exposure_count": out_exposures,
            "capacity_binding_consumer_count": binding_consumers,
            "capacity_binding_rate": (
                binding_consumers / config.simulation.agent_count
            ),
            "round_content_balance": _content_balance(support, oppose),
            "cumulative_content_balance": _content_balance(
                cumulative_support,
                cumulative_oppose,
            ),
            "round_exposure_balance": _content_balance(
                exposed_support,
                exposed_oppose,
            ),
            "formation_opportunity_count": len(formation),
            "dissolution_opportunity_count": len(dissolution),
            "accepted_addition_count": sum(
                decision.accepted for decision in formation
            ),
            "accepted_removal_count": sum(
                decision.accepted for decision in dissolution
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


def platform_transition_ledger(
    result: SimulationResult,
    config: PlatformConfig,
    *,
    last_round: int = 5,
) -> pd.DataFrame:
    """Return one reconstructable start-to-end row per agent and round."""
    if last_round < 1:
        raise ValueError("last_round must be at least one.")
    selection_rule = config.platform.build_message_selection()
    network_rule = config.platform.build_network_update()
    rows = []
    for round_result in result.rounds:
        round_index = round_result.next_state.round_index
        if round_index > last_round:
            continue
        selection_traces = _selection_traces_for_round(
            result,
            round_result,
            selection_rule,
        )
        network_trace = _network_trace_for_round(
            result,
            round_result,
            network_rule,
        )
        decisions_by_consumer: dict[int, list] = defaultdict(list)
        for decision in network_trace.decisions:
            decisions_by_consumer[
                decision.opportunity.consumer_id
            ].append(decision)
        outcomes = {
            outcome.agent_id: outcome
            for outcome in round_result.events.origination_outcomes
        }
        for agent_id in sorted(round_result.snapshot.agents):
            before = round_result.snapshot.agents[agent_id].belief
            after = round_result.next_state.agents[agent_id].belief
            outcome = outcomes[agent_id]
            evidence = round_result.events.evidence_by_agent[agent_id]
            selection_trace = selection_traces[agent_id]
            retained = tuple(
                decision
                for decision in selection_trace.decisions
                if decision.retained
            )
            network_decisions = decisions_by_consumer.get(agent_id, [])
            formation = tuple(
                decision
                for decision in network_decisions
                if decision.opportunity.action == "tie"
            )
            dissolution = tuple(
                decision
                for decision in network_decisions
                if decision.opportunity.action == "untie"
            )
            rows.append(
                {
                    "round": round_index,
                    "agent_id": agent_id,
                    "a_before": before.a,
                    "b_before": before.b,
                    "origination_probability": outcome.origination_probability,
                    "p_support_at_origination": outcome.support_probability,
                    "did_originate": outcome.did_originate,
                    "originated_stance": (
                        outcome.message.stance
                        if outcome.message is not None
                        else np.nan
                    ),
                    "candidate_pool_size": sum(
                        decision.available
                        for decision in selection_trace.decisions
                    ),
                    "capacity_binding": (
                        selection_trace.decisions[0].capacity_binding
                        if selection_trace.decisions
                        else False
                    ),
                    "tied_exposure_count": sum(
                        decision.channel == "tie" for decision in retained
                    ),
                    "out_of_network_exposure_count": sum(
                        decision.channel == "out_of_network"
                        for decision in retained
                    ),
                    "n_support": evidence.n_support,
                    "n_oppose": evidence.n_oppose,
                    "weighted_support": evidence.weighted_support,
                    "weighted_oppose": evidence.weighted_oppose,
                    "formation_opportunity_count": len(formation),
                    "accepted_addition_count": sum(
                        decision.accepted for decision in formation
                    ),
                    "dissolution_opportunity_count": len(dissolution),
                    "accepted_removal_count": sum(
                        decision.accepted for decision in dissolution
                    ),
                    "following_degree_before": len(
                        round_result.snapshot.network.eligible_producers(agent_id)
                    ),
                    "following_degree_after": len(
                        round_result.next_state.network.eligible_producers(agent_id)
                    ),
                    "a_after": after.a,
                    "b_after": after.b,
                    "signed_mean_before": before.signed_mean,
                    "signed_mean_after": after.signed_mean,
                }
            )
    return pd.DataFrame(rows)


def platform_diagnostic_frames(
    result: SimulationResult,
    config: PlatformConfig,
    *,
    last_round: int = 5,
) -> dict[str, pd.DataFrame]:
    """Return shared tables plus complete platform traces through a round."""
    if last_round < 1:
        raise ValueError("last_round must be at least one.")
    frames = {
        name: frame.loc[frame["round"] <= last_round].reset_index(drop=True)
        for name, frame in simulation_frames(result).items()
    }
    frames["selection_decisions"] = platform_selection_decision_frame(
        result,
        config.platform.build_message_selection(),
        last_round=last_round,
    )
    frames["network_decisions"] = platform_network_decision_frame(
        result,
        config.platform.build_network_update(),
        last_round=last_round,
    )
    frames["transitions"] = platform_transition_ledger(
        result,
        config,
        last_round=last_round,
    )
    return frames


__all__ = [
    "platform_diagnostic_frames",
    "platform_network_decision_frame",
    "platform_round_metrics",
    "platform_selection_decision_frame",
    "platform_transition_ledger",
]
