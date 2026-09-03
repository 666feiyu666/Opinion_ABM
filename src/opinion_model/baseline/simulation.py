"""Synchronous orchestration for the coupled null baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from opinion_model.baseline.config import SimulationConfig
from opinion_model.baseline.initialization import initialize_baseline
from opinion_model.baseline.message_aggregation import aggregate_messages
from opinion_model.baseline.message_production import produce_message
from opinion_model.baseline.message_selection import select_messages
from opinion_model.baseline.network_update import propose_static_network
from opinion_model.baseline.opinion_update import propose_opinion_update
from opinion_model.baseline.randomness import RandomStreams
from opinion_model.core import (
    AgentState,
    AggregationContext,
    MessageAggregation,
    MessageProduction,
    MessageSelection,
    NetworkState,
    NetworkUpdate,
    NetworkUpdateContext,
    OpinionUpdate,
    ProductionContext,
    ProductionOutcome,
    RoundEvents,
    SelectionContext,
    WorldState,
)


Initializer = Callable[[SimulationConfig, np.random.Generator], WorldState]


@dataclass(frozen=True)
class ModelComponents:
    """Replaceable scientific rules called by the shared scheduler."""

    initializer: Initializer
    message_production: MessageProduction
    message_selection: MessageSelection
    message_aggregation: MessageAggregation
    opinion_update: OpinionUpdate
    network_update: NetworkUpdate


@dataclass(frozen=True)
class RoundResult:
    snapshot: WorldState
    events: RoundEvents
    next_state: WorldState


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    initial_state: WorldState
    rounds: tuple[RoundResult, ...]

    @property
    def final_state(self) -> WorldState:
        return self.rounds[-1].next_state if self.rounds else self.initial_state


BASELINE_COMPONENTS = ModelComponents(
    initializer=initialize_baseline,
    message_production=produce_message,
    message_selection=select_messages,
    message_aggregation=aggregate_messages,
    opinion_update=propose_opinion_update,
    network_update=propose_static_network,
)


def _validated_agent_order(
    snapshot: WorldState,
    agent_order: tuple[int, ...] | None,
) -> tuple[int, ...]:
    expected = tuple(sorted(snapshot.agents))
    if agent_order is None:
        return expected
    order = tuple(agent_order)
    if len(order) != len(expected) or set(order) != set(expected):
        raise ValueError("agent_order must contain every agent ID exactly once.")
    return order


def run_round(
    snapshot: WorldState,
    config: SimulationConfig,
    components: ModelComponents,
    random_streams: RandomStreams,
    agent_order: tuple[int, ...] | None = None,
) -> RoundResult:
    """Compute all events and proposals from one snapshot, then commit together."""
    order = _validated_agent_order(snapshot, agent_order)
    round_index = snapshot.round_index + 1
    production_context = ProductionContext(
        round_index=round_index,
        post_probability=config.post_probability,
    )

    outcomes = tuple(
        components.message_production(
            agent_id,
            snapshot.agents[agent_id],
            production_context,
            random_streams.posting(round_index, agent_id),
            random_streams.stance(round_index, agent_id),
        )
        for agent_id in order
    )
    outcomes = tuple(sorted(outcomes, key=lambda outcome: outcome.agent_id))
    message_pool = tuple(
        outcome.message
        for outcome in outcomes
        if outcome.message is not None
    )

    selection_context = SelectionContext(
        round_index=round_index,
        capacity=config.consumption_capacity,
        exclude_self_messages=config.exclude_self_messages,
    )
    aggregation_context = AggregationContext(
        evidence_weight=config.evidence_weight,
    )
    proposed_agents: dict[int, AgentState] = {}
    exposures = []
    evidence_by_agent = {}

    for consumer_id in order:
        consumed = components.message_selection(
            consumer_id,
            message_pool,
            snapshot.network,
            selection_context,
            random_streams.selection(round_index, consumer_id),
        )
        evidence = components.message_aggregation(consumed, aggregation_context)
        proposed_agents[consumer_id] = components.opinion_update(
            snapshot.agents[consumer_id],
            evidence,
        )
        exposures.extend(consumed)
        evidence_by_agent[consumer_id] = evidence

    events = RoundEvents(
        production_outcomes=outcomes,
        exposures=tuple(
            sorted(
                exposures,
                key=lambda exposure: (
                    exposure.consumer_id,
                    exposure.message.producer_id,
                    exposure.message.message_id,
                ),
            )
        ),
        evidence_by_agent=evidence_by_agent,
    )
    next_network = components.network_update(
        snapshot.network,
        snapshot,
        events,
        NetworkUpdateContext(round_index=round_index),
        random_streams.network(round_index),
    )
    next_state = WorldState(
        round_index=round_index,
        agents=proposed_agents,
        network=next_network,
    )
    return RoundResult(snapshot=snapshot, events=events, next_state=next_state)


def run_simulation(
    config: SimulationConfig,
    components: ModelComponents = BASELINE_COMPONENTS,
    agent_order: tuple[int, ...] | None = None,
) -> SimulationResult:
    """Run the fixed synchronous schedule for the configured number of rounds."""
    random_streams = RandomStreams(config.seed)
    initial_state = components.initializer(config, random_streams.initialization())
    snapshot = initial_state
    round_results = []
    for _ in range(config.rounds):
        result = run_round(
            snapshot,
            config,
            components,
            random_streams,
            agent_order,
        )
        round_results.append(result)
        snapshot = result.next_state
    return SimulationResult(
        config=config,
        initial_state=initial_state,
        rounds=tuple(round_results),
    )
