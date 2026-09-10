"""Exposure-driven adaptation of the platform following network."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, log1p
from typing import Literal

import numpy as np
from scipy.special import expit

from opinion_model.shared.message_origination import beta_tail_support_probability
from opinion_model.core import (
    AgentState,
    Exposure,
    NetworkState,
    NetworkUpdateContext,
    RoundEvents,
    WorldState,
)


NetworkAction = Literal["tie", "untie"]


def _strict_probability(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")
    return value


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _alignment(value: float) -> float:
    value = float(value)
    if not isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("alignment must lie in [-1, 1].")
    return value


def normalized_following_degree(following_degree: int, agent_count: int) -> float:
    """Return following degree as a fraction of the maximum possible degree."""
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count < 2:
        raise ValueError("agent_count must be an integer of at least two.")
    if (
        isinstance(following_degree, bool)
        or not isinstance(following_degree, int)
        or not 0 <= following_degree <= agent_count - 1
    ):
        raise ValueError("following_degree must lie between 0 and agent_count - 1.")
    return following_degree / (agent_count - 1)


def belief_message_alignment(state: AgentState, message_stance: int) -> float:
    """Return Beta-tail alignment between a private belief and message stance."""
    if message_stance not in (-1, 1):
        raise ValueError("message_stance must be -1 or +1.")
    support_probability = beta_tail_support_probability(state)
    return float(message_stance * (2.0 * support_probability - 1.0))


def tie_formation_probability(
    *,
    midpoint_probability: float,
    following_degree: int,
    agent_count: int,
    alignment: float,
    degree_log_odds_strength: float,
    alignment_log_odds_strength: float,
) -> float:
    """Return the probability of following an exposed untied producer."""
    midpoint = _strict_probability(midpoint_probability, "midpoint_probability")
    degree_strength = _positive(
        degree_log_odds_strength,
        "degree_log_odds_strength",
    )
    alignment_strength = _positive(
        alignment_log_odds_strength,
        "alignment_log_odds_strength",
    )
    centered_degree = 2.0 * normalized_following_degree(
        following_degree,
        agent_count,
    ) - 1.0
    linear_predictor = (
        log(midpoint)
        - log1p(-midpoint)
        - degree_strength * centered_degree
        + alignment_strength * _alignment(alignment)
    )
    return float(expit(linear_predictor))


def tie_dissolution_probability(
    *,
    midpoint_probability: float,
    following_degree: int,
    agent_count: int,
    alignment: float,
    degree_log_odds_strength: float,
    alignment_log_odds_strength: float,
) -> float:
    """Return the probability of unfollowing an exposed tied producer."""
    midpoint = _strict_probability(midpoint_probability, "midpoint_probability")
    degree_strength = _positive(
        degree_log_odds_strength,
        "degree_log_odds_strength",
    )
    alignment_strength = _positive(
        alignment_log_odds_strength,
        "alignment_log_odds_strength",
    )
    centered_degree = 2.0 * normalized_following_degree(
        following_degree,
        agent_count,
    ) - 1.0
    linear_predictor = (
        log(midpoint)
        - log1p(-midpoint)
        + degree_strength * centered_degree
        - alignment_strength * _alignment(alignment)
    )
    return float(expit(linear_predictor))


@dataclass(frozen=True)
class NetworkDecisionOpportunity:
    """One producer-level tie or untie opportunity and its assigned probability."""

    round_index: int
    consumer_id: int
    producer_id: int
    action: NetworkAction
    following_degree: int
    normalized_degree: float
    support_probability: float
    message_stance: int
    alignment: float
    probability: float


@dataclass(frozen=True)
class PlatformNetworkUpdate:
    """Propose independent exposure-based network changes synchronously."""

    formation_midpoint_probability: float
    formation_degree_log_odds_strength: float
    formation_alignment_log_odds_strength: float
    dissolution_midpoint_probability: float
    dissolution_degree_log_odds_strength: float
    dissolution_alignment_log_odds_strength: float

    def __post_init__(self) -> None:
        _strict_probability(
            self.formation_midpoint_probability,
            "formation_midpoint_probability",
        )
        _positive(
            self.formation_degree_log_odds_strength,
            "formation_degree_log_odds_strength",
        )
        _positive(
            self.formation_alignment_log_odds_strength,
            "formation_alignment_log_odds_strength",
        )
        _strict_probability(
            self.dissolution_midpoint_probability,
            "dissolution_midpoint_probability",
        )
        _positive(
            self.dissolution_degree_log_odds_strength,
            "dissolution_degree_log_odds_strength",
        )
        _positive(
            self.dissolution_alignment_log_odds_strength,
            "dissolution_alignment_log_odds_strength",
        )

    def _validate_inputs(
        self,
        network: NetworkState,
        snapshot: WorldState,
        events: RoundEvents,
        context: NetworkUpdateContext,
    ) -> None:
        if network != snapshot.network:
            raise ValueError("network must be the start-of-round snapshot network.")
        if context.round_index != snapshot.round_index + 1:
            raise ValueError("Network-update round must follow the snapshot round.")

        agent_ids = set(snapshot.agents)
        if set(network.neighbors_by_agent) != agent_ids:
            raise ValueError("Network and snapshot agent IDs must agree.")
        for consumer_id, producers in network.neighbors_by_agent.items():
            unknown = set(producers) - agent_ids
            if unknown:
                raise ValueError(
                    f"Consumer {consumer_id} follows unknown producers {sorted(unknown)}."
                )
        for exposure in events.exposures:
            if exposure.round_index != context.round_index:
                raise ValueError("Every exposure must belong to the update round.")
            if exposure.consumer_id not in agent_ids:
                raise ValueError(
                    f"Consumer {exposure.consumer_id} is not registered in the snapshot."
                )
            if exposure.message.producer_id not in agent_ids:
                raise ValueError(
                    f"Producer {exposure.message.producer_id} is not registered in the snapshot."
                )
            if exposure.consumer_id == exposure.message.producer_id:
                raise ValueError("Self-exposures cannot create network decisions.")

    def opportunities(
        self,
        network: NetworkState,
        snapshot: WorldState,
        events: RoundEvents,
        context: NetworkUpdateContext,
    ) -> tuple[NetworkDecisionOpportunity, ...]:
        """Return deterministic decision inputs before Bernoulli sampling."""
        self._validate_inputs(network, snapshot, events, context)
        agent_count = len(snapshot.agents)
        exposure_by_pair: dict[tuple[int, int], Exposure] = {}

        for exposure in sorted(
            events.exposures,
            key=lambda item: (
                item.consumer_id,
                item.message.producer_id,
                item.message.message_id,
            ),
        ):
            pair = (exposure.consumer_id, exposure.message.producer_id)
            prior = exposure_by_pair.get(pair)
            if prior is not None:
                if prior.message.stance != exposure.message.stance:
                    raise ValueError(
                        "Multiple retained messages from one producer with different "
                        "stances are outside the current one-message origination boundary."
                    )
                continue
            exposure_by_pair[pair] = exposure

        opportunities = []
        consumer_inputs: dict[int, tuple[float, int, float]] = {}
        for (consumer_id, producer_id), exposure in sorted(exposure_by_pair.items()):
            if consumer_id not in consumer_inputs:
                state = snapshot.agents[consumer_id]
                support_probability = beta_tail_support_probability(state)
                following_degree = len(network.eligible_producers(consumer_id))
                normalized_degree = normalized_following_degree(
                    following_degree,
                    agent_count,
                )
                consumer_inputs[consumer_id] = (
                    support_probability,
                    following_degree,
                    normalized_degree,
                )
            support_probability, following_degree, normalized_degree = (
                consumer_inputs[consumer_id]
            )
            alignment = float(
                exposure.message.stance * (2.0 * support_probability - 1.0)
            )
            is_tied = producer_id in network.eligible_producers(consumer_id)

            if is_tied:
                action: NetworkAction = "untie"
                probability = tie_dissolution_probability(
                    midpoint_probability=self.dissolution_midpoint_probability,
                    following_degree=following_degree,
                    agent_count=agent_count,
                    alignment=alignment,
                    degree_log_odds_strength=(
                        self.dissolution_degree_log_odds_strength
                    ),
                    alignment_log_odds_strength=(
                        self.dissolution_alignment_log_odds_strength
                    ),
                )
            else:
                action = "tie"
                probability = tie_formation_probability(
                    midpoint_probability=self.formation_midpoint_probability,
                    following_degree=following_degree,
                    agent_count=agent_count,
                    alignment=alignment,
                    degree_log_odds_strength=self.formation_degree_log_odds_strength,
                    alignment_log_odds_strength=(
                        self.formation_alignment_log_odds_strength
                    ),
                )

            opportunities.append(
                NetworkDecisionOpportunity(
                    round_index=context.round_index,
                    consumer_id=consumer_id,
                    producer_id=producer_id,
                    action=action,
                    following_degree=following_degree,
                    normalized_degree=normalized_degree,
                    support_probability=support_probability,
                    message_stance=exposure.message.stance,
                    alignment=alignment,
                    probability=probability,
                )
            )

        return tuple(opportunities)

    def __call__(
        self,
        network: NetworkState,
        snapshot: WorldState,
        events: RoundEvents,
        context: NetworkUpdateContext,
        rng: np.random.Generator,
    ) -> NetworkState:
        """Sample every opportunity, then commit accepted proposals together."""
        opportunities = self.opportunities(network, snapshot, events, context)
        proposed = {
            consumer_id: set(producers)
            for consumer_id, producers in network.neighbors_by_agent.items()
        }

        for opportunity in opportunities:
            if rng.random() >= opportunity.probability:
                continue
            if opportunity.action == "tie":
                proposed[opportunity.consumer_id].add(opportunity.producer_id)
            else:
                proposed[opportunity.consumer_id].discard(opportunity.producer_id)

        return NetworkState(
            {
                consumer_id: tuple(sorted(producers))
                for consumer_id, producers in proposed.items()
            }
        )
