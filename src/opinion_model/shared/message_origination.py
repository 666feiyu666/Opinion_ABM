"""Shared ordinary-agent message-origination rule."""

from __future__ import annotations

from math import isfinite, log, log1p

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_distribution

from opinion_model.core import (
    AgentState,
    Message,
    OriginationContext,
    OriginationOutcome,
)


def beta_tail_support_probability(state: AgentState) -> float:
    """Return the probability mass above the neutral Beta threshold."""
    belief = state.belief
    return float(beta_distribution.sf(0.5, belief.a, belief.b))


def temporal_origination_probability(
    *,
    base_origination_probability: float,
    round_index: int,
    interest_decay: float,
    log_odds_shift: float = 0.0,
) -> float:
    """Return a shared decaying probability with an optional odds shift."""
    base_probability = float(base_origination_probability)
    decay = float(interest_decay)
    shift = float(log_odds_shift)
    if not isfinite(base_probability) or not 0.0 <= base_probability <= 1.0:
        raise ValueError("base_origination_probability must lie in [0, 1].")
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index <= 0
    ):
        raise ValueError("round_index must be a positive integer.")
    if not isfinite(decay) or decay < 0.0:
        raise ValueError("interest_decay must be finite and non-negative.")
    if not isfinite(shift):
        raise ValueError("log_odds_shift must be finite.")
    if base_probability in (0.0, 1.0):
        return base_probability
    base_log_odds = log(base_probability) - log1p(-base_probability)
    return float(expit(base_log_odds - decay * (round_index - 1) + shift))


def originate_message(
    agent_id: int,
    state: AgentState,
    context: OriginationContext,
    origination_rng: np.random.Generator,
    stance_rng: np.random.Generator,
) -> OriginationOutcome:
    """Originate zero or one message, then draw its stance from the belief."""
    p_support = beta_tail_support_probability(state)
    probability = temporal_origination_probability(
        base_origination_probability=context.base_origination_probability,
        round_index=context.round_index,
        interest_decay=context.interest_decay,
    )
    did_originate = bool(origination_rng.random() < probability)
    message = None
    if did_originate:
        stance = 1 if stance_rng.random() < p_support else -1
        message = Message(
            message_id=f"r{context.round_index}:a{agent_id}",
            round_index=context.round_index,
            producer_id=agent_id,
            stance=stance,
        )
    return OriginationOutcome(
        round_index=context.round_index,
        agent_id=agent_id,
        did_originate=did_originate,
        origination_probability=probability,
        support_probability=p_support,
        message=message,
    )
