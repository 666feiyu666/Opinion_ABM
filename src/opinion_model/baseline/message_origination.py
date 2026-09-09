"""Message-origination rule for the coupled null baseline."""

from __future__ import annotations

import numpy as np
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


def originate_message(
    agent_id: int,
    state: AgentState,
    context: OriginationContext,
    origination_rng: np.random.Generator,
    stance_rng: np.random.Generator,
) -> OriginationOutcome:
    """Originate zero or one message, then draw its stance from the belief."""
    p_support = beta_tail_support_probability(state)
    did_originate = bool(
        origination_rng.random() < context.base_origination_probability
    )
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
        origination_probability=context.base_origination_probability,
        support_probability=p_support,
        message=message,
    )
