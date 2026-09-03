"""Message-production rule for the coupled null baseline."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_distribution

from opinion_model.core import (
    AgentState,
    Message,
    ProductionContext,
    ProductionOutcome,
)


def support_probability(state: AgentState) -> float:
    belief = state.belief
    return float(beta_distribution.sf(0.5, belief.a, belief.b))


def produce_message(
    agent_id: int,
    state: AgentState,
    context: ProductionContext,
    posting_rng: np.random.Generator,
    stance_rng: np.random.Generator,
) -> ProductionOutcome:
    """Decide whether to post, then draw stance conditional on posting."""
    p_support = support_probability(state)
    did_post = bool(posting_rng.random() < context.post_probability)
    message = None
    if did_post:
        stance = 1 if stance_rng.random() < p_support else -1
        message = Message(
            message_id=f"r{context.round_index}:a{agent_id}",
            round_index=context.round_index,
            producer_id=agent_id,
            stance=stance,
        )
    return ProductionOutcome(
        round_index=context.round_index,
        agent_id=agent_id,
        did_post=did_post,
        post_probability=context.post_probability,
        support_probability=p_support,
        message=message,
    )
