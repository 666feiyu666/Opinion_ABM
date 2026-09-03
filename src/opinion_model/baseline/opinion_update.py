"""Beta-belief information effect for the coupled null baseline."""

from __future__ import annotations

from opinion_model.core import AgentState, BetaBelief, MessageEvidence


def propose_opinion_update(
    prior: AgentState,
    evidence: MessageEvidence,
) -> AgentState:
    """Return a proposed state; leave the prior immutable."""
    return AgentState(
        belief=BetaBelief(
            a=prior.belief.a + evidence.weighted_support,
            b=prior.belief.b + evidence.weighted_oppose,
        )
    )
