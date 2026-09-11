"""One-stage message origination for the opinion-leader case."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from opinion_model.shared.message_origination import (
    beta_tail_support_probability,
    temporal_origination_probability,
)
from opinion_model.core import (
    AgentState,
    Message,
    OriginationContext,
    OriginationOutcome,
)


def origination_probability(
    *,
    base_origination_probability: float,
    round_index: int,
    is_leader: bool,
    interest_decay: float,
    leader_log_odds_advantage: float,
) -> float:
    """Return the analytical probability that one agent originates a message.

    ``base_origination_probability`` is the round-one probability for an
    ordinary agent. Interest decay and leader advantage act additively on the
    log-odds scale.
    """
    if not isinstance(is_leader, bool):
        raise ValueError("is_leader must be Boolean.")
    leader_advantage = float(leader_log_odds_advantage)
    if not isfinite(leader_advantage) or leader_advantage < 0.0:
        raise ValueError(
            "leader_log_odds_advantage must be finite and non-negative."
        )
    return temporal_origination_probability(
        base_origination_probability=base_origination_probability,
        round_index=round_index,
        interest_decay=interest_decay,
        log_odds_shift=leader_advantage * int(is_leader),
    )


@dataclass(frozen=True)
class OpinionLeaderMessageOrigination:
    """Originate at most one belief-derived message for every adaptive agent.

    All agents use the same rule. Leader status changes origination odds but
    not the conditional stance mapping. A successful origination draw creates
    exactly one message; there is no second posting-probability gate.
    """

    leader_ids: frozenset[int]
    leader_log_odds_advantage: float

    def __post_init__(self) -> None:
        normalized_ids = frozenset(self.leader_ids)
        for leader_id in normalized_ids:
            if (
                isinstance(leader_id, bool)
                or not isinstance(leader_id, int)
                or leader_id < 0
            ):
                raise ValueError("Leader IDs must be non-negative integers.")
        if (
            not isfinite(self.leader_log_odds_advantage)
            or self.leader_log_odds_advantage < 0.0
        ):
            raise ValueError(
                "leader_log_odds_advantage must be finite and non-negative."
            )
        object.__setattr__(self, "leader_ids", normalized_ids)

    def __call__(
        self,
        agent_id: int,
        state: AgentState,
        context: OriginationContext,
        origination_rng: np.random.Generator,
        stance_rng: np.random.Generator,
    ) -> OriginationOutcome:
        """Return one origination outcome without mutating the agent state."""
        probability = origination_probability(
            base_origination_probability=context.base_origination_probability,
            round_index=context.round_index,
            is_leader=agent_id in self.leader_ids,
            interest_decay=context.interest_decay,
            leader_log_odds_advantage=self.leader_log_odds_advantage,
        )
        support_probability = beta_tail_support_probability(state)
        did_originate = bool(origination_rng.random() < probability)
        message = None
        if did_originate:
            stance = 1 if stance_rng.random() < support_probability else -1
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
            support_probability=support_probability,
            message=message,
        )
