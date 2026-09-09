"""One-stage message origination for the opinion-leader case."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, log1p

import numpy as np
from scipy.special import expit

from opinion_model.baseline.message_origination import (
    beta_tail_support_probability,
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
    base_probability = float(base_origination_probability)
    decay = float(interest_decay)
    leader_advantage = float(leader_log_odds_advantage)

    if not isfinite(base_probability) or not 0.0 <= base_probability <= 1.0:
        raise ValueError("base_origination_probability must lie in [0, 1].")
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index <= 0
    ):
        raise ValueError("round_index must be a positive integer.")
    if not isinstance(is_leader, bool):
        raise ValueError("is_leader must be Boolean.")
    if not isfinite(decay) or decay < 0.0:
        raise ValueError("interest_decay must be finite and non-negative.")
    if not isfinite(leader_advantage) or leader_advantage < 0.0:
        raise ValueError(
            "leader_log_odds_advantage must be finite and non-negative."
        )

    if base_probability in (0.0, 1.0):
        return base_probability

    base_log_odds = log(base_probability) - log1p(-base_probability)
    linear_predictor = (
        base_log_odds
        - decay * (round_index - 1)
        + leader_advantage * int(is_leader)
    )
    return float(expit(linear_predictor))


@dataclass(frozen=True)
class OpinionLeaderMessageOrigination:
    """Originate at most one belief-derived message for every adaptive agent.

    All agents use the same rule. Leader status changes origination odds but
    not the conditional stance mapping. A successful origination draw creates
    exactly one message; there is no second posting-probability gate.
    """

    leader_ids: frozenset[int]
    interest_decay: float
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
        if not isfinite(self.interest_decay) or self.interest_decay < 0.0:
            raise ValueError("interest_decay must be finite and non-negative.")
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
            interest_decay=self.interest_decay,
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
