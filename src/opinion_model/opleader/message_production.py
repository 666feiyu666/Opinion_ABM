"""Message-production mechanisms for the opinion-leader case."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import numpy as np

from opinion_model.baseline.message_production import (
    support_probability as confidence_sensitive_support_probability,
)
from opinion_model.core import (
    AgentState,
    Message,
    ProductionContext,
    ProductionOutcome,
)


SupportProbabilityRule = Callable[[AgentState], float]


def posterior_mean_support_probability(state: AgentState) -> float:
    """Return the Beta posterior mean as the probability of a support message."""
    return state.belief.mean


@dataclass(frozen=True)
class LeaderMessageProduction:
    """Let eligible leaders post with a constant, state-independent probability.

    ``leader_ids`` controls production eligibility. Agents not in that set are
    silent even when the shared ``ProductionContext`` has a non-zero posting
    probability. Conditional on an eligible leader posting, the injected
    support-probability rule maps its start-of-round private state to the
    probability of a positive message.

    The first probe treats all leaders as already informed. Topic-knowledge
    activation is intentionally outside this mechanism.
    """

    leader_ids: frozenset[int]
    support_probability_rule: SupportProbabilityRule = (
        posterior_mean_support_probability
    )

    def __post_init__(self) -> None:
        normalized_ids = frozenset(self.leader_ids)
        for leader_id in normalized_ids:
            if (
                isinstance(leader_id, bool)
                or not isinstance(leader_id, int)
                or leader_id < 0
            ):
                raise ValueError("Leader IDs must be non-negative integers.")
        if not callable(self.support_probability_rule):
            raise ValueError("support_probability_rule must be callable.")
        object.__setattr__(self, "leader_ids", normalized_ids)

    def __call__(
        self,
        agent_id: int,
        state: AgentState,
        context: ProductionContext,
        posting_rng: np.random.Generator,
        stance_rng: np.random.Generator,
    ) -> ProductionOutcome:
        """Return one leader production opportunity without mutating state."""
        p_support = float(self.support_probability_rule(state))
        if not isfinite(p_support) or not 0.0 <= p_support <= 1.0:
            raise ValueError(
                "support_probability_rule must return a finite value in [0, 1]."
            )

        is_leader = agent_id in self.leader_ids
        effective_post_probability = context.post_probability if is_leader else 0.0
        did_post = bool(
            is_leader and posting_rng.random() < effective_post_probability
        )
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
            post_probability=effective_post_probability,
            support_probability=p_support,
            message=message,
        )


def produce_press_message(
    press_id: int,
    round_index: int,
    stance: int,
) -> ProductionOutcome:
    """Produce one deterministic external press message for a round."""
    message = Message(
        message_id=f"r{round_index}:press{press_id}",
        round_index=round_index,
        producer_id=press_id,
        stance=stance,
    )
    return ProductionOutcome(
        round_index=round_index,
        agent_id=press_id,
        did_post=True,
        post_probability=1.0,
        support_probability=1.0 if stance == 1 else 0.0,
        message=message,
    )


@dataclass(frozen=True)
class ScheduledPressMessageProduction:
    """Provide one deterministic press message from an exogenous stance schedule."""

    press_id: int
    stance_by_round: Mapping[int, int]

    def __post_init__(self) -> None:
        if (
            isinstance(self.press_id, bool)
            or not isinstance(self.press_id, int)
            or self.press_id < 0
        ):
            raise ValueError("press_id must be a non-negative integer.")

        normalized_schedule: dict[int, int] = {}
        for round_index, stance in self.stance_by_round.items():
            if (
                isinstance(round_index, bool)
                or not isinstance(round_index, int)
                or round_index <= 0
            ):
                raise ValueError("Press schedule rounds must be positive integers.")
            if stance not in (-1, 1):
                raise ValueError("Press schedule stances must be -1 or +1.")
            normalized_schedule[round_index] = stance

        object.__setattr__(
            self,
            "stance_by_round",
            MappingProxyType(dict(sorted(normalized_schedule.items()))),
        )

    def __call__(self, round_index: int) -> ProductionOutcome:
        """Return the configured press outcome or reject an unspecified round."""
        try:
            stance = self.stance_by_round[round_index]
        except KeyError as error:
            raise ValueError(
                f"No press stance configured for round {round_index}."
            ) from error
        return produce_press_message(self.press_id, round_index, stance)
