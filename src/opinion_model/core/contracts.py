"""Behavioral contracts owned by the generalized model.

The legacy generic contracts remain available. The explicit callable contracts
separate production opportunities, message exposure, evidence aggregation,
opinion proposals, and network proposals for the coupled baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

from opinion_model.core.entities import (
    AgentState,
    AggregationContext,
    Exposure,
    Message,
    MessageEvidence,
    NetworkState,
    NetworkUpdateContext,
    ProductionContext,
    ProductionOutcome,
    RoundEvents,
    SelectionContext,
    WorldState,
)


AgentStateT = TypeVar("AgentStateT")
FormationContextT = TypeVar("FormationContextT")
InformationT = TypeVar("InformationT")


@runtime_checkable
class InformationFormation(Protocol[AgentStateT, FormationContextT, InformationT]):
    """Form an information item conditional on an external expression event."""

    def form_information(
        self,
        agent_state: AgentStateT,
        context: FormationContextT,
    ) -> InformationT:
        """Return information implied by the supplied state and context."""


@runtime_checkable
class InformationEffect(Protocol[AgentStateT, InformationT]):
    """Apply consumed information to an agent state."""

    def apply_information(
        self,
        agent_state: AgentStateT,
        consumed_information: Sequence[InformationT],
    ) -> AgentStateT:
        """Return the proposed next state without mutating the input state."""


@runtime_checkable
class MessageProduction(Protocol):
    """Give one agent one opportunity to produce zero or one message."""

    def __call__(
        self,
        agent_id: int,
        state: AgentState,
        context: ProductionContext,
        posting_rng: np.random.Generator,
        stance_rng: np.random.Generator,
    ) -> ProductionOutcome: ...


@runtime_checkable
class MessageSelection(Protocol):
    """Map a round message pool to consumed exposures for one agent."""

    def __call__(
        self,
        consumer_id: int,
        message_pool: tuple[Message, ...],
        network: NetworkState,
        context: SelectionContext,
        rng: np.random.Generator,
    ) -> tuple[Exposure, ...]: ...


@runtime_checkable
class MessageAggregation(Protocol):
    """Aggregate consumed exposures into explicit opinion evidence."""

    def __call__(
        self,
        exposures: tuple[Exposure, ...],
        context: AggregationContext,
    ) -> MessageEvidence: ...


@runtime_checkable
class OpinionUpdate(Protocol):
    """Return a proposed private state without mutating the prior state."""

    def __call__(
        self,
        prior: AgentState,
        evidence: MessageEvidence,
    ) -> AgentState: ...


@runtime_checkable
class NetworkUpdate(Protocol):
    """Return a proposed network for the next round."""

    def __call__(
        self,
        network: NetworkState,
        snapshot: WorldState,
        events: RoundEvents,
        context: NetworkUpdateContext,
        rng: np.random.Generator,
    ) -> NetworkState: ...
