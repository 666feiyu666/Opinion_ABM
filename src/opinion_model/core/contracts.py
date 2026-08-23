"""Behavioral contracts owned by the generalized model.

The core is called only after an environment has supplied an expression event
or a batch of information that was actually consumed. Whether those events
occur, and how a platform or case produces them, is outside this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable


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
