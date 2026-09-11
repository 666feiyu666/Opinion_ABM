"""Uniform platform-mediated selection and network-adaptation mechanisms."""

from opinion_model.platform.message_selection import (
    MessageSelectionDecision,
    PlatformMessageSelection,
    PlatformSelectionTrace,
    SelectionChannel,
)
from opinion_model.platform.network_update import (
    NetworkDecision,
    NetworkDecisionOpportunity,
    PlatformNetworkUpdate,
    PlatformNetworkUpdateTrace,
    belief_message_alignment,
    normalized_following_degree,
    tie_dissolution_probability,
    tie_formation_probability,
)
from opinion_model.platform.observation import network_decision_frame

__all__ = [
    "MessageSelectionDecision",
    "NetworkDecision",
    "NetworkDecisionOpportunity",
    "PlatformMessageSelection",
    "PlatformNetworkUpdate",
    "PlatformNetworkUpdateTrace",
    "PlatformSelectionTrace",
    "SelectionChannel",
    "belief_message_alignment",
    "network_decision_frame",
    "normalized_following_degree",
    "tie_dissolution_probability",
    "tie_formation_probability",
]
