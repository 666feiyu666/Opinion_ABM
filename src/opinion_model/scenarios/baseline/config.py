"""Value-free configuration schema for the integrated baseline scenario.

The schema names the parameters already required by the implemented opinion-
leader and platform mechanisms. It deliberately supplies no scientific
defaults; parameter values and initialization choices remain pending the
baseline experiment-design discussion.
"""

from __future__ import annotations

from dataclasses import dataclass

from opinion_model.opleader import (
    OpinionLeaderMessageAggregation,
    OpinionLeaderMessageOrigination,
)
from opinion_model.platform import PlatformMessageSelection, PlatformNetworkUpdate
from opinion_model.shared import SimulationConfig


@dataclass(frozen=True)
class OpinionLeaderMechanismConfig:
    """Parameters of the two opinion-leader advantages used by baseline."""

    interest_decay: float
    leader_log_odds_advantage: float
    leader_evidence_multiplier: float

    def __post_init__(self) -> None:
        # Delegate domain validation to the canonical mechanism implementations.
        self.build_message_origination(frozenset())
        self.build_message_aggregation(frozenset())

    def build_message_origination(
        self,
        leader_ids: frozenset[int],
    ) -> OpinionLeaderMessageOrigination:
        return OpinionLeaderMessageOrigination(
            leader_ids=leader_ids,
            interest_decay=self.interest_decay,
            leader_log_odds_advantage=self.leader_log_odds_advantage,
        )

    def build_message_aggregation(
        self,
        leader_ids: frozenset[int],
    ) -> OpinionLeaderMessageAggregation:
        return OpinionLeaderMessageAggregation(
            leader_ids=leader_ids,
            leader_evidence_multiplier=self.leader_evidence_multiplier,
        )


@dataclass(frozen=True)
class PlatformMechanismConfig:
    """Parameters of platform reach and exposure-driven network adaptation."""

    out_of_network_availability_probability: float
    formation_midpoint_probability: float
    formation_degree_log_odds_strength: float
    formation_alignment_log_odds_strength: float
    dissolution_midpoint_probability: float
    dissolution_degree_log_odds_strength: float
    dissolution_alignment_log_odds_strength: float

    def __post_init__(self) -> None:
        # Delegate domain validation to the canonical mechanism implementations.
        self.build_message_selection()
        self.build_network_update()

    def build_message_selection(self) -> PlatformMessageSelection:
        return PlatformMessageSelection(
            self.out_of_network_availability_probability,
        )

    def build_network_update(self) -> PlatformNetworkUpdate:
        return PlatformNetworkUpdate(
            formation_midpoint_probability=self.formation_midpoint_probability,
            formation_degree_log_odds_strength=(
                self.formation_degree_log_odds_strength
            ),
            formation_alignment_log_odds_strength=(
                self.formation_alignment_log_odds_strength
            ),
            dissolution_midpoint_probability=self.dissolution_midpoint_probability,
            dissolution_degree_log_odds_strength=(
                self.dissolution_degree_log_odds_strength
            ),
            dissolution_alignment_log_odds_strength=(
                self.dissolution_alignment_log_odds_strength
            ),
        )


@dataclass(frozen=True)
class BaselineConfig:
    """Resolved scheduler and mechanism values for one baseline run.

    Initialization parameters are intentionally absent until the topology,
    ordinary-agent beliefs, leader selection, and leader orientation are
    specified by the researcher.
    """

    simulation: SimulationConfig
    opinion_leader: OpinionLeaderMechanismConfig
    platform: PlatformMechanismConfig

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("simulation must be a SimulationConfig.")
        if not isinstance(self.opinion_leader, OpinionLeaderMechanismConfig):
            raise TypeError(
                "opinion_leader must be an OpinionLeaderMechanismConfig."
            )
        if not isinstance(self.platform, PlatformMechanismConfig):
            raise TypeError("platform must be a PlatformMechanismConfig.")
