"""Transparent null boundary conditions for framework-level checks.

Null baselines are declared experimental fixtures, not substantive models of
platform action or information allocation.
"""

from opinion_model.baseline.config import SimulationConfig
from opinion_model.baseline.initialization import (
    belief_from_mean_concentration,
    initialize_baseline,
)
from opinion_model.baseline.message_aggregation import aggregate_messages
from opinion_model.baseline.message_production import (
    produce_message,
    support_probability,
)
from opinion_model.baseline.message_selection import select_messages
from opinion_model.baseline.network_update import propose_static_network
from opinion_model.baseline.observation import simulation_frames
from opinion_model.baseline.opinion_update import propose_opinion_update
from opinion_model.baseline.randomness import RandomStreams
from opinion_model.baseline.simulation import (
    BASELINE_COMPONENTS,
    ModelComponents,
    RoundResult,
    SimulationResult,
    run_round,
    run_simulation,
)

__all__ = [
    "BASELINE_COMPONENTS",
    "ModelComponents",
    "RandomStreams",
    "RoundResult",
    "SimulationConfig",
    "SimulationResult",
    "aggregate_messages",
    "belief_from_mean_concentration",
    "initialize_baseline",
    "produce_message",
    "propose_opinion_update",
    "propose_static_network",
    "run_round",
    "run_simulation",
    "select_messages",
    "simulation_frames",
    "support_probability",
]
