"""Initialization rule for the coupled null baseline."""

from __future__ import annotations

import numpy as np

from opinion_model.baseline.config import SimulationConfig
from opinion_model.core import AgentState, BetaBelief, NetworkState, WorldState


def belief_from_mean_concentration(mean: float, concentration: float) -> BetaBelief:
    return BetaBelief(mean * concentration, (1.0 - mean) * concentration)


def initialize_baseline(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> WorldState:
    """Create symmetric beliefs and a complete directed network without self-links."""
    del rng  # The current initialization is deterministic; the stream is reserved.
    initial_belief = belief_from_mean_concentration(
        config.initial_mean,
        config.initial_concentration,
    )
    agent_ids = tuple(range(config.agent_count))
    agents = {
        agent_id: AgentState(belief=initial_belief)
        for agent_id in agent_ids
    }
    network = NetworkState(
        {
            consumer_id: tuple(
                producer_id
                for producer_id in agent_ids
                if producer_id != consumer_id
            )
            for consumer_id in agent_ids
        }
    )
    return WorldState(round_index=0, agents=agents, network=network)
