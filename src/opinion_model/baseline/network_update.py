"""Static-network identity rule for the coupled null baseline."""

from __future__ import annotations

import numpy as np

from opinion_model.core import (
    NetworkState,
    NetworkUpdateContext,
    RoundEvents,
    WorldState,
)


def propose_static_network(
    network: NetworkState,
    snapshot: WorldState,
    events: RoundEvents,
    context: NetworkUpdateContext,
    rng: np.random.Generator,
) -> NetworkState:
    """Return the current network unchanged for use in the next round."""
    del snapshot, events, context, rng
    return network
