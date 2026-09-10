"""Read-only network-decision tables for platform simulations."""

from __future__ import annotations

import pandas as pd

from opinion_model.shared import SimulationResult
from opinion_model.core import NetworkUpdateContext
from opinion_model.platform.network_update import PlatformNetworkUpdate


DECISION_COLUMNS = [
    "round",
    "consumer_id",
    "producer_id",
    "action",
    "following_degree",
    "normalized_degree",
    "support_probability",
    "message_stance",
    "alignment",
    "probability",
    "accepted",
]


def network_decision_frame(
    result: SimulationResult,
    rule: PlatformNetworkUpdate,
) -> pd.DataFrame:
    """Reconstruct assigned probabilities and realized network decisions."""
    rows = []
    for round_result in result.rounds:
        snapshot = round_result.snapshot
        next_state = round_result.next_state
        context = NetworkUpdateContext(round_index=next_state.round_index)
        opportunities = rule.opportunities(
            snapshot.network,
            snapshot,
            round_result.events,
            context,
        )

        for opportunity in opportunities:
            next_neighbors = set(
                next_state.network.eligible_producers(opportunity.consumer_id)
            )
            accepted = (
                opportunity.producer_id in next_neighbors
                if opportunity.action == "tie"
                else opportunity.producer_id not in next_neighbors
            )
            rows.append(
                {
                    "round": opportunity.round_index,
                    "consumer_id": opportunity.consumer_id,
                    "producer_id": opportunity.producer_id,
                    "action": opportunity.action,
                    "following_degree": opportunity.following_degree,
                    "normalized_degree": opportunity.normalized_degree,
                    "support_probability": opportunity.support_probability,
                    "message_stance": opportunity.message_stance,
                    "alignment": opportunity.alignment,
                    "probability": opportunity.probability,
                    "accepted": accepted,
                }
            )

    return pd.DataFrame(rows, columns=DECISION_COLUMNS)
