from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_SEED, make_params
from initialization import initialize_model
from main import run_one_round
from metrics import format_round_summary
from utils import set_random_seed


def debug_single_round():
    params = make_params({"T_rounds": 1})
    rng = set_random_seed(DEFAULT_SEED)
    graph, _, agents, _, _ = initialize_model(params, seed=DEFAULT_SEED)
    graph, agents, posts, exposure_sets, summary = run_one_round(
        graph,
        agents,
        params,
        rng,
    )

    print(format_round_summary(1, summary))
    print(f"Posts created in round 1: {len(posts)}")
    print(f"Agents with at least one exposure: {sum(1 for items in exposure_sets.values() if items)}")
    print(agents[['node', 'o_t', 'A_t', 'F_t', 'L']].head(10))
    return graph, agents, posts, exposure_sets, summary


if __name__ == "__main__":
    debug_single_round()
