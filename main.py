from __future__ import annotations

from config import DEFAULT_SEED, make_params
from initialization import initialize_model
from metrics import build_history_frame, build_opinion_trajectory_frame, summarize_final_state
from modules.content_creation import create_posts
from modules.diffusion import diffuse_posts
from modules.network_update import adapt_network
from modules.opinion_update import update_opinions
from modules.origination import apply_origination
from modules.round_closure import finalize_round
from utils import set_random_seed, sign_opinion, tanh_mapping
from visualization import prepare_graph_for_visualization


def prepare_agents_for_round(graph, agents, params: dict):
    agents = agents.copy()
    n_agents = params["N"]
    agents["O_t"] = 0
    agents["C_t"] = 0
    agents["F_t"] = [graph.in_degree(i) for i in range(n_agents)]
    agents["s_t"] = agents["o_t"].apply(sign_opinion)
    agents["m_t"] = agents["o_t"].apply(lambda x: tanh_mapping(x, params["kappa"]))
    return agents


def run_one_round(graph, agents, blocks: dict, params: dict, rng):
    graph_current = graph.copy()
    agents_round = prepare_agents_for_round(graph_current, agents, params)
    agents_round = apply_origination(agents_round, params, rng)
    agents_round, posts = create_posts(agents_round, params, rng)
    agents_round, exposure_sets, _ = diffuse_posts(
        graph_current,
        agents_round,
        blocks,
        posts,
        params,
        rng,
    )
    agents_round = update_opinions(agents_round, params)
    graph_next = adapt_network(graph_current, agents_round, exposure_sets, posts, params, rng)
    graph_next, agents_next, summary = finalize_round(
        graph_next,
        agents_round,
        exposure_sets,
        posts,
        params,
    )
    return graph_next, agents_next, posts, exposure_sets, summary


def _capture_opinion_snapshot(agents, round_number: int) -> list[dict]:
    return [
        {
            "round": round_number,
            "node": int(node),
            "opinion": float(opinion),
        }
        for node, opinion in agents["o_t"].items()
    ]


def run_simulation(
    params: dict | None = None,
    seed: int = DEFAULT_SEED,
    rounds: int | None = None,
    track_opinions: bool = False,
):
    sim_params = make_params(params)
    if rounds is not None:
        sim_params["T_rounds"] = rounds

    rng = set_random_seed(seed)
    graph, graph_undirected, agents, blocks, pos = initialize_model(sim_params, seed=seed)

    round_records = []
    all_posts_by_round = {}
    all_exposure_sets_by_round = {}
    opinion_snapshots = []

    if track_opinions:
        opinion_snapshots.extend(_capture_opinion_snapshot(agents, round_number=0))

    for round_number in range(1, sim_params["T_rounds"] + 1):
        graph, agents, posts, exposure_sets, summary = run_one_round(
            graph,
            agents,
            blocks,
            sim_params,
            rng,
        )
        summary["round"] = round_number
        round_records.append(summary)
        all_posts_by_round[round_number] = posts
        all_exposure_sets_by_round[round_number] = exposure_sets
        if track_opinions:
            opinion_snapshots.extend(_capture_opinion_snapshot(agents, round_number=round_number))

    history_df = build_history_frame(round_records)
    graph_updated = prepare_graph_for_visualization(graph, agents)
    final_state = summarize_final_state(graph_updated, agents)

    return {
        "params": sim_params,
        "seed": seed,
        "G": graph,
        "G_initial_undirected": graph_undirected,
        "G_updated": graph_updated,
        "agents": agents,
        "blocks": blocks,
        "pos": pos,
        "history_df": history_df,
        "final_state": final_state,
        "round_records": round_records,
        "all_posts_by_round": all_posts_by_round,
        "all_exposure_sets_by_round": all_exposure_sets_by_round,
        "opinion_trajectory_df": build_opinion_trajectory_frame(opinion_snapshots),
    }


if __name__ == "__main__":
    results = run_simulation()
    print(results["history_df"])
