"""Scientific generator contracts and staged batch persistence."""

from dataclasses import replace
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opinion_model.core import NetworkState
from opinion_model.scenarios.topology import TopologyConfig
from opinion_model.scenarios.matched_initialization import directed_network
from opinion_model.experiments.observations import network_structure, structural_metrics
from opinion_model.experiments.planning import load_design, build_plan, validate_plan, resolved_config
from opinion_model.experiments.runner import execute_batch, RUNNERS, file_hash
from opinion_model.visualization.results import load_figure_data


class TopologyTests(unittest.TestCase):
    def test_scenario_toml_loading_and_comparison_rejects_topology_drift(self):
        from opinion_model.scenarios.null.config import load_null_experiment_config
        from opinion_model.scenarios.opleader.config import load_opleader_experiment_config
        from opinion_model.scenarios.platform.config import load_platform_experiment_config
        from opinion_model.scenarios.baseline.config import load_baseline_experiment_config
        loaders = dict(null=load_null_experiment_config, opleader=load_opleader_experiment_config,
                       platform=load_platform_experiment_config, baseline=load_baseline_experiment_config)
        with tempfile.TemporaryDirectory() as temp:
            for case, loader in loaders.items():
                path = Path(temp) / f"{case}.toml"
                path.write_text((ROOT / "configs" / f"{case}.toml").read_text()
                                + '\n[initialization.topology]\nfamily = "ws"\nws_k = 8\nws_rewire_probability = 0.2\n')
                experiment = loader(path)
                config = getattr(experiment, "platform_case" if case == "platform" else case)
                self.assertEqual(config.initialization.topology.family, "ws")
                self.assertEqual(config.initialization.topology.ws_k, 8)
        design = load_design(ROOT / "configs/main_experiment/main_grid.toml")
        template = validate_plan(design, build_plan(design))
        changed = replace(template.null.null, initialization=replace(template.null.null.initialization,
                          topology=TopologyConfig(family="er")))
        with self.assertRaises(ValueError):
            replace(template, null=replace(template.null, null=changed))

    def test_ba_exactly_preserves_previous_algorithm_and_rng(self):
        # Independent reconstruction of the pre-change initializer.
        for n in (12, 30, 500):
            graph = nx.barabasi_albert_graph(n, 3, seed=42)
            rng = np.random.default_rng(19)
            expected = {i: [] for i in range(n)}
            for a, b in sorted(graph.edges):
                consumer, producer = (a, b) if rng.random() < .5 else (b, a)
                expected[consumer].append(producer)
            current_rng = np.random.default_rng(19)
            actual = directed_network(n, 3, 42, current_rng)
            self.assertEqual(actual, NetworkState({i: tuple(sorted(js)) for i, js in expected.items()}))
            self.assertEqual(rng.random(), current_rng.random())

    def test_expected_edges_and_sbm_share(self):
        for n in (500, 503):
            target = 3 * (n - 3)
            er = TopologyConfig(family="er").parameters(n, 3)
            self.assertAlmostEqual(n * (n - 1) / 2 * er["er_probability"], target)
            sbm = TopologyConfig(family="sbm").parameters(n, 3)
            self.assertEqual(sum(sbm["block_sizes"]), n)
            self.assertLessEqual(max(sbm["block_sizes"]) - min(sbm["block_sizes"]), 1)
            within = sum(s * (s - 1) / 2 for s in sbm["block_sizes"])
            between = n * (n - 1) / 2 - within
            self.assertAlmostEqual(within * sbm["sbm_p_in"], .7 * target)
            self.assertAlmostEqual(between * sbm["sbm_p_out"], .3 * target)
        self.assertEqual(TopologyConfig(family="ws").parameters(500, 3)["expected_edges"], 1500)

    def test_invalid_topologies_fail_without_silent_clipping(self):
        for kwargs in ({"family": "other"}, {"ws_k": 5}, {"sbm_blocks": 1},
                       {"ws_rewire_probability": float("nan")}, {"sbm_within_share": 1.2}):
            with self.assertRaises(ValueError):
                TopologyConfig(**kwargs)
        for topology, n in ((TopologyConfig(family="ws"), 6),
                            (TopologyConfig(family="sbm"), 5)):
            with self.assertRaises(ValueError):
                topology.parameters(n, 3)

    def test_graph_invariants_reproducibility_and_density(self):
        for family in ("ba", "er", "ws", "sbm"):
            counts = []
            for seed in range(10):
                config = TopologyConfig(family=family)
                graph = directed_network(500, 3, seed, np.random.default_rng(seed), config)
                again = directed_network(500, 3, seed, np.random.default_rng(seed), config)
                self.assertEqual(graph, again)
                self.assertEqual(set(graph.neighbors_by_agent), set(range(500)))
                edges = {(i, j) for i, js in graph.neighbors_by_agent.items() for j in js}
                self.assertTrue(all(i != j and (j, i) not in edges for i, j in edges))
                counts.append(len(edges))
            target = config.parameters(500, 3)["expected_edges"]
            if family in ("ba", "ws"):
                self.assertEqual(set(counts), {target})
            else:
                # Broad fixed-seed check; legitimate realized edge counts are never forced equal.
                self.assertLess(abs(np.mean(counts) - target), 50)

    def test_hand_calculated_structure_and_empty_edges(self):
        triangle_and_isolate = NetworkState({0: (1, 2), 1: (2,), 2: (), 3: ()})
        metrics = network_structure(triangle_and_isolate)
        self.assertEqual(metrics["structural_top_count"], 1)
        self.assertAlmostEqual(metrics["structural_top_in_degree_share"], 2 / 3)
        self.assertAlmostEqual(metrics["undirected_clustering"], .75)
        self.assertEqual(metrics["largest_weak_component_size"], 3)
        self.assertEqual(metrics["largest_weak_component_fraction"], .75)
        empty = network_structure(NetworkState({0: (), 1: ()}))
        self.assertTrue(np.isnan(empty["structural_top_in_degree_share"]))
        self.assertEqual(empty["undirected_clustering"], 0)
        self.assertEqual(empty["largest_weak_component_size"], 1)

    def test_all_topologies_match_four_scenarios_and_checkpoint_observation(self):
        design = replace(load_design(ROOT / "tests/fixtures/small_experiment.toml"),
                         seeds=(8,), rounds=1, extended_rounds=2, topologies=("ba", "er", "ws", "sbm"))
        plan = build_plan(design)
        template = validate_plan(design, plan)
        for topology in design.topologies:
            initial_networks, ordinary_beliefs = [], []
            for scenario, runner in RUNNERS.items():
                spec = next(s for s in plan.runs.values() if s.topology == topology and s.scenario == scenario
                            and s.orientation in ("positive", "none"))
                config = resolved_config(template, spec, design.topology_settings)
                arguments = {"orientation": "positive"} if spec.leader_share else {}
                run = runner(config, extremism_threshold=.8, **arguments)
                initial_networks.append(run.initialization.state.network)
                leaders = getattr(run.initialization, "leader_ids", frozenset())
                ordinary_beliefs.append({i: a.belief for i, a in run.initialization.state.agents.items() if i not in leaders})
                before = run.simulation_result
                frame = structural_metrics(run, run.round_metrics, 1)
                self.assertTrue(frame.structural_top_count.notna().all())
                pd.testing.assert_frame_equal(frame, structural_metrics(run, run.round_metrics, 1))
                self.assertEqual(before, run.simulation_result)
            self.assertTrue(all(network == initial_networks[0] for network in initial_networks))
            for beliefs in ordinary_beliefs[1:]:
                self.assertTrue(all(belief == ordinary_beliefs[0][i] for i, belief in beliefs.items()))


class StagedExecutionTests(unittest.TestCase):
    def test_reference_pause_reuse_all_topologies_and_resume(self):
        design = replace(load_design(ROOT / "tests/fixtures/small_experiment.toml"),
                         shares=(.1, .2), seeds=(1,), rounds=1, extended_rounds=2,
                         topologies=("ba", "er", "ws", "sbm"),
                         topology_settings=TopologyConfig(ws_rewire_probability=.25, sbm_within_share=.6))
        plan = build_plan(design)
        template = validate_plan(design, plan)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            batch = Path(temp) / "main"
            partial = execute_batch(batch, design, plan, "main", template, reference_only=True, max_runs=1)
            self.assertEqual(partial["status"], "paused")
            self.assertFalse((batch / "main_reference_round1_effect_summary.csv").exists())
            reference = execute_batch(batch, design, plan, "main", template, reference_only=True, resume=True)
            self.assertEqual(reference["status"], "paused")
            self.assertEqual(reference["reference_status"], "complete")
            self.assertEqual(reference["completed_runs"], 10)
            self.assertEqual(reference["required_runs"], 18)
            self.assertFalse((batch / "main_round1_effect_summary.csv").exists())
            outcomes = pd.read_csv(batch / "main_reference_round1_outcomes.csv", keep_default_na=False)
            self.assertEqual(set(outcomes.run_id), set(plan.horizon_ids))
            self.assertEqual(set(outcomes["round"]), {1})
            with self.assertRaises(ValueError):
                load_figure_data(batch)
            figures = load_figure_data(batch, reference_only=True)
            self.assertEqual(figures.manifest["kind"], "main_reference")
            self.assertEqual(figures.endpoint, 1)
            saved = {p: file_hash(p) for p in batch.glob("runs/*/*")}
            for kind in ("topology", "horizon", "reach"):
                support = Path(temp) / kind
                done = execute_batch(support, design, plan, kind, template, source_batch=batch)
                self.assertEqual(done["status"], "complete")
                if kind == "topology":
                    status = pd.read_csv(support / "topology_run_status.csv")
                    self.assertEqual(sum(status.status == "reused"), 10)
                    self.assertEqual(sum(status.status == "completed"), 30)
                    metrics = pd.read_csv(support / "topology_network_snapshots.csv")
                    self.assertEqual(set(metrics.topology), {"ba", "er", "ws", "sbm"})
                    manifests = [json.loads(p.read_text()) for p in support.glob("runs/*/*manifest.json")]
                    ws = next(r for r in manifests if r["spec"]["topology"] == "ws")
                    self.assertEqual(ws["topology_parameters"]["ws_rewire_probability"], .25)
                    checks = pd.read_csv(support / "topology_initialization_checks.csv")
                    self.assertTrue(checks.groupby("topology").initial_network_sha256.nunique().eq(1).all())
                    self.assertTrue(checks.pre_leader_beliefs_sha256.nunique() == 1)
            finished = execute_batch(batch, design, plan, "main", template, resume=True)
            self.assertEqual(finished["status"], "complete")
            self.assertEqual(finished["completed_runs"], 18)
            self.assertEqual(saved, {p: file_hash(p) for p in saved})
            status = pd.read_csv(batch / "main_run_status.csv")
            self.assertEqual(sum(status.status == "resumed"), 10)
            self.assertEqual(sum(status.status == "completed"), 8)


if __name__ == "__main__":
    unittest.main()
