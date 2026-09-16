from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opinion_model.experiments.analysis import endpoint_metrics, paired_contrasts, summarize_contrasts, OUTCOMES
from opinion_model.experiments.planning import load_design, build_plan, validate_plan, resolved_config, RunSpec
from opinion_model.experiments.runner import execute_batch, execute_one, code_record, run_paths, load_completed
from opinion_model.scenarios.baseline.experiment import run_baseline_condition
from opinion_model.scenarios.platform.experiment import run_platform_condition
from opinion_model.experiments.observations import role_channel_metrics, structural_metrics


class PlanningTests(unittest.TestCase):
    def test_retained_matrix_and_reuse(self):
        design = load_design(ROOT / "configs/main_experiment/main_grid.toml")
        plan = build_plan(design)
        validate_plan(design, plan)
        self.assertEqual(len(plan.runs), 1180)
        self.assertEqual(plan.run_frame().owner.value_counts().to_dict(), {"main": 780, "topology": 300, "reach": 100})
        self.assertEqual(len(plan.horizon_ids), 100)
        self.assertEqual(sum(s.simulation_rounds == 100 for s in plan.runs.values()), 100)
        self.assertEqual(len(plan.required_ids("reach")), 200)
        self.assertEqual(len(plan.required_ids("topology")), 400)
        self.assertTrue(all(plan.runs[i].owner == "main" for i in plan.horizon_ids))
        # Reusing controls across shares must not introduce nominal leader fields.
        controls = [s for s in plan.runs.values() if s.scenario == "null"]
        self.assertEqual(len(controls), 60)
        self.assertTrue(all(s.leader_share is None and s.reach is None for s in controls))

    def test_invalid_designs_rejected(self):
        design = load_design(ROOT / "configs/main_experiment/main_grid.toml")
        for kwargs in ({"seeds": (1, 1)}, {"rounds": 100}, {"reference_share": 0.2}, {"reach": float("nan")}):
            with self.assertRaises(ValueError):
                replace(design, **kwargs)


class AnalysisTests(unittest.TestCase):
    def synthetic(self):
        design = replace(load_design(ROOT / "tests/fixtures/small_experiment.toml"), rounds=50, extended_rounds=100)
        plan = build_plan(design)
        rows = []
        for spec in plan.runs.values():
            if spec.owner != "main":
                continue
            # Effects: without platform=.1, with platform=.1 + seed offset/100.
            y = {"null": 0.0, "opleader": 0.1, "platform": 0.2,
                 "baseline": 0.3 + (spec.seed - 7100) / 100}[spec.scenario]
            for round_index in (0, 50, 100):
                rows.append({"run_id": spec.run_id, "scenario": spec.scenario, "population": spec.population,
                             "topology": spec.topology, "leader_share": spec.leader_share,
                             "reach": spec.reach, "seed": spec.seed, "orientation": spec.orientation,
                             "round": round_index, **{m: y if round_index == 50 else 0.9 for m in OUTCOMES}})
        return design, plan, pd.DataFrame(rows)

    def test_round50_is_default_not_last(self):
        _, _, frame = self.synthetic()
        result = endpoint_metrics(frame)
        self.assertTrue(result["round"].eq(50).all())
        self.assertFalse(result.mean_signed_belief.eq(0.9).any())
        with self.assertRaises(ValueError):
            endpoint_metrics(frame[~((frame.run_id == frame.run_id.iloc[0]) & (frame["round"] == 50))])
        with self.assertRaises(ValueError):
            endpoint_metrics(pd.concat([frame, frame.iloc[:1]]))

    def test_paired_effects_and_t_interval_by_hand(self):
        design, plan, frame = self.synthetic()
        contrasts = paired_contrasts(endpoint_metrics(frame), plan.comparisons.query("experiment == 'main'"))
        selected = contrasts.query("metric == 'mean_signed_belief' and contrast == 'platform_leader_interaction'")
        self.assertEqual(len(selected), 9)  # three seeds x positive, negative, balanced
        np.testing.assert_allclose(selected.query("orientation == 'balanced'").effect, [0.01, 0.02, 0.03])
        summary = summarize_contrasts(contrasts, design.seeds)
        row = summary.query("metric == 'mean_signed_belief' and contrast == 'platform_leader_interaction' and orientation == 'positive'").iloc[0]
        self.assertAlmostEqual(row["mean"], 0.02)
        self.assertAlmostEqual(row["std"], 0.01)
        self.assertAlmostEqual(row.ci95_high, 0.02 + t.ppf(.975, 2) * .01 / np.sqrt(3))
        with self.assertRaises(ValueError):
            summarize_contrasts(contrasts[contrasts.seed != 7101], design.seeds)

    def test_missing_mirror_and_wrong_seed_rejected(self):
        _, plan, frame = self.synthetic()
        comparisons = plan.comparisons.query("experiment == 'main'")
        endpoint = endpoint_metrics(frame)
        with self.assertRaises(ValueError):
            paired_contrasts(endpoint, comparisons[comparisons.orientation != "balanced_negative"])
        endpoint.loc[0, "seed"] = 999
        with self.assertRaises(ValueError):
            paired_contrasts(endpoint, comparisons)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.design = load_design(ROOT / "tests/fixtures/small_experiment.toml")
        self.plan = build_plan(self.design)
        self.template = validate_plan(self.design, self.plan)

    def test_prefix_50_100_and_observer_invariance(self):
        spec = RunSpec("baseline", 20, "ba", .1, "positive", .8, 44, 100, "main")
        config = resolved_config(self.template, spec)
        run100 = run_baseline_condition(config, "positive", extremism_threshold=.8)
        run50 = run_baseline_condition(replace(config, simulation=replace(config.simulation, rounds=50)),
                                       "positive", extremism_threshold=.8)
        self.assertEqual(run100.simulation_result.rounds[:50], run50.simulation_result.rounds)
        pd.testing.assert_frame_equal(run100.round_metrics.iloc[:51].reset_index(drop=True), run50.round_metrics)
        frame = role_channel_metrics(run100)
        self.assertTrue((frame.leader_tied_exposure_count + frame.ordinary_tied_exposure_count == frame.tied_exposure_count).all())
        self.assertTrue((frame.leader_out_of_network_exposure_count + frame.ordinary_out_of_network_exposure_count == frame.out_of_network_exposure_count).all())
        self.assertTrue(frame.edge_count.diff().iloc[1:].eq((frame.accepted_addition_count-frame.accepted_removal_count).iloc[1:]).all())
        # Building the role observer repeatedly cannot mutate the retained states.
        pd.testing.assert_frame_equal(frame, role_channel_metrics(run100))
        observed100 = structural_metrics(run100, frame, 50)
        observed50 = structural_metrics(run50, role_channel_metrics(run50), 50)
        pd.testing.assert_frame_equal(observed100.iloc[:51].reset_index(drop=True), observed50)
        self.assertEqual(set(observed100.loc[observed100.structural_top_count.notna(), "round"]),
                         {0, 10, 30, 50, 75, 100})

    def test_neutral_leaders_give_same_platform_diagnostics(self):
        # Keep the leader initialization neutral too: use a shared initializer through scheduler.
        from opinion_model.scenarios.baseline.assembly import assemble_baseline_components
        from opinion_model.scenarios.baseline.initialization import FixedBaselineInitializer
        from opinion_model.shared import run_simulation
        from opinion_model.scenarios.baseline.observation import baseline_round_metrics
        spec = RunSpec("platform", 30, "ba", None, "none", .3, 44, 3, "main")
        platform_config = resolved_config(self.template, spec)
        p = run_platform_condition(platform_config, extremism_threshold=.8)
        b_config = resolved_config(self.template, replace(spec, scenario="baseline", leader_share=.1, orientation="positive"))
        b_config = replace(b_config, opinion_leader=replace(b_config.opinion_leader, leader_log_odds_advantage=0., leader_evidence_multiplier=1.))
        components = assemble_baseline_components(b_config,
            initializer=FixedBaselineInitializer(p.initialization.state), leader_ids={0})
        result = run_simulation(b_config.simulation, components)
        b = baseline_round_metrics(result, config=b_config, seed=44, orientation="positive", leader_ids=frozenset({0}), extremism_threshold=.8)
        self.assertEqual(result.rounds, p.simulation_result.rounds)
        columns = [c for c in p.round_metrics if c not in ("condition",)]
        pd.testing.assert_frame_equal(b[columns], p.round_metrics[columns])

    def test_all_other_scenarios_keep_their_first_50_rounds(self):
        from opinion_model.scenarios.null.experiment import run_null_condition
        from opinion_model.scenarios.opleader.experiment import run_opleader_condition
        for scenario, runner in (("null", run_null_condition), ("opleader", run_opleader_condition),
                                 ("platform", run_platform_condition)):
            with self.subTest(scenario=scenario):
                spec = RunSpec(scenario, 20, "ba", .1 if scenario == "opleader" else None,
                               "negative" if scenario == "opleader" else "none",
                               .04 if scenario == "platform" else None, 55, 100, "main")
                config = resolved_config(self.template, spec)
                arguments = {"extremism_threshold": .8}
                if scenario == "opleader":
                    arguments["orientation"] = "negative"
                long = runner(config, **arguments)
                short = runner(replace(config, simulation=replace(config.simulation, rounds=50)), **arguments)
                self.assertEqual(long.simulation_result.rounds[:50], short.simulation_result.rounds)
                pd.testing.assert_frame_equal(long.round_metrics.iloc[:51].reset_index(drop=True), short.round_metrics)

    def test_role_classification_uses_pre_update_ties(self):
        from types import SimpleNamespace
        from opinion_model.core import NetworkState
        # Producer 2 becomes tied after exposure: that exposure still came from outside.
        exposure = SimpleNamespace(consumer_id=0, message=SimpleNamespace(producer_id=2))
        step = SimpleNamespace(snapshot=SimpleNamespace(network=NetworkState({0: (1,), 1: (), 2: ()})),
                               next_state=SimpleNamespace(round_index=1, network=NetworkState({0: (1, 2), 1: (), 2: ()})),
                               events=SimpleNamespace(exposures=(exposure,)))
        run = SimpleNamespace(initialization=SimpleNamespace(leader_ids=frozenset({2})),
                              simulation_result=SimpleNamespace(rounds=(step,)),
                              round_metrics=pd.DataFrame({"round": [0, 1], "exposure_count": [0, 1]}))
        result = role_channel_metrics(run).iloc[1]
        self.assertEqual(result.leader_out_of_network_exposure_count, 1)
        self.assertEqual(result.leader_tied_exposure_count, 0)


class PersistenceTests(unittest.TestCase):
    def test_pause_resume_and_support_reuse(self):
        design = replace(load_design(ROOT / "tests/fixtures/small_experiment.toml"),
                         populations=(12,), reference_population=12, seeds=(1, 2), rounds=1, extended_rounds=2)
        plan = build_plan(design)
        template = validate_plan(design, plan)
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp) / "pilot_main"
            first = execute_batch(batch, design, plan, "main", template, max_runs=1)
            self.assertEqual(first["status"], "paused")
            path = next(batch.glob("runs/*/*round_metrics.csv"))
            before = path.read_bytes()
            done = execute_batch(batch, design, plan, "main", template, resume=True)
            self.assertEqual(done["status"], "complete")
            self.assertEqual(path.read_bytes(), before)
            status = pd.read_csv(batch / "main_run_status.csv")
            self.assertEqual(sum(status.status == "resumed"), 1)
            reach = execute_batch(Path(temp) / "reach", design, plan, "reach", template, source_batch=batch)
            self.assertEqual(reach["status"], "complete")
            statuses = pd.read_csv(Path(temp) / "reach/reach_run_status.csv")
            self.assertEqual(sum(statuses.status == "reused"), 20)
            horizon = execute_batch(Path(temp) / "horizon", design, plan, "horizon", template, source_batch=batch)
            self.assertEqual(horizon["status"], "complete")
            with self.assertRaises(ValueError):
                execute_batch(batch, design, plan, "main", template)
            # A corrupt persisted output is not silently accepted or overwritten.
            path.write_text("broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                execute_batch(batch, design, plan, "main", template, resume=True)

    def test_formal_provisional_and_missing_topology_sources_fail_before_running(self):
        design = replace(load_design(ROOT / "configs/main_experiment/main_grid.toml"), status="provisional")
        plan = build_plan(design)
        template = validate_plan(design, plan)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                execute_batch(Path(temp), design, plan, "main", template)
            with self.assertRaises(ValueError):
                execute_batch(Path(temp), replace(design, status="pilot"), plan, "topology", template)


if __name__ == "__main__":
    unittest.main()
