from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.scenarios.baseline import initialize_baseline
from opinion_model.scenarios.comparison import (
    ComparisonExperimentConfig,
    load_comparison_experiment_config,
    run_comparison_experiment,
)
from opinion_model.scenarios.null import initialize_null, run_null_condition
from opinion_model.scenarios.opleader import initialize_opleader
from opinion_model.scenarios.platform import initialize_platform
from opinion_model.shared import RandomStreams


class ComparisonScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_comparison_experiment_config(
            PROJECT_ROOT / "configs" / "comparison.toml"
        )

    def test_repository_configs_form_the_reviewed_2x2_contract(self) -> None:
        config = self.config
        self.assertEqual(config.experiment_id, "olim2-full-comparison")
        self.assertEqual(config.seeds, tuple(range(20260910, 20260920)))
        simulations = (
            config.null.null.simulation,
            config.opleader.opleader.simulation,
            config.platform.platform_case.simulation,
            config.baseline.baseline.simulation,
        )
        self.assertEqual({value.interest_decay for value in simulations}, {0.02})
        self.assertEqual(
            (
                simulations[0].consumption_capacity,
                simulations[1].consumption_capacity,
            ),
            (499, 499),
        )
        self.assertEqual(
            (
                simulations[2].consumption_capacity,
                simulations[3].consumption_capacity,
            ),
            (10, 10),
        )

    def test_initial_states_are_matched_across_scenarios(self) -> None:
        config = self.config
        seed = config.seeds[0]

        def initialization_rng():
            return RandomStreams(seed).initialization()

        null = initialize_null(config.null.null, initialization_rng()).state
        platform = initialize_platform(
            config.platform.platform_case,
            initialization_rng(),
        ).state
        opleader = initialize_opleader(
            config.opleader.opleader,
            "positive",
            initialization_rng(),
        )
        baseline = initialize_baseline(
            config.baseline.baseline,
            "positive",
            initialization_rng(),
        )

        self.assertEqual(null.network, platform.network)
        self.assertEqual(null.network, opleader.state.network)
        self.assertEqual(null.network, baseline.state.network)
        self.assertEqual(null.agents, platform.agents)
        self.assertEqual(opleader.leader_ids, baseline.leader_ids)
        self.assertEqual(
            opleader.positive_leader_ids,
            baseline.positive_leader_ids,
        )
        for agent_id in set(null.agents) - opleader.leader_ids:
            self.assertEqual(null.agents[agent_id], opleader.state.agents[agent_id])
            self.assertEqual(null.agents[agent_id], baseline.state.agents[agent_id])

    def test_shared_interest_decay_is_active_in_null(self) -> None:
        source = self.config.null.null
        config = replace(
            source,
            simulation=replace(
                source.simulation,
                agent_count=8,
                rounds=2,
                seed=101,
                base_origination_probability=0.2,
                consumption_capacity=7,
            ),
            initialization=replace(source.initialization, network_m=2),
        )
        result = run_null_condition(config, extremism_threshold=0.8)
        by_round = [
            round_result.events.origination_outcomes[0].origination_probability
            for round_result in result.simulation_result.rounds
        ]
        self.assertGreater(by_round[0], by_round[1])

    def test_comparison_runner_aligns_all_scenarios(self) -> None:
        config = self._small_config()
        result = run_comparison_experiment(config)
        self.assertEqual(len(result.round_metrics), 20)
        self.assertEqual(len(result.final_metrics), 8)
        self.assertEqual(len(result.final_summary), 8)
        self.assertEqual(
            set(result.scenario_round_metrics),
            {"null", "opleader", "platform", "baseline"},
        )
        factors = result.round_metrics[
            ["scenario", "opinion_leader_enabled", "platform_enabled"]
        ].drop_duplicates()
        self.assertEqual(len(factors), 4)

    def test_comparison_rejects_shared_parameter_drift(self) -> None:
        platform = self.config.platform
        mismatched_platform = replace(
            platform,
            platform_case=replace(
                platform.platform_case,
                simulation=replace(
                    platform.platform_case.simulation,
                    interest_decay=0.03,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "Shared simulation parameters"):
            replace(self.config, platform=mismatched_platform)

    def _small_config(self) -> ComparisonExperimentConfig:
        source = self.config
        null = replace(
            source.null,
            seeds=(101,),
            null=replace(
                source.null.null,
                simulation=replace(
                    source.null.null.simulation,
                    agent_count=8,
                    rounds=1,
                    seed=101,
                    consumption_capacity=7,
                ),
                initialization=replace(
                    source.null.null.initialization,
                    network_m=2,
                ),
            ),
        )
        opleader = replace(
            source.opleader,
            seeds=(101,),
            opleader=replace(
                source.opleader.opleader,
                simulation=replace(
                    source.opleader.opleader.simulation,
                    agent_count=8,
                    rounds=1,
                    seed=101,
                    consumption_capacity=7,
                ),
                initialization=replace(
                    source.opleader.opleader.initialization,
                    network_m=2,
                    leader_share=0.25,
                ),
            ),
        )
        platform = replace(
            source.platform,
            seeds=(101,),
            platform_case=replace(
                source.platform.platform_case,
                simulation=replace(
                    source.platform.platform_case.simulation,
                    agent_count=8,
                    rounds=1,
                    seed=101,
                    consumption_capacity=3,
                ),
                initialization=replace(
                    source.platform.platform_case.initialization,
                    network_m=2,
                ),
            ),
        )
        baseline = replace(
            source.baseline,
            seeds=(101,),
            baseline=replace(
                source.baseline.baseline,
                simulation=replace(
                    source.baseline.baseline.simulation,
                    agent_count=8,
                    rounds=1,
                    seed=101,
                    consumption_capacity=3,
                ),
                initialization=replace(
                    source.baseline.baseline.initialization,
                    network_m=2,
                    leader_share=0.25,
                ),
            ),
        )
        return replace(
            source,
            null=null,
            opleader=opleader,
            platform=platform,
            baseline=baseline,
        )


if __name__ == "__main__":
    unittest.main()
