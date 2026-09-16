"""Check seed accounting, setting selection, and metadata-driven figure behavior."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opinion_model.visualization.results import (
    Setting, FigureData, SCENARIOS, RAW_ORIENTATIONS, METRICS, EFFECTS,
    select_setting, prepare_trajectories, validate_effect_tables, effects_figure, render_figures,
)
import matplotlib.pyplot as plt


def fixture():
    rows = []
    for scenario in SCENARIOS:
        leader, platform = scenario in ("opleader", "baseline"), scenario in ("platform", "baseline")
        for orientation in RAW_ORIENTATIONS if leader else ("none",):
            for seed in (1, 2):
                for round_index in range(3):
                    value = {"balanced_positive": .4, "balanced_negative": -.2}.get(orientation, .1) * seed
                    rows.append({"run_id": f"{scenario}-{orientation}-{seed}", "scenario": scenario,
                                 "orientation": orientation, "seed": seed, "round": round_index,
                                 "population": 20, "topology": "ba", "leader_share": .1 if leader else np.nan,
                                 "reach": .04 if platform else np.nan,
                                 **{metric: value for metric in METRICS},
                                 "leader_tied_exposure_count": 2 if leader else np.nan,
                                 "leader_out_of_network_exposure_count": 3 if leader else np.nan})
    frame = pd.DataFrame(rows)
    frame.loc[frame["round"] == 0, "cumulative_content_balance"] = np.nan
    return frame


class ResultFigureTests(unittest.TestCase):
    def test_controls_preserved_and_ambiguous_grid_rejected(self):
        frame = fixture()
        setting, selected = select_setting(frame)
        self.assertEqual(setting.population, 20)
        self.assertEqual(set(selected.scenario), set(SCENARIOS))
        other = frame.copy()
        other.population = 30
        combined = pd.concat([frame, other])
        with self.assertRaises(ValueError):
            select_setting(combined)
        _, chosen = select_setting(combined, population=30)
        self.assertEqual(set(chosen.scenario), set(SCENARIOS))
        self.assertEqual(set(chosen.population), {30})

    def test_balanced_is_two_seeds_not_four_and_missing_stays_missing(self):
        result, horizon, _ = prepare_trajectories(fixture(), (1, 2))
        group = result.query("scenario == 'baseline' and orientation == 'balanced' and round == 1")
        self.assertEqual(len(group), 2)
        np.testing.assert_allclose(group.mean_signed_belief, [.1, .2])
        self.assertTrue(result.query("round == 0").cumulative_content_balance.isna().all())
        self.assertTrue(result.query("scenario == 'null'").leader_exposure_count.isna().all())
        self.assertEqual(horizon, 2)

    def test_missing_seed_mirror_or_round_rejected(self):
        frame = fixture()
        for bad in (frame[frame.seed != 2], frame[frame.orientation != "balanced_negative"],
                    frame[~((frame.scenario == "null") & (frame["round"] == 1))]):
            with self.assertRaises(ValueError):
                prepare_trajectories(bad, (1, 2))
        with self.assertRaises(ValueError):
            prepare_trajectories(frame, (1, 2), through_round=50)

    def data(self):
        trajectories, stop, maximum = prepare_trajectories(fixture(), (1, 2))
        points, summary = [], []
        for orientation in ("positive", "negative", "balanced"):
            for effect in EFFECTS:
                points.extend([dict(orientation=orientation, contrast=effect, seed=s, effect=.1*s) for s in (1, 2)])
                summary.append(dict(orientation=orientation, contrast=effect, mean=.15, seed_count=2,
                                    interval_status="complete", ci95_low=-.1, ci95_high=.4))
        return FigureData(Path("."), {"design_status": "pilot", "kind": "main", "expected_seeds": [1, 2],
                                       "analysis_round": 2, "code": {"fingerprint": "fixture-only"}},
                          Setting(20, "ba", .1, .04), trajectories, pd.DataFrame(points), pd.DataFrame(summary), stop, maximum, [])

    def test_pilot_omits_intervals_and_retained_uses_stored_intervals(self):
        data = self.data()
        validate_effect_tables(data.contrasts, data.summary, data.seeds)
        pilot = effects_figure(data)
        retained = effects_figure(replace(data, manifest={**data.manifest, "design_status": "frozen"}))
        self.assertEqual(len(pilot.axes[0].collections), 6)  # seed points + means, no CI artists
        self.assertEqual(len(retained.axes[0].collections), 12)
        self.assertGreater(retained.axes[0].get_xlim()[1], .4)
        plt.close(pilot)
        plt.close(retained)
        bad = data.summary.copy()
        bad.loc[0, "mean"] = 99
        with self.assertRaises(ValueError):
            validate_effect_tables(data.contrasts, bad, data.seeds)

    def test_editable_and_raster_exports_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            index = render_figures(self.data(), folder, dpi=72)
            self.assertTrue(index.exists())
            self.assertEqual(len(list(Path(folder).glob("*.png"))), 4)
            self.assertEqual(len(list(Path(folder).glob("*.pdf"))), 4)
            self.assertEqual(len(list(Path(folder).glob("*.svg"))), 4)
            self.assertIn("PILOT", index.read_text(encoding="utf-8"))
            self.assertEqual(len(list(Path(folder).glob("*figure_manifest.json"))), 1)


if __name__ == "__main__":
    unittest.main()
