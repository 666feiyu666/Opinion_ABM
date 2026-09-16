"""Display aggregation and audit failures that would alter scientific interpretation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from opinion_model.experiments.review import seed_display, descriptive_summary, equal_tables


class StudyReviewTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            dict(population=500,topology="ba",leader_share=.03,reach=.04,scenario="baseline",
                 orientation=orientation,seed=seed,round=50,run_id=f"{seed}-{orientation}",
                 outcome=value,undefined=np.nan)
            for seed,positive,negative in ((1,.8,-.4),(2,.9,-.1))
            for orientation,value in (("balanced_positive",positive),("balanced_negative",negative))])

    def test_mirror_average_precedes_seed_interval(self):
        display=seed_display(self.frame(),["outcome","undefined"])
        np.testing.assert_allclose(display.outcome,[.2,.4])
        self.assertTrue(display.undefined.isna().all())
        self.assertTrue(display.source_run_ids.str.contains(";").all())
        summary=descriptive_summary(display,["outcome","undefined"])
        row=summary.query("metric == 'outcome'").iloc[0]
        self.assertEqual(row.seed_count,2)
        self.assertAlmostEqual(row['mean'],.3)
        self.assertAlmostEqual(row.ci95_high,.3+t.ppf(.975,1)*.1)
        self.assertEqual(summary.query("metric == 'undefined'").iloc[0].finite_seed_count,0)

    def test_missing_mirror_and_duplicate_seed_rejected(self):
        with self.assertRaises(ValueError):
            seed_display(self.frame().iloc[:-1],["outcome"])
        duplicated = self.frame().copy()
        duplicated.loc[1, "orientation"] = "balanced_positive"
        with self.assertRaises(ValueError):
            seed_display(duplicated, ["outcome"])
        display=seed_display(self.frame(),["outcome"])
        with self.assertRaises(ValueError):
            descriptive_summary(pd.concat([display,display.iloc[:1]]),["outcome"])

    def test_audit_table_comparison_rejects_changed_values_and_coverage(self):
        original=pd.DataFrame({"run_id":["null-1","null-2"],"round":[50,50],"effect":[.1,.2]})
        equal_tables(original,original.iloc[::-1],["run_id","round"])
        changed=original.copy();changed.loc[0,"effect"] = .11
        with self.assertRaises(AssertionError):
            equal_tables(original,changed,["run_id","round"])
        with self.assertRaises(AssertionError):
            equal_tables(original,original.iloc[:1],["run_id","round"])


if __name__ == "__main__":
    unittest.main()
