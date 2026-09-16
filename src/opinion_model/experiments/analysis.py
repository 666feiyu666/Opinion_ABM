"""Explicit endpoints and paired seed contrasts, with no pooled-agent inference."""

import numpy as np
import pandas as pd
from scipy.stats import t

OUTCOMES = ("mean_signed_belief", "mean_absolute_belief", "extremist_ratio",
            "cumulative_content_balance", "edge_count", "mean_following_degree", "homophily_ratio")
DIMENSIONS = ["experiment", "population", "topology", "leader_share", "reach", "analysis_round"]
SOURCES = [f"{name}_run_id" for name in ("null", "opleader", "platform", "baseline")]


def endpoint_metrics(round_metrics, analysis_round=50):
    """Require exactly one requested endpoint for every input trajectory."""
    if type(analysis_round) is not int or analysis_round < 0:
        raise ValueError("analysis_round must be a nonnegative integer")
    if round_metrics.empty or round_metrics.duplicated(["run_id", "round"]).any():
        raise ValueError("Empty data or duplicate run-round observations")
    selected = round_metrics.loc[round_metrics["round"] == analysis_round].copy()
    missing = set(round_metrics.run_id) - set(selected.run_id)
    if missing:
        raise ValueError(f"Missing requested round {analysis_round}: {sorted(missing)}")
    selected["analysis_round"] = analysis_round
    return selected.reset_index(drop=True)


def paired_contrasts(endpoints, comparisons):
    """Make matched effects first, then average balanced mirrors within seed."""
    if endpoints.duplicated("run_id").any():
        raise ValueError("Duplicate endpoint run IDs")
    keys = DIMENSIONS + ["seed", "orientation"]
    if comparisons.empty or comparisons.duplicated(keys).any():
        raise ValueError("Empty or duplicate comparisons")
    lookup = endpoints.set_index("run_id")
    rows = []
    for comparison in comparisons.to_dict("records"):
        selected = {}
        for scenario, source in zip(("null", "opleader", "platform", "baseline"), SOURCES):
            run_id = comparison[source]
            if run_id not in lookup.index:
                raise ValueError(f"Missing paired source: {run_id}")
            row = lookup.loc[run_id]
            expected = {"scenario": scenario, "seed": comparison["seed"],
                        "population": comparison["population"], "topology": comparison["topology"],
                        "analysis_round": comparison["analysis_round"]}
            if scenario in ("opleader", "baseline"):
                expected.update(leader_share=comparison["leader_share"], orientation=comparison["orientation"])
            if scenario in ("platform", "baseline"):
                expected["reach"] = comparison["reach"]
            if any(row[key] != value for key, value in expected.items()):
                raise ValueError(f"Mismatched paired source: {run_id}")
            selected[scenario] = row
        for metric in OUTCOMES:
            values = {name: row[metric] for name, row in selected.items()}
            if metric == "mean_signed_belief" and not all(np.isfinite(v) for v in values.values()):
                raise ValueError("Primary outcome must be finite for all paired sources")
            effects = {
                "leader_without_platform": values["opleader"] - values["null"],
                "leader_with_platform": values["baseline"] - values["platform"],
                "platform_without_leaders": values["platform"] - values["null"],
                "platform_with_leaders": values["baseline"] - values["opleader"],
                "platform_leader_interaction": (values["baseline"] - values["platform"])
                                               - (values["opleader"] - values["null"]),
            }
            for contrast, effect in effects.items():
                rows.append({**comparison, "metric": metric, "contrast": contrast, "effect": effect})
    raw = pd.DataFrame(rows)
    directional = raw[raw.orientation.isin(["positive", "negative"])].copy()
    balanced = raw[raw.orientation.str.startswith("balanced_")]
    balanced_rows = []
    for _, group in balanced.groupby(DIMENSIONS + ["seed", "metric", "contrast"], dropna=False):
        if set(group.orientation) != {"balanced_positive", "balanced_negative"} or len(group) != 2:
            raise ValueError("Both balanced mirrors are required once per seed")
        row = group.iloc[0].to_dict()
        row.update(orientation="balanced", effect=float(np.mean(group.effect.to_numpy())))
        for source in SOURCES:
            row[source] = ";".join(sorted(set(group[source])))
        balanced_rows.append(row)
    return pd.concat([directional, pd.DataFrame(balanced_rows)], ignore_index=True)


def summarize_contrasts(contrasts, expected_seeds):
    """Intervals use paired seed effects; missing secondary values stay explicit."""
    groups = DIMENSIONS + ["orientation", "metric", "contrast"]
    rows = []
    for values, group in contrasts.groupby(groups, dropna=False):
        if group.seed.duplicated().any() or set(group.seed) != set(expected_seeds):
            raise ValueError("Incomplete or duplicated seed coverage; no complete-batch summary")
        effects = group.effect.to_numpy(dtype=float)
        valid = effects[np.isfinite(effects)]
        n = len(valid)
        mean = float(valid.mean()) if n else float("nan")
        sd = float(valid.std(ddof=1)) if n > 1 else float("nan")
        # No interval from a subset of seeds with undefined observations.
        half_width = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 and n == len(effects) else float("nan")
        rows.append(dict(zip(groups, values)) | {
            "seed_count": len(effects), "finite_seed_count": n, "mean": mean, "std": sd,
            "ci95_low": mean - half_width, "ci95_high": mean + half_width,
            "interval_status": ("complete" if n > 1 and n == len(effects) else "unavailable"),
        })
    return pd.DataFrame(rows)
