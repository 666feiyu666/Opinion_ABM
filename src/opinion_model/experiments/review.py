"""Read-only verification and display tables for completed retained studies."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import t

from opinion_model.experiments.analysis import endpoint_metrics, paired_contrasts, summarize_contrasts, DIMENSIONS


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def equal_tables(left, right, keys):
    require(set(left.columns) <= set(right.columns), "Missing retained columns")
    pd.testing.assert_frame_equal(left.sort_values(keys).reset_index(drop=True),
                                  right[left.columns].sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)


@dataclass
class ReviewedBatch:
    path: Path
    kind: str
    manifest: dict
    plan: pd.DataFrame
    rounds: pd.DataFrame
    endpoints: pd.DataFrame
    contrasts: pd.DataFrame | None
    summary: pd.DataFrame | None


def audit_study(paths):
    """Verify individual files, aggregate tables, reuse, pairing and intervals."""
    inputs, batches, checks, unique, canonical, signatures = {}, {}, [], {}, {}, {}

    def remember(path):
        path = Path(path).resolve()
        inputs[str(path)] = digest_file(path)
        return path

    def table(path):
        return read_csv(remember(path))

    fingerprints = set()
    for kind, folder in paths.items():
        folder = Path(folder).resolve()
        manifest = json.loads(remember(folder / f"{kind}_batch_manifest.json").read_text())
        require(manifest["status"] == "complete" and manifest["design_status"] == "frozen", "Incomplete/non-retained batch")
        fingerprints.add(manifest["code"]["fingerprint"])
        with zipfile.ZipFile(remember(folder / f"{kind}_source_snapshot.zip")) as archive:
            for name, expected in manifest["code"]["files"].items():
                require(sha256(archive.read(name)).hexdigest() == expected, f"Changed code snapshot: {name}")
        plan = table(folder / f"{kind}_run_plan.csv")
        sources = table(folder / f"{kind}_source_runs.csv")
        statuses = table(folder / f"{kind}_run_status.csv")
        rounds = table(folder / f"{kind}_round_metrics.csv")
        configs = json.loads(remember(folder / f"{kind}_resolved_configs.json").read_text())
        for frame in (plan, sources, statuses):
            require(not frame.run_id.duplicated().any(), "Duplicate run IDs")
        ids = set(plan.run_id)
        require(ids == set(sources.run_id) == set(statuses.run_id) == set(rounds.run_id) == set(configs), "Run coverage mismatch")
        require(len(ids) == manifest["required_runs"] == manifest["completed_runs"], "Manifest count mismatch")
        require(set(statuses.status) <= {"completed", "resumed", "reused"}, "Unfinished trajectory")
        require(not rounds.duplicated(["run_id", "round"]).any(), "Duplicate run-round rows")
        grouped = {run_id: frame for run_id, frame in rounds.groupby("run_id")}
        planned = plan.set_index("run_id")
        for source in sources.itertuples(index=False):
            path = remember(source.source_path)
            require(inputs[str(path)] == source.source_sha256, f"Changed trajectory: {source.run_id}")
            if source.run_id in canonical:
                require(canonical[source.run_id] == str(path), "Reuse points to another physical trajectory")
            canonical[source.run_id] = str(path)
            record_path = path.with_name(path.name.replace("round_metrics.csv", "run_manifest.json"))
            record = json.loads(remember(record_path).read_text())
            require(record["round_metrics_sha256"] == source.source_sha256, "Run hash mismatch")
            require(record["run_id"] == source.run_id and record["code_fingerprint"] in fingerprints, "Run provenance mismatch")
            require(record["resolved_config"] == configs[source.run_id]["config"], "Resolved configuration mismatch")
            config_hash = sha256((json.dumps(record["resolved_config"], indent=2, sort_keys=True, default=str, allow_nan=False) + "\n").encode()).hexdigest()
            require(record["config_hash"] == config_hash, "Configuration hash mismatch")
            agents_path = remember(path.with_name(path.name.replace("round_metrics.csv", "initial_agents.csv")))
            require(inputs[str(agents_path)] == record["initial_agents_sha256"], "Initial agent hash mismatch")
            agents = read_csv(agents_path)
            require(len(agents) == record["spec"]["population"] and not agents.agent_id.duplicated().any(), "Initial agent coverage")
            require(agents.initial_in_degree.sum() == agents.initial_out_degree.sum() == record["initial_edge_count"], "Initial degrees do not reconcile")
            require(int(agents.is_leader.sum()) == record["realized_leader_count"], "Realized leader count mismatch")
            # Floating CSV round-trips need not reproduce JSON bytes; shared recorded hashes
            # are checked across conditions and the CSV bytes are checked above.
            key = (record["spec"]["population"], record["spec"]["topology"], record["spec"]["seed"])
            signature = (record["initial_network_sha256"], record["pre_leader_beliefs_sha256"])
            require(signatures.setdefault(key, signature) == signature, "Cross-scenario initialization mismatch")
            frame = read_csv(path)
            require(frame["round"].tolist() == list(range(record["spec"]["simulation_rounds"] + 1)), "Missing rounds")
            for column, value in record["spec"].items():
                require(frame[column].isna().all() if value is None else frame[column].eq(value).all(), f"Invalid metadata: {column}")
                observed = planned.loc[source.run_id, column]
                require(pd.isna(observed) if value is None else observed == value, f"Plan metadata mismatch: {column}")
            equal_tables(frame, grouped[source.run_id], ["run_id", "round"])
            require(frame.mean_signed_belief.between(-1, 1).all(), "Primary outcome outside range")
            if record["spec"]["reach"] is not None:
                require((frame.tied_exposure_count + frame.out_of_network_exposure_count).eq(frame.exposure_count).all(), "Channel totals")
                require(frame.edge_count.diff().iloc[1:].eq((frame.accepted_addition_count-frame.accepted_removal_count).iloc[1:]).all(), "Tie accounting")
            if record["spec"]["leader_share"] is not None:
                channels = [f"{role}_{channel}_exposure_count" for role in ("leader", "ordinary") for channel in ("tied", "out_of_network")]
                require(frame[channels].sum(axis=1).eq(frame.exposure_count).all(), "Role-channel accounting")
            checkpoints = {r for r in (0, 10, 30, 50, 75, 100) if r <= record["spec"]["simulation_rounds"]}
            observed = frame.loc[frame.structural_top_count.notna()]
            require(set(observed["round"]) == checkpoints, "Missing structural checkpoints")
            require(observed.undirected_clustering.between(0, 1).all() and observed.largest_weak_component_fraction.between(0, 1).all(), "Invalid network measures")
            if record["spec"]["owner"] == kind:
                require(source.action != "reused", "Owned run marked reused")
                unique[source.run_id] = kind
            else:
                require(source.action == "reused" and source.run_id in unique, "Unrecognized reused source")
        endpoints = table(folder / f"{kind}_round50_outcomes.csv")
        equal_tables(endpoint_metrics(rounds, 50), endpoints, ["run_id"])
        snapshots = table(folder / f"{kind}_network_snapshots.csv")
        equal_tables(rounds.loc[rounds.structural_top_count.notna()], snapshots, ["run_id", "round"])
        contrasts = summary = None
        if kind != "horizon":
            comparisons = table(folder / f"{kind}_comparison_plan.csv")
            contrasts = table(folder / f"{kind}_round50_seed_contrasts.csv")
            summary = table(folder / f"{kind}_round50_effect_summary.csv")
            recomputed = paired_contrasts(endpoints, comparisons)
            keys = DIMENSIONS + ["orientation", "metric", "contrast"]
            equal_tables(recomputed, contrasts, keys + ["seed"])
            equal_tables(summarize_contrasts(recomputed, manifest["expected_seeds"]), summary, keys)
        else:
            checkpoints = table(folder / "horizon_checkpoint_outcomes.csv")
            equal_tables(pd.concat([endpoint_metrics(rounds, r) for r in (30, 50, 75, 100)]), checkpoints, ["run_id", "round"])
        checks.append({"batch": kind, "required_runs": len(ids), "owned_runs": sum(plan.owner == kind),
                       "reused_runs": sum(plan.owner != kind), "round_rows": len(rounds),
                       "historical_error": manifest.get("error"), "status": "passed"})
        batches[kind] = ReviewedBatch(folder, kind, manifest, plan, rounds, endpoints, contrasts, summary)
    require(len(fingerprints) == 1, "Mixed simulation code fingerprints")
    return batches, {"status": "passed", "unique_trajectories": len(unique), "batches": checks,
                     "simulation_fingerprint": next(iter(fingerprints)), "input_sha256": inputs}


def seed_display(frame, metrics):
    """Average only mirrored orientations within seed; leave controls unduplicated."""
    frame = frame.copy()
    frame["source_orientation"] = frame.orientation
    frame["orientation"] = frame.orientation.replace({"balanced_positive": "balanced", "balanced_negative": "balanced"})
    keys = ["population", "topology", "leader_share", "reach", "scenario", "orientation", "seed", "round"]
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        require(len(group) == (2 if row["orientation"] == "balanced" else 1), "Missing mirror or duplicated display observation")
        if row["orientation"] == "balanced":
            require(set(group.source_orientation) == {"balanced_positive", "balanced_negative"}, "Duplicated mirror orientation")
        row["source_run_ids"] = ";".join(sorted(set(group.run_id)))
        for metric in metrics:
            row[metric] = float(np.mean(group[metric].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_summary(frame, metrics):
    """Seed-level descriptive summaries with explicit finite counts and t intervals."""
    keys = [c for c in ("population", "topology", "leader_share", "reach", "scenario", "orientation", "round") if c in frame]
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        require(not group.seed.duplicated().any(), "Duplicate seed in descriptive group")
        for metric in metrics:
            v = group[metric].to_numpy(dtype=float)
            finite = v[np.isfinite(v)]
            n = len(finite)
            mean = np.mean(finite) if n else np.nan
            sd = np.std(finite, ddof=1) if n > 1 else np.nan
            half = t.ppf(.975, n-1) * sd / np.sqrt(n) if n == len(v) and n > 1 else np.nan
            rows.append(dict(zip(keys, values)) | {"metric": metric, "seed_count": len(v), "finite_seed_count": n,
                        "mean": mean, "std": sd, "ci95_low": mean-half, "ci95_high": mean+half})
    return pd.DataFrame(rows)
