"""Sequential, per-trajectory persistence with strict resume and source checks."""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from opinion_model.experiments.analysis import endpoint_metrics, paired_contrasts, summarize_contrasts
from opinion_model.experiments.observations import initialization_record, role_channel_metrics
from opinion_model.experiments.planning import resolved_config
from opinion_model.scenarios.baseline.experiment import run_baseline_condition
from opinion_model.scenarios.null.experiment import run_null_condition
from opinion_model.scenarios.opleader.experiment import run_opleader_condition
from opinion_model.scenarios.platform.experiment import run_platform_condition
from opinion_model.scenarios.matched_initialization import matched_initialization_streams, ordinary_agent_states
from opinion_model.shared import RandomStreams

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNERS = dict(null=run_null_condition, opleader=run_opleader_condition,
               platform=run_platform_condition, baseline=run_baseline_condition)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"


def digest(value):
    return sha256(json_text(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def code_record():
    files = sorted([*PROJECT_ROOT.glob("src/**/*.py"), *PROJECT_ROOT.glob("scripts/*.py"),
                    PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"])
    hashes = {p.relative_to(PROJECT_ROOT).as_posix(): file_hash(p) for p in files}
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, encoding="utf-8", check=True).stdout.strip()
    return {"revision": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")), "files": hashes,
            "fingerprint": digest(hashes), "python": sys.version, "platform": platform.platform()}


def process_peak_working_set():
    """Windows process lifetime peak, not a claimed per-run allocation peak."""
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def run_paths(batch, spec):
    folder = batch / "runs" / spec.condition_name
    return (folder / f"seed-{spec.seed}__round_metrics.csv",
            folder / f"seed-{spec.seed}__run_manifest.json")


def validate_rounds(frame, spec):
    if len(frame) != spec.simulation_rounds + 1 or list(frame["round"]) != list(range(spec.simulation_rounds + 1)):
        raise ValueError(f"Incomplete round sequence: {spec.run_id}")
    for name, value in {"run_id": spec.run_id, **asdict(spec)}.items():
        if value is None:
            if not frame[name].isna().all():
                raise ValueError(f"Unexpected applicable field: {name}")
        elif not frame[name].eq(value).all():
            raise ValueError(f"Run metadata mismatch: {name}")
    if not np.isfinite(frame.mean_signed_belief).all() or not frame.mean_signed_belief.between(-1, 1).all():
        raise ValueError("Invalid primary outcome")
    if spec.reach is not None:
        if not (frame.tied_exposure_count + frame.out_of_network_exposure_count == frame.exposure_count).all():
            raise ValueError("Channel totals do not match exposures")
        changes = frame.edge_count.diff().iloc[1:]
        expected = (frame.accepted_addition_count - frame.accepted_removal_count).iloc[1:]
        if not changes.eq(expected).all():
            raise ValueError("Tie additions/removals do not reconcile with edge counts")


def load_completed(batch, spec, config, fingerprint):
    metric_path, manifest_path = run_paths(batch, spec)
    if not manifest_path.exists():
        return None
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record["code_fingerprint"] != fingerprint or record["config_hash"] != digest(asdict(config)):
        raise ValueError(f"Cannot reuse run with changed code/configuration: {spec.run_id}")
    if record["spec"] != asdict(spec) or record["run_id"] != spec.run_id:
        raise ValueError("Source run definition changed")
    if not metric_path.exists() or file_hash(metric_path) != record["round_metrics_sha256"]:
        raise ValueError(f"Missing or altered completed output: {metric_path}")
    agents_path = metric_path.with_name(f"seed-{spec.seed}__initial_agents.csv")
    if not agents_path.exists() or file_hash(agents_path) != record["initial_agents_sha256"]:
        raise ValueError(f"Missing or altered initial agents: {agents_path}")
    # "null" is a scenario name, not a missing-value token.
    frame = pd.read_csv(metric_path, keep_default_na=False, na_values=[""])
    validate_rounds(frame, spec)
    return frame, record, metric_path


def execute_one(batch, spec, config, threshold, code):
    started = utc_now()
    tick = time.perf_counter()
    arguments = {"extremism_threshold": threshold}
    if spec.leader_share is not None:
        arguments["orientation"] = spec.orientation
    run = RUNNERS[spec.scenario](config, **arguments)
    frame = role_channel_metrics(run)
    for name, value in {"run_id": spec.run_id, **asdict(spec)}.items():
        frame[name] = value
    validate_rounds(frame, spec)
    initialization = initialization_record(run)
    streams = matched_initialization_streams(RandomStreams(spec.seed).initialization())
    ordinary = ordinary_agent_states(spec.population, config.initialization.ordinary_mean_alpha,
                                     config.initialization.ordinary_concentration, streams.ordinary_belief)
    leaders = set(initialization["leader_ids"])
    if any(agent != run.initialization.state.agents[i] for i, agent in ordinary.items() if i not in leaders):
        raise AssertionError("Ordinary-agent initialization does not match shared draws")
    initialization["pre_leader_beliefs_sha256"] = digest(
        [(i, a.belief.a, a.belief.b) for i, a in sorted(ordinary.items())])
    metric_path, manifest_path = run_paths(batch, spec)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(metric_path, frame)
    initial_agents = pd.DataFrame([
        {"agent_id": i, "a": a.belief.a, "b": a.belief.b,
         "pre_leader_a": ordinary[i].belief.a, "pre_leader_b": ordinary[i].belief.b,
         "is_leader": i in leaders}
        for i, a in sorted(run.initialization.state.agents.items())])
    agents_path = metric_path.with_name(f"seed-{spec.seed}__initial_agents.csv")
    write_csv(agents_path, initial_agents)
    record = {"run_id": spec.run_id, "spec": asdict(spec), "resolved_config": asdict(config),
              "config_hash": digest(asdict(config)), "code_fingerprint": code["fingerprint"],
              "shared_framework_revision": code["revision"], "scenario_revision": code["revision"],
              "code_dirty": code["dirty"], "started_at": started, "completed_at": utc_now(),
              "duration_seconds": time.perf_counter() - tick,
              "process_lifetime_peak_working_set_bytes": process_peak_working_set(),
              "round_metrics_sha256": file_hash(metric_path), **initialization}
    record["initial_agents_sha256"] = file_hash(agents_path)
    write_json(manifest_path, record)
    return frame, record, metric_path


def export_plan(batch, design, plan, kind, template):
    batch.mkdir(parents=True, exist_ok=True)
    specs = [plan.runs[i] for i in sorted(plan.required_ids(kind))]
    frame = pd.DataFrame([{"run_id": s.run_id, **asdict(s), "reuse": s.owner != kind}
                          for s in specs])
    write_csv(batch / f"{kind}_run_plan.csv", frame)
    comparisons = plan.comparisons.loc[plan.comparisons.experiment == kind]
    if kind != "horizon":
        write_csv(batch / f"{kind}_comparison_plan.csv", comparisons)
    write_json(batch / f"{kind}_resolved_configs.json",
               {s.run_id: {"topology": s.topology, "config": asdict(resolved_config(template, s)),
                           "execution_supported": s.topology == "ba"} for s in specs})
    write_json(batch / f"{kind}_plan_summary.json", {
        "status": design.status, "analysis_round": design.rounds,
        "total_study_unique_runs": len(plan.runs),
        "unique_runs_by_owner": plan.run_frame().owner.value_counts().to_dict(),
        "required_runs": len(specs), "new_runs": sum(s.owner == kind for s in specs),
        "reused_runs": sum(s.owner != kind for s in specs),
        "topology_execution": "BA only in stages I-II; ER/WS/SBM generators are deferred",
    })
    return specs


def execute_batch(batch, design, plan, kind, template, *, resume=False, source_batch=None, max_runs=None):
    batch = Path(batch).resolve()
    if design.status == "provisional":
        raise ValueError("Provisional retained settings are plan-only. Use a pilot config until reviewed and frozen.")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")
    required = [plan.runs[i] for i in sorted(plan.required_ids(kind))]
    if any(s.topology != "ba" for s in required):
        raise NotImplementedError("Topology generators are a later stage; this runner will not substitute BA.")
    if any(s.owner != kind for s in required) and source_batch is None:
        raise ValueError("Supporting execution requires --source-batch for shared main controls")
    code = code_record()
    identity = {"design": asdict(design), "kind": kind, "code_fingerprint": code["fingerprint"],
                "configs": {s.run_id: asdict(resolved_config(template, s)) for s in required},
                "extremism_threshold": template.extremism_threshold}
    identity_hash = digest(identity)
    manifest_path = batch / f"{kind}_batch_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or old["identity_hash"] != identity_hash:
            raise ValueError("Existing batch requires --resume and identical code/configuration")
        manifest = old
    else:
        if any(batch.glob("runs/*/*")):
            raise ValueError("Refusing to adopt existing runs without a batch manifest")
        batch.mkdir(parents=True, exist_ok=True)
        manifest = {"identity_hash": identity_hash, "created_at": utc_now(), "kind": kind,
                    "design_status": design.status, "analysis_round": design.rounds,
                    "expected_seeds": list(design.seeds), "code": code, "execution": "sequential",
                    "source_batch": str(Path(source_batch).resolve()) if source_batch else None}
        with zipfile.ZipFile(batch / f"{kind}_source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in code["files"]:
                archive.write(PROJECT_ROOT / name, name)
    export_plan(batch, design, plan, kind, template)
    manifest.update(status="running", last_started_at=utc_now())
    write_json(manifest_path, manifest)
    collected, records, sources, status_rows = [], [], [], []
    new_count = 0
    try:
        # Validate reusable source runs before doing any new simulation work.
        for spec in required:
            if spec.owner != kind:
                source = load_completed(Path(source_batch), spec, resolved_config(template, spec), code["fingerprint"])
                if source is None:
                    raise ValueError(f"Missing completed main control: {spec.run_id}")
        for index, spec in enumerate(required, 1):
            config = resolved_config(template, spec)
            origin = batch if spec.owner == kind else Path(source_batch)
            result = load_completed(origin, spec, config, code["fingerprint"])
            action = "reused" if spec.owner != kind else "resumed"
            if result is None:
                if max_runs is not None and new_count >= max_runs:
                    status_rows.append({"run_id": spec.run_id, "status": "pending"})
                    continue
                print(f"[{index}/{len(required)}] {spec.run_id}", flush=True)
                with (batch / f"{kind}_attempts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"run_id": spec.run_id, "event": "started", "at": utc_now()}) + "\n")
                try:
                    result = execute_one(batch, spec, config, template.extremism_threshold, code)
                except Exception as error:
                    with (batch / f"{kind}_attempts.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"run_id": spec.run_id, "event": "failed", "at": utc_now(), "error": repr(error)}) + "\n")
                    status_rows.append({"run_id": spec.run_id, "status": "failed", "error": repr(error)})
                    raise
                with (batch / f"{kind}_attempts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"run_id": spec.run_id, "event": "completed", "at": utc_now()}) + "\n")
                action = "completed"
                new_count += 1
            frame, record, path = result
            collected.append(frame)
            records.append(record)
            sources.append({"run_id": spec.run_id, "source_path": str(path.resolve()),
                            "source_sha256": record["round_metrics_sha256"], "action": action})
            status_rows.append({"run_id": spec.run_id, "status": action,
                                "duration_seconds": record["duration_seconds"],
                                "process_lifetime_peak_working_set_bytes": record["process_lifetime_peak_working_set_bytes"]})
            write_csv(batch / f"{kind}_run_status.csv", pd.DataFrame(status_rows))
        if len(collected) != len(required):
            manifest.update(status="paused", completed_runs=len(collected), required_runs=len(required))
            return manifest
        # The matching unit is population/topology/seed, not leader share or direction.
        matching = {}
        for spec, record in zip(required, records):
            key = (spec.population, spec.topology, spec.seed)
            signatures = (record["initial_network_sha256"], record["pre_leader_beliefs_sha256"])
            if matching.setdefault(key, signatures) != signatures:
                raise AssertionError("Cross-scenario initialization mismatch")
        rounds = pd.DataFrame.from_records([row for frame in collected for row in frame.to_dict("records")])
        rounds["design_status"] = design.status
        write_csv(batch / f"{kind}_round_metrics.csv", rounds)
        write_csv(batch / f"{kind}_source_runs.csv", pd.DataFrame(sources))
        write_csv(batch / f"{kind}_initialization_checks.csv", pd.DataFrame([
            {"run_id": r["run_id"], **{key: r[key] for key in ("initial_network_sha256", "pre_leader_beliefs_sha256",
                                                              "realized_leader_count", "realized_leader_share", "initial_edge_count")}}
            for r in records]))
        endpoint = endpoint_metrics(rounds, design.rounds)
        write_csv(batch / f"{kind}_round{design.rounds}_outcomes.csv", endpoint)
        if kind == "horizon":
            checkpoints = sorted({r for r in (30, design.rounds, 75, 100, design.extended_rounds)
                                  if r <= design.extended_rounds})
            check_frames = [endpoint_metrics(rounds, r) for r in checkpoints]
            write_csv(batch / f"horizon_checkpoint_outcomes.csv", pd.concat(check_frames, ignore_index=True))
        else:
            comparisons = plan.comparisons[plan.comparisons.experiment == kind]
            contrasts = paired_contrasts(endpoint, comparisons)
            summary = summarize_contrasts(contrasts, design.seeds)
            contrasts["design_status"] = design.status
            summary["design_status"] = design.status
            write_csv(batch / f"{kind}_round{design.rounds}_seed_contrasts.csv", contrasts)
            write_csv(batch / f"{kind}_round{design.rounds}_effect_summary.csv", summary)
        manifest.update(status="complete", completed_runs=len(required), required_runs=len(required), completed_at=utc_now())
    except Exception as error:
        manifest.update(status="failed", error=repr(error), failed_at=utc_now())
        raise
    finally:
        write_csv(batch / f"{kind}_run_status.csv", pd.DataFrame(status_rows))
        write_json(manifest_path, manifest)
    return manifest
