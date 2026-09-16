"""Validate study grids and enumerate unique runs and explicit control reuse."""

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from itertools import product
import math
import tomllib

import pandas as pd

from opinion_model.scenarios.comparison.config import load_comparison_experiment_config

ORIENTATIONS = ("positive", "negative", "balanced_positive", "balanced_negative")


@dataclass(frozen=True)
class Design:
    source: Path
    comparison_path: Path
    status: str
    populations: tuple[int, ...]
    shares: tuple[float, ...]
    seeds: tuple[int, ...]
    rounds: int
    extended_rounds: int
    reference_population: int
    reference_share: float
    reach: float
    reach_levels: tuple[float, ...]
    topologies: tuple[str, ...]

    def __post_init__(self):
        for name in ("populations", "shares", "seeds", "reach_levels", "topologies"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
        for n in (*self.populations, self.reference_population, self.rounds, self.extended_rounds):
            if type(n) is not int or n < 1:
                raise ValueError("Population and round counts must be positive integers")
        if any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("Seeds must be nonnegative integers")
        if self.status not in ("pilot", "provisional", "frozen"):
            raise ValueError("Unknown design status")
        if self.rounds != 50 and self.status != "pilot":
            raise ValueError("The main scientific endpoint must be round 50")
        if self.extended_rounds < self.rounds:
            raise ValueError("Extended horizon cannot precede main endpoint")
        if self.reference_population not in self.populations or self.reference_share not in self.shares:
            raise ValueError("Reference setting must be in the main grid")
        if any(not math.isfinite(v) or not 0 < v < 1 for v in self.shares):
            raise ValueError("Leader shares must lie in (0, 1)")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (*self.reach_levels, self.reach)):
            raise ValueError("Reach probabilities must lie in [0, 1]")
        if self.reach not in self.reach_levels:
            raise ValueError("Reach levels must include the main value")
        if "ba" not in self.topologies or set(self.topologies) - {"ba", "er", "ws", "sbm"}:
            raise ValueError("Topologies must include ba and use known families")


def load_design(path):
    source = Path(path).resolve()
    raw = tomllib.loads(source.read_text(encoding="utf-8"))["experiment"]
    return Design(source=source, comparison_path=(source.parent / raw["comparison"]).resolve(),
                  status=raw["status"], populations=tuple(raw["populations"]),
                  shares=tuple(raw["leader_shares"]), seeds=tuple(raw["seeds"]),
                  rounds=raw["analysis_round"], extended_rounds=raw["extended_rounds"],
                  reference_population=raw["reference_population"],
                  reference_share=raw["reference_share"], reach=raw["main_reach"],
                  reach_levels=tuple(raw["reach_levels"]), topologies=tuple(raw["topologies"]))


def number(value):
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    population: int
    topology: str
    leader_share: float | None
    orientation: str
    reach: float | None
    seed: int
    simulation_rounds: int
    owner: str

    @property
    def condition_name(self):
        parts = [self.scenario, self.topology, f"n{self.population}"]
        if self.leader_share is not None:
            parts.extend([f"leaders{number(self.leader_share * 100)}pct", self.orientation])
        if self.reach is not None:
            parts.append(f"reach{number(self.reach)}")
        return "__".join(parts)

    @property
    def run_id(self):
        return f"{self.condition_name}__seed-{self.seed}"


@dataclass
class StudyPlan:
    runs: dict[str, RunSpec]
    comparisons: pd.DataFrame
    horizon_ids: tuple[str, ...]

    def run_frame(self, kind=None):
        rows = [{"run_id": r.run_id, **asdict(r)} for r in self.runs.values()
                if kind is None or r.owner == kind]
        return pd.DataFrame(rows)

    def required_ids(self, kind):
        if kind == "horizon":
            return set(self.horizon_ids)
        subset = self.comparisons.loc[self.comparisons.experiment == kind]
        return {value for name in ("null", "opleader", "platform", "baseline")
                for value in subset[f"{name}_run_id"]}


def build_plan(design):
    runs, comparisons, horizon_ids = {}, [], set()

    def add(scenario, n, share, orientation, reach, topology, seed, owner):
        leader = scenario in ("opleader", "baseline")
        platform = scenario in ("platform", "baseline")
        reference = (topology == "ba" and n == design.reference_population
                     and (not leader or share == design.reference_share)
                     and (not platform or reach == design.reach))
        rounds = design.extended_rounds if reference else design.rounds
        spec = RunSpec(scenario, n, topology, share if leader else None,
                       orientation if leader else "none", reach if platform else None,
                       seed, rounds, owner)
        old = runs.setdefault(spec.run_id, spec)
        if replace(old, owner=owner) != spec:
            raise ValueError("Conflicting definitions for the same physical run")
        if reference:
            horizon_ids.add(spec.run_id)
        return spec.run_id

    def comparison(kind, n, share, reach, topology, seed, orientation):
        row = dict(experiment=kind, population=n, topology=topology,
                   leader_share=share, reach=reach, seed=seed, orientation=orientation,
                   analysis_round=design.rounds)
        for scenario in ("null", "opleader", "platform", "baseline"):
            row[f"{scenario}_run_id"] = add(scenario, n, share, orientation, reach,
                                             topology, seed, kind)
        comparisons.append(row)

    for n, share, seed, orientation in product(design.populations, design.shares, design.seeds, ORIENTATIONS):
        comparison("main", n, share, design.reach, "ba", seed, orientation)
    for reach, seed, orientation in product(design.reach_levels, design.seeds, ORIENTATIONS):
        comparison("reach", design.reference_population, design.reference_share,
                   reach, "ba", seed, orientation)
    for topology, seed, orientation in product(design.topologies, design.seeds, ORIENTATIONS):
        comparison("topology", design.reference_population, design.reference_share,
                   design.reach, topology, seed, orientation)
    return StudyPlan(runs, pd.DataFrame(comparisons), tuple(sorted(horizon_ids)))


def resolved_config(template, spec):
    attr = "platform_case" if spec.scenario == "platform" else spec.scenario
    config = getattr(getattr(template, spec.scenario), attr)
    simulation = replace(config.simulation, agent_count=spec.population,
                         rounds=spec.simulation_rounds, seed=spec.seed,
                         consumption_capacity=(config.simulation.consumption_capacity
                                               if spec.reach is not None else spec.population - 1))
    changes = {"simulation": simulation}
    if spec.leader_share is not None:
        changes["initialization"] = replace(config.initialization, leader_share=spec.leader_share)
    if spec.reach is not None:
        changes["platform"] = replace(config.platform, out_of_network_availability_probability=spec.reach)
    return replace(config, **changes)


def validate_plan(design, plan):
    template = load_comparison_experiment_config(design.comparison_path)
    for spec in plan.runs.values():
        resolved_config(template, spec)
    if plan.comparisons.duplicated(["experiment", "population", "topology", "leader_share", "reach", "seed", "orientation"]).any():
        raise ValueError("Duplicate comparison")
    return template
