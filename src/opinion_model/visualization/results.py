"""Render matched experiment figures from saved tables, independently of simulation."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SCENARIOS = ("null", "opleader", "platform", "baseline")
ORIENTATIONS = ("positive", "negative", "balanced")
TITLES = {"positive": "Positive leaders", "negative": "Negative leaders", "balanced": "Balanced leaders"}
COLORS = {"null": "#697586", "opleader": "#B46B05", "platform": "#007F87", "baseline": "#6844A3"}
STYLES = {"null": "--", "opleader": "-.", "platform": ":", "baseline": "-"}
EFFECTS = ("leader_without_platform", "leader_with_platform", "platform_leader_interaction")
EFFECT_LABELS = ("Without platform", "With platform", "Interaction")
EFFECT_COLORS = (COLORS["opleader"], COLORS["baseline"], "#356A9C")
METRICS = ("mean_signed_belief", "cumulative_content_balance", "out_of_network_exposure_count",
           "leader_exposure_count", "capacity_binding_rate", "accepted_addition_count", "accepted_removal_count")
RAW_ORIENTATIONS = ("positive", "negative", "balanced_positive", "balanced_negative")


def read_table(path):
    # The literal scenario name 'null' must survive CSV loading.
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class Setting:
    population: int
    topology: str
    leader_share: float
    reach: float

    @property
    def label(self):
        return f"N = {self.population:,}  |  {self.topology.upper()}  |  leaders = {100 * self.leader_share:g}%  |  reach = {self.reach:g}"

    @property
    def slug(self):
        return f"n{self.population}__{self.topology}__leaders{number(self.leader_share * 100)}pct__reach{number(self.reach)}"


def select_setting(frame, population=None, topology=None, leader_share=None, reach=None):
    """Choose one unambiguous setting, keeping mechanism-inapplicable controls."""
    candidates = frame.loc[frame.scenario == "baseline", ["population", "topology", "leader_share", "reach"]].drop_duplicates()
    for name, value in dict(population=population, topology=topology, leader_share=leader_share, reach=reach).items():
        if value is not None:
            candidates = candidates.loc[np.isclose(candidates[name], value) if name in ("leader_share", "reach") else candidates[name].eq(value)]
    if len(candidates) != 1:
        raise ValueError("Select exactly one population/topology/leader-share/reach setting; available matches: "
                         + candidates.to_json(orient="records"))
    selected = candidates.iloc[0]
    setting = Setting(int(selected.population), str(selected.topology), float(selected.leader_share), float(selected.reach))
    mask = frame.population.eq(setting.population) & frame.topology.eq(setting.topology)
    mask &= ~frame.scenario.isin(["opleader", "baseline"]) | np.isclose(frame.leader_share, setting.leader_share)
    mask &= ~frame.scenario.isin(["platform", "baseline"]) | np.isclose(frame.reach, setting.reach)
    return setting, frame.loc[mask].copy()


def prepare_trajectories(frame, expected_seeds, through_round=None):
    """Mechanical display reshape: mirror averaging stays within seed and round."""
    if frame.duplicated(["run_id", "round"]).any():
        raise ValueError("Duplicate run-round observations")
    if set(frame.scenario) != set(SCENARIOS):
        raise ValueError("All four scenarios are required")
    horizons = []
    for scenario in SCENARIOS:
        scenario_frame = frame[frame.scenario == scenario]
        expected_orientations = RAW_ORIENTATIONS if scenario in ("opleader", "baseline") else ("none",)
        if set(scenario_frame.orientation) != set(expected_orientations):
            raise ValueError(f"Missing or unexpected orientations for {scenario}")
        for orientation in expected_orientations:
            subset = scenario_frame[scenario_frame.orientation == orientation]
            if set(subset.seed) != set(expected_seeds):
                raise ValueError("Incomplete seed coverage")
            for seed, group in subset.groupby("seed"):
                if group.run_id.nunique() != 1:
                    raise ValueError("More than one run for the same seed/scenario/orientation")
                indices = sorted(group["round"].tolist())
                if indices != list(range(indices[-1] + 1)):
                    raise ValueError("Incomplete round sequence")
                horizons.append(indices[-1])
    common_horizon = min(horizons)
    stop = common_horizon if through_round is None else through_round
    if type(stop) is not int or not 1 <= stop <= common_horizon:
        raise ValueError(f"through-round must be between 1 and the common horizon {common_horizon}")
    selected = frame[frame["round"] <= stop].copy()
    selected["leader_exposure_count"] = selected["leader_tied_exposure_count"] + selected["leader_out_of_network_exposure_count"]
    rows = []
    for orientation in ORIENTATIONS:
        for scenario in SCENARIOS:
            group = selected[selected.scenario == scenario]
            if scenario in ("opleader", "baseline"):
                labels = ("balanced_positive", "balanced_negative") if orientation == "balanced" else (orientation,)
                group = group[group.orientation.isin(labels)]
            for (seed, round_index), observations in group.groupby(["seed", "round"], sort=True):
                record = {"scenario": scenario, "orientation": orientation, "seed": seed, "round": round_index,
                          "source_run_ids": ";".join(sorted(observations.run_id.unique()))}
                for metric in METRICS:
                    # np.mean propagates missingness: never average only one undefined mirror.
                    record[metric] = float(np.mean(observations[metric].to_numpy(dtype=float)))
                rows.append(record)
    return pd.DataFrame(rows), stop, max(horizons)


@dataclass
class FigureData:
    batch: Path
    manifest: dict
    setting: Setting
    trajectories: pd.DataFrame
    contrasts: pd.DataFrame | None
    summary: pd.DataFrame | None
    through_round: int
    maximum_source_round: int
    inputs: list[Path]

    @property
    def pilot(self):
        return self.manifest["design_status"] == "pilot"

    @property
    def seeds(self):
        return tuple(self.manifest["expected_seeds"])

    @property
    def endpoint(self):
        return int(self.manifest["analysis_round"])

    @property
    def seed_label(self):
        return f"{len(self.seeds)} {'seed' if len(self.seeds) == 1 else 'seeds'}"

    @property
    def prefix(self):
        status = "pilot_" if self.pilot else ""
        return f"{status}{self.manifest['kind']}__{self.setting.slug}"


def load_figure_data(batch, *, population=None, topology=None, leader_share=None, reach=None, through_round=None,
                     reference_only=False):
    batch = Path(batch).resolve()
    paths = list(batch.glob("*_batch_manifest.json"))
    if len(paths) != 1:
        raise ValueError("Expected one completed batch manifest")
    manifest_path = paths[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_complete = (reference_only and manifest["kind"] == "main"
                          and manifest.get("reference_status") == "complete"
                          and manifest["status"] in ("paused", "complete"))
    if reference_only and not reference_complete:
        raise ValueError("Figures require a complete main reference condition")
    if (manifest["status"] != "complete" and not reference_complete) or manifest["design_status"] not in ("pilot", "frozen"):
        raise ValueError("Figures require a complete pilot or frozen batch")
    if reference_only:
        manifest = {**manifest, "kind": "main_reference"}
    kind, endpoint = manifest["kind"], manifest["analysis_round"]
    rounds_path = batch / f"{kind}_round_metrics.csv"
    frame = read_table(rounds_path)
    if set(frame.design_status) != {manifest["design_status"]}:
        raise ValueError("Table status disagrees with batch metadata")
    setting, selected = select_setting(frame, population, topology, leader_share, reach)
    trajectories, stop, maximum = prepare_trajectories(selected, manifest["expected_seeds"], through_round)
    inputs = [manifest_path, rounds_path]
    contrasts = summary = None
    if kind != "horizon":
        contrast_path = batch / f"{kind}_round{endpoint}_seed_contrasts.csv"
        summary_path = batch / f"{kind}_round{endpoint}_effect_summary.csv"
        tables = []
        for path in (contrast_path, summary_path):
            table = read_table(path)
            table = table[table.population.eq(setting.population) & table.topology.eq(setting.topology)
                          & np.isclose(table.leader_share, setting.leader_share) & np.isclose(table.reach, setting.reach)
                          & table.metric.eq("mean_signed_belief") & table.contrast.isin(EFFECTS)].copy()
            if table.empty or set(table.analysis_round) != {endpoint} or set(table.design_status) != {manifest["design_status"]}:
                raise ValueError("Missing or mismatched effect tables")
            tables.append(table)
        contrasts, summary = tables
        validate_effect_tables(contrasts, summary, manifest["expected_seeds"])
        inputs.extend([contrast_path, summary_path])
    return FigureData(batch, manifest, setting, trajectories, contrasts, summary, stop, maximum, inputs)


def validate_effect_tables(contrasts, summary, seeds):
    for orientation in ORIENTATIONS:
        for effect in EFFECTS:
            points = contrasts[(contrasts.orientation == orientation) & (contrasts.contrast == effect)]
            aggregates = summary[(summary.orientation == orientation) & (summary.contrast == effect)]
            if len(aggregates) != 1 or len(points) != len(seeds) or set(points.seed) != set(seeds):
                raise ValueError("Missing or duplicated seed effects or summaries")
            aggregate = aggregates.iloc[0]
            if not np.isfinite(points.effect).all() or aggregate.seed_count != len(seeds):
                raise ValueError("Invalid effect observations")
            if not np.isclose(points.effect.mean(), aggregate["mean"], rtol=1e-10, atol=1e-12):
                raise ValueError("Stored effect summary disagrees with seed values")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#CAD0D8", "axes.labelcolor": "#243342",
                         "xtick.color": "#526070", "ytick.color": "#526070", "figure.facecolor": "white",
                         "axes.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"})


def header(fig, data, title, subtitle, scenario_legend=True):
    status = "PILOT  |  " if data.pilot else ""
    height = fig.get_figheight()
    fig.suptitle(title, x=.07, y=.978, ha="left", fontsize=18, fontweight="bold", color="#1C2B3B")
    fig.text(.07, 1 - .65 / height, f"{status}{data.setting.label}  |  {data.seed_label}", color="#526070", fontsize=10)
    fig.text(.07, 1 - .91 / height, subtitle, color="#526070", fontsize=9)
    if scenario_legend:
        scenarios = SCENARIOS if scenario_legend is True else scenario_legend
        handles = [Line2D([0], [0], color=COLORS[s], linestyle=STYLES[s], lw=2.5, label=s) for s in scenarios]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.97, .979), frameon=False, ncol=4, fontsize=9)


def draw_series(ax, data, orientation, metric, scenarios=SCENARIOS, multiplier=1):
    for scenario in scenarios:
        subset = data.trajectories[(data.trajectories.orientation == orientation) & (data.trajectories.scenario == scenario)]
        for _, seed in subset.groupby("seed"):
            ax.plot(seed["round"], seed[metric] * multiplier, color=COLORS[scenario], linestyle=STYLES[scenario], lw=.9, alpha=.42 if data.pilot else .22)
        mean = subset.groupby("round")[metric].agg(lambda x: np.mean(x.to_numpy()))
        ax.plot(mean.index, mean * multiplier, color=COLORS[scenario], linestyle=STYLES[scenario], lw=2.1)
    if data.endpoint <= data.through_round:
        ax.axvline(data.endpoint, color="#94A0AC", lw=.9, linestyle=(0, (3, 3)), zorder=0)
    ax.set_xlim(0, data.through_round)
    ax.grid(axis="y", alpha=.24, color="#A7B3BF", linewidth=.6)
    ax.tick_params(labelsize=9)


def row_bounds(axes, zero=False, symmetric=False):
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    if symmetric:
        bound = max(abs(low), abs(high), .01)
        low, high = -bound, bound
    elif zero:
        low, high = 0, max(high, .01)
    for ax in axes:
        ax.set_ylim(low, high)


def trajectories_figure(data):
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.7), sharex=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.12, top=.80, wspace=.22, hspace=.28)
    header(fig, data, "Belief and content dynamics", "Thin lines: individual seeds. Thick lines: seed means. Dashed vertical line: main endpoint.")
    for col, orientation in enumerate(ORIENTATIONS):
        axes[0, col].set_title(TITLES[orientation], loc="left", pad=12)
        for row, metric in enumerate(("mean_signed_belief", "cumulative_content_balance")):
            draw_series(axes[row, col], data, orientation, metric)
            axes[row, col].axhline(0, color="#A7B3BF", lw=.7, zorder=0)
        axes[1, col].set_xlabel("Round")
    axes[0, 0].set_ylabel("Mean signed belief")
    axes[1, 0].set_ylabel("Cumulative content balance")
    for row in axes:
        row_bounds(row, symmetric=True)
    fig.text(.08, .035, "Balanced: mirrored assignments averaged within each seed. No confidence bands; all available seeds are shown.", fontsize=9, color="#526070")
    return fig


def effects_figure(data):
    if data.contrasts is None:
        raise ValueError("This batch does not contain endpoint effect tables")
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.97, bottom=.24, top=.72, wspace=.18)
    intervals = not data.pilot
    subtitle = "Dots: paired seed effects. Diamonds: stored seed means."
    subtitle += " Bars: stored 95% t intervals where available." if intervals else " Intervals omitted for pilot preview."
    header(fig, data, f"Leader effects at round {data.endpoint}", subtitle, scenario_legend=False)
    extent = [0.01]
    for col, orientation in enumerate(ORIENTATIONS):
        ax = axes[col]
        ax.set_title(TITLES[orientation], loc="left", pad=12)
        ax.axvline(0, color="#94A0AC", lw=1)
        for row, effect in enumerate(EFFECTS):
            y = 2 - row
            points = data.contrasts[(data.contrasts.orientation == orientation) & (data.contrasts.contrast == effect)].sort_values("seed")
            summary = data.summary[(data.summary.orientation == orientation) & (data.summary.contrast == effect)].iloc[0]
            offsets = np.linspace(-.09, .09, len(points)) if len(points) > 1 else [0]
            ax.scatter(points.effect, y + np.asarray(offsets), color=EFFECT_COLORS[row], s=26, alpha=.65, edgecolors="none", zorder=3)
            ax.scatter([summary["mean"]], [y], color=EFFECT_COLORS[row], marker="D", s=58, edgecolors="white", linewidth=.8, zorder=4)
            extent.extend(np.abs(points.effect).tolist())
            if intervals and summary.interval_status == "complete":
                low, high = summary.ci95_low, summary.ci95_high
                if not np.isfinite([low, high]).all() or low > high:
                    raise ValueError("Invalid stored interval")
                ax.hlines(y, low, high, color=EFFECT_COLORS[row], lw=1.8)
                ax.vlines([low, high], y-.05, y+.05, color=EFFECT_COLORS[row], lw=1.2)
                extent.extend([abs(low), abs(high)])
        ax.set_ylim(-.55, 2.55)
        ax.set_yticks([2, 1, 0], EFFECT_LABELS)
        ax.grid(axis="x", alpha=.22)
        ax.set_xlabel("Difference in mean signed belief")
    bound = max(extent) * 1.2
    axes[0].set_xlim(-bound, bound)
    fig.text(.14, .105, "Interaction = (baseline − platform) − (opleader − null). Balanced mirrors are averaged within seed.", fontsize=9, color="#526070")
    fig.text(.14, .058, "Negative values indicate a more negative contribution; their sign is not a universal label for weaker influence.", fontsize=9, color="#526070")
    return fig


def diagnostics_figure(data, ties=False):
    specs = [("accepted_addition_count", "Accepted tie additions\nper round", ("platform", "baseline"), 1),
             ("accepted_removal_count", "Accepted tie removals\nper round", ("platform", "baseline"), 1)] if ties else [
        ("out_of_network_exposure_count", "Beyond-network exposures\nper round", ("platform", "baseline"), 1),
        ("leader_exposure_count", "Leader-source exposures\nper round", ("opleader", "baseline"), 1),
        ("capacity_binding_rate", "Consumers with binding\nattention capacity (%)", ("platform", "baseline"), 100)]
    fig, axes = plt.subplots(len(specs), 3, figsize=(13.4, 7.5 if ties else 10.2), sharex=True)
    fig.subplots_adjust(left=.095, right=.97, bottom=.11, top=.81, wspace=.23, hspace=.28)
    header(fig, data, "Adaptive tie dynamics" if ties else "Exposure and attention diagnostics",
           "Thin lines: individual seeds. Thick lines: seed means. Only applicable scenarios are drawn.",
           scenario_legend=("platform", "baseline") if ties else ("opleader", "platform", "baseline"))
    for row, (metric, label, scenarios, multiplier) in enumerate(specs):
        for col, orientation in enumerate(ORIENTATIONS):
            draw_series(axes[row, col], data, orientation, metric, scenarios, multiplier)
            if row == 0:
                axes[row, col].set_title(TITLES[orientation], loc="left", pad=12)
            if row == len(specs)-1:
                axes[row, col].set_xlabel("Round")
        axes[row, 0].set_ylabel(label)
        row_bounds(axes[row], zero=True)
    footer = "Counts describe accepted changes, not probabilities. Additions and removals are displayed separately." if ties else (
        "Exposure counts refer to consumer–message events, not unique users. Binding rate denominator: all agents.")
    fig.text(.095, .034, footer, fontsize=9, color="#526070")
    return fig


def render_figures(data, output=None, formats=("png", "pdf", "svg"), dpi=180):
    if dpi < 72 or not formats or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("Use png/pdf/svg and dpi >= 72")
    output = Path(output).resolve() if output else data.batch / "figures"
    output.mkdir(parents=True, exist_ok=True)
    style()
    prefix = data.prefix + f"__through{data.through_round}"
    figures = [("belief_content_trajectories", trajectories_figure,
                "Mean signed belief and cumulative content balance over rounds, across four scenarios and three leader orientations."),
               ("exposure_attention_diagnostics", lambda d: diagnostics_figure(d),
                "Beyond-network exposure, leader-source exposure, and attention capacity binding over rounds, split by leader orientation."),
               ("adaptive_tie_diagnostics", lambda d: diagnostics_figure(d, ties=True),
                "Accepted tie additions and removals over rounds for platform and baseline, split by leader orientation.")]
    if data.contrasts is not None:
        figures.insert(1, (f"round{data.endpoint}_leader_effects", effects_figure,
                           "Paired seed effects of leaders without platform, with platform, and their interaction, split by orientation."))
    trajectory_path = output / f"{prefix}__trajectory_plot_data.csv"
    data.trajectories.to_csv(trajectory_path, index=False)
    artifacts = []
    status = "PILOT: technical preview, not retained findings." if data.pilot else "Retained simulation results conditional on model assumptions."
    common = (f"{status} {data.setting.label}; {data.seed_label}. "
              f"Main endpoint: round {data.endpoint}; trajectory display: rounds 0–{data.through_round}. "
              "Balanced assignments are averaged within seed; shared controls are reused, not additional replications. ")
    index = ["# Experiment figures", "", common, "",
             "Source tables are read without running simulations or recomputing effects/intervals. "
             "Only mechanical selection, within-seed mirror averaging, and seed-mean display are performed.", ""]
    for name, draw, description in figures:
        figure = draw(data)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        figure_boxes = [text.get_window_extent(renderer) for text in figure.texts]
        panel_boxes = [ax.title.get_window_extent(renderer) for ax in figure.axes if ax.title.get_text()]
        # loc='left' titles are separate Matplotlib artists.
        panel_boxes.extend(ax._left_title.get_window_extent(renderer) for ax in figure.axes if ax._left_title.get_text())
        for i, box in enumerate(figure_boxes):
            if any(box.overlaps(other) for other in figure_boxes[i+1:] + panel_boxes):
                raise ValueError(f"Overlapping figure text in {name}")
        stem = prefix + "__" + name
        exports = []
        for format in formats:
            path = output / f"{stem}.{format}"
            figure.savefig(path, dpi=dpi, facecolor="white")
            exports.append(path)
        plt.close(figure)
        is_effect = "leader_effects" in name
        encoding = ("Points are individual paired seeds; diamonds are stored means. "
                    + ("Pilot intervals are omitted." if data.pilot else "Bars use the stored 95% t intervals, when available.")) if is_effect else (
                        "Thin lines show seeds; thick lines show means. Scenario colors and line styles are consistent. "
                        "Vertical markers indicate the main endpoint; no confidence bands are drawn.")
        if is_effect:
            data_path = output / f"{stem}__plot_data.csv"
            data.contrasts.to_csv(data_path, index=False)
        else:
            data_path = trajectory_path
        caption = common + description + " " + encoding
        if not is_effect:
            caption += " Undefined observations remain gaps; inapplicable scenarios are omitted."
        if "belief_content" in name:
            caption += " Content balance is (support − oppose)/(support + oppose), cumulative from initialization; round 0 is undefined."
        if "exposure" in name:
            caption += " Exposures count consumer–message events; channels use pre-update ties. Leader-source counts sum tied and beyond-network exposures."
        if data.maximum_source_round > data.through_round:
            caption += f" Some source trajectories extend to {data.maximum_source_round}; this display uses only their common/requested horizon."
        caption += " The figures do not isolate causal contributions of individual platform rules or establish empirical validity."
        artifacts.append({"name": name, "caption": caption, "alt_text": description + " " + encoding,
                          "plot_data": str(data_path), "exports": [{"path": str(p), "sha256": file_hash(p)} for p in exports]})
        index.extend([f"## {name.replace('_', ' ')}", "", caption, ""])
        png = next((p for p in exports if p.suffix == ".png"), None)
        if png:
            index.extend([f"![{description}]({png.as_posix()})", ""])
        index.append(" · ".join(f"[{p.suffix[1:].upper()}]({p.as_posix()})" for p in exports))
        index.extend(["", f"[Plot data]({data_path.as_posix()})", ""])
    provenance = {"batch": str(data.batch), "kind": data.manifest["kind"], "design_status": data.manifest["design_status"],
                  "setting": data.setting.__dict__, "through_round": data.through_round, "analysis_round": data.endpoint,
                  "seed_ids": list(data.seeds), "input_files": {str(p): file_hash(p) for p in data.inputs},
                  "simulation_code_fingerprint": data.manifest["code"]["fingerprint"],
                  "plotting_source": str(Path(__file__).resolve()), "plotting_source_sha256": file_hash(__file__),
                  "matplotlib_version": matplotlib.__version__, "formats": list(formats), "dpi": dpi,
                  "effect_interval_policy": "omitted_for_pilot" if data.pilot else "stored_95pct_t_intervals",
                  "trajectory_interval_policy": "none; individual seeds and means", "artifacts": artifacts}
    provenance["layout_checks"] = "Rendered figure text checked for overlaps with other figure text and panel titles."
    (output / f"{prefix}__figure_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    index_path = output / f"{prefix}__figure_index.md"
    index_path.write_text("\n".join(index), encoding="utf-8")
    return index_path
