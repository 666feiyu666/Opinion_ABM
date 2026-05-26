from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ensure_directory

MODE_COLORS = {
    "balanced": "#6b7280",
    "positive": "#b91c1c",
    "negative": "#1d4ed8",
}

PERTURBATION_ORDER = [
    "nominal",
    "alpha2_low",
    "alpha2_high",
    "beta1_diff_low",
    "beta1_diff_high",
    "w_l_low",
    "w_l_high",
    "lambda_L_low",
    "lambda_L_high",
]

PERTURBATION_LABELS = {
    "nominal": "Reference",
    "alpha2_low": "Posting advantage: low",
    "alpha2_high": "Posting advantage: high",
    "beta1_diff_low": "Diffusion visibility: low",
    "beta1_diff_high": "Diffusion visibility: high",
    "w_l_low": "Exposure weight: low",
    "w_l_high": "Exposure weight: high",
    "lambda_L_low": "Network attraction: low",
    "lambda_L_high": "Network attraction: high",
}

METRIC_SPECS = [
    ("delta_final_mean_opinion", "Directional effect", "Delta final mean opinion"),
    ("delta_extremist_ratio", "Extremity effect", "Delta extremist ratio"),
    ("delta_homophily_ratio", "Homophily effect", "Delta homophily ratio"),
]


def _ordered_perturbations(frame: pd.DataFrame) -> list[str]:
    perturbations = set(frame["perturbation_id"].drop_duplicates().tolist())
    ordered = [item for item in PERTURBATION_ORDER if item in perturbations]
    ordered.extend(sorted(item for item in perturbations if item not in ordered))
    return ordered


def plot_core_effects(summary_df: pd.DataFrame):
    core = summary_df[summary_df["study_name"] == "core_robustness"].copy()
    topologies = sorted(core["topology"].unique())
    perturbations = _ordered_perturbations(core)
    fig, axes = plt.subplots(
        len(topologies),
        len(METRIC_SPECS),
        figsize=(5.8 * len(METRIC_SPECS), 5.1 * len(topologies)),
        squeeze=False,
        sharey=True,
    )
    y_values = np.arange(len(perturbations))
    offsets = {"balanced": 0.22, "positive": 0.0, "negative": -0.22}

    for row_index, topology in enumerate(topologies):
        topology_df = core[core["topology"] == topology]
        for column_index, (metric, title, ylabel) in enumerate(METRIC_SPECS):
            ax = axes[row_index, column_index]
            for mode in ["balanced", "positive", "negative"]:
                subset = (
                    topology_df[topology_df["leader_mode"] == mode]
                    .set_index("perturbation_id")
                    .reindex(perturbations)
                )
                if subset.empty:
                    continue
                values = subset[f"{metric}_mean"].to_numpy(dtype=float)
                lower = values - subset[f"{metric}_ci_low"].to_numpy(dtype=float)
                upper = subset[f"{metric}_ci_high"].to_numpy(dtype=float) - values
                ax.errorbar(
                    values,
                    y_values + offsets[mode],
                    xerr=np.vstack([lower, upper]),
                    marker="o",
                    linestyle="none",
                    markersize=5,
                    elinewidth=1.4,
                    capsize=2.5,
                    color=MODE_COLORS[mode],
                    label=mode,
                )
            ax.axvline(0.0, color="#111827", linestyle=":", linewidth=0.9)
            ax.set_title(f"{topology}: {title}")
            ax.set_xlabel(ylabel)
            ax.set_yticks(y_values)
            ax.set_yticklabels([PERTURBATION_LABELS.get(item, item) for item in perturbations])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2)
            if column_index != 0:
                ax.tick_params(axis="y", labelleft=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Matched effects under leader-mechanism perturbations (95% CI)", y=0.995)
    fig.legend(
        handles,
        labels,
        title="Leader orientation",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def plot_share_gradient(summary_df: pd.DataFrame):
    gradient = summary_df[summary_df["study_name"] == "share_gradient"].copy()
    topologies = sorted(gradient["topology"].unique())
    perturbations = _ordered_perturbations(gradient)
    modes = ["positive", "negative"]
    share_values = sorted(gradient["leader_share"].unique())
    metric_limits = {}
    for metric, _, _ in METRIC_SPECS:
        values = gradient[f"{metric}_mean"].to_numpy(dtype=float)
        if metric == "delta_final_mean_opinion":
            bound = float(np.nanmax(np.abs(values)))
            metric_limits[metric] = (-bound, bound, "RdBu_r")
        else:
            metric_limits[metric] = (0.0, float(np.nanmax(values)), "YlOrRd")
    fig, axes = plt.subplots(
        len(topologies),
        len(METRIC_SPECS),
        figsize=(5.0 * len(METRIC_SPECS), 5.9 * len(topologies)),
        squeeze=False,
    )

    for row_index, topology in enumerate(topologies):
        topology_df = gradient[gradient["topology"] == topology]
        for column_index, (metric, title, ylabel) in enumerate(METRIC_SPECS):
            ax = axes[row_index, column_index]
            matrix = np.full((len(modes) * len(perturbations), len(share_values)), np.nan)
            rows = [(mode, perturbation) for mode in modes for perturbation in perturbations]
            for matrix_row, (mode, perturbation) in enumerate(rows):
                subset = topology_df[
                    (topology_df["leader_mode"] == mode)
                    & (topology_df["perturbation_id"] == perturbation)
                ].set_index("leader_share")
                for matrix_column, share in enumerate(share_values):
                    if share in subset.index:
                        matrix[matrix_row, matrix_column] = subset.loc[share, f"{metric}_mean"]
            vmin, vmax, cmap = metric_limits[metric]
            image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            threshold = (vmin + vmax) / 2.0
            for matrix_row in range(matrix.shape[0]):
                for matrix_column in range(matrix.shape[1]):
                    value = matrix[matrix_row, matrix_column]
                    if np.isnan(value):
                        continue
                    color = "white" if value > threshold and metric != "delta_final_mean_opinion" else "#111827"
                    if metric == "delta_final_mean_opinion" and abs(value) > vmax * 0.58:
                        color = "white"
                    ax.text(matrix_column, matrix_row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
            ax.set_title(f"{topology}: {title}")
            ax.set_xlabel("Leader share")
            ax.set_xticks(np.arange(len(share_values)))
            ax.set_xticklabels([f"{share * 100:.0f}%" for share in share_values])
            ax.set_yticks(np.arange(len(rows)))
            ax.set_yticklabels(
                [f"{mode.capitalize()} | {PERTURBATION_LABELS.get(perturbation, perturbation)}" for mode, perturbation in rows]
            )
            if column_index != 0:
                ax.tick_params(axis="y", labelleft=False)
            fig.colorbar(image, ax=ax, shrink=0.75, pad=0.02, label=ylabel)

    fig.suptitle("Share-gradient matched effects: magnitude and high-share plateau", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def save_sensitivity_figures(summary_df: pd.DataFrame, output_root: str | Path) -> dict[str, Path]:
    if summary_df.empty or "study_name" not in summary_df.columns:
        return {}
    figures_dir = ensure_directory(Path(output_root) / "figures")
    saved_paths = {}
    if (summary_df["study_name"] == "core_robustness").any():
        core_path = figures_dir / "core_matched_effects.png"
        fig = plot_core_effects(summary_df)
        fig.savefig(core_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths["core_figure"] = core_path
    if (summary_df["study_name"] == "share_gradient").any():
        share_path = figures_dir / "share_gradient_effects.png"
        fig = plot_share_gradient(summary_df)
        fig.savefig(share_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths["share_figure"] = share_path
    return saved_paths
