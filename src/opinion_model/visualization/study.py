"""Study-wide static figures, retaining seed-level data and figure provenance."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from opinion_model.visualization.results import (
    style, COLORS, STYLES, SCENARIOS, ORIENTATIONS, TITLES, EFFECTS, EFFECT_COLORS,
)
from opinion_model.experiments.review import digest_file, require

EFFECT_NAMES = ("Leader effect without platform", "Leader effect with platform", "Platform × leader interaction")
NETWORK = ("edge_count", "structural_top_in_degree_share", "undirected_clustering", "largest_weak_component_fraction")
NETWORK_LABELS = ("Directed edge count", "Top 3% in-degree share", "Undirected clustering", "Largest weak component\n(population fraction)")
SCENARIO_MARKERS = dict(zip(SCENARIOS, ("o", "s", "^", "D")))


def canvas(rows, cols, title, subtitle, height=None):
    style()
    fig, axes = plt.subplots(rows, cols, figsize=(13.6, height or (2.55 * rows + 1.65)), squeeze=False)
    fig.subplots_adjust(left=.105, right=.975, bottom=.10, top=1-1.60/fig.get_figheight(), wspace=.24, hspace=.40)
    fig.suptitle(title, x=.095, y=.976, ha="left", fontsize=17, fontweight="bold", color="#1C2B3B")
    fig.text(.095, 1-.90/fig.get_figheight(), subtitle, fontsize=9.5, color="#526070")
    return fig, axes


def axes_finish(axes, symmetric=False):
    for row in axes:
        low = min(ax.get_ylim()[0] for ax in row)
        high = max(ax.get_ylim()[1] for ax in row)
        if symmetric:
            high = max(abs(low), abs(high), .005)
            low = -high
        for ax in row:
            ax.set_ylim(low, high)
            ax.grid(axis="y", alpha=.22)
            ax.tick_params(labelsize=9)


def effect_points(ax, points, summary, factor, levels, color):
    for x, level in enumerate(levels):
        p = points.loc[points[factor] == level].sort_values("seed")
        s = summary.loc[summary[factor] == level]
        require(len(s) == 1 and len(p) == int(s.iloc[0].seed_count), "Missing effect level")
        s = s.iloc[0]
        ax.scatter(x + np.linspace(-.11, .11, len(p)), p.effect, s=13, alpha=.45, color=color, linewidths=0, zorder=3)
        ax.errorbar(x, s["mean"], yerr=[[s["mean"]-s.ci95_low], [s.ci95_high-s["mean"]]],
                    fmt="D", markersize=5, color=color, capsize=4, lw=1.6, zorder=4)
    ax.axhline(0, color="#8794A1", lw=.8)
    ax.set_xlim(-.5, len(levels)-.5)
    ax.set_xticks(range(len(levels)))


def main_effects(points, summary, orientation):
    populations = sorted(points.population.unique())
    levels = sorted(points.leader_share.unique())
    fig, axes = canvas(3, len(populations), f"Main experiment · {TITLES[orientation].lower()}",
                       "Round 50 · BA · reach = 0.04 · 10 paired seeds · small dots: seeds; diamonds/bars: mean and 95% t interval")
    for row, effect in enumerate(EFFECTS):
        for col, n in enumerate(populations):
            mask = (points.orientation == orientation) & (points.contrast == effect) & (points.population == n)
            smask = (summary.orientation == orientation) & (summary.contrast == effect) & (summary.population == n)
            effect_points(axes[row,col], points[mask], summary[smask], "leader_share", levels, EFFECT_COLORS[row])
            axes[row,col].set_xticklabels([f"{100*v:g}%" for v in levels])
            if row == 0:
                axes[row,col].set_title(f"N = {n:,}", loc="left", pad=10)
            if row == 2:
                axes[row,col].set_xlabel("Nominal leader share")
        axes[row,0].set_ylabel(EFFECT_NAMES[row] + "\n(signed-belief units)")
    axes_finish(axes, symmetric=True)
    fig.text(.095, .025, "Interaction = (baseline − platform) − (opleader − null). Balanced mirrors form one observation per seed.", fontsize=9)
    return fig


def supporting_effects(points, summary, factor, levels, title, subtitle):
    fig, axes = canvas(3, 3, title, subtitle + "\nSmall dots: paired seeds; diamonds/bars: means and 95% t intervals; 10 seeds per condition.")
    for row, effect in enumerate(EFFECTS):
        for col, orientation in enumerate(ORIENTATIONS):
            p = points[(points.orientation == orientation)&(points.contrast == effect)]
            s = summary[(summary.orientation == orientation)&(summary.contrast == effect)]
            effect_points(axes[row,col], p, s, factor, levels, EFFECT_COLORS[row])
            labels = [str(v).upper() if isinstance(v,str) else f"{v:g}" for v in levels]
            axes[row,col].set_xticklabels(labels)
            if row == 0:
                axes[row,col].set_title(TITLES[orientation], loc="left", pad=10)
            if row == 2:
                axes[row,col].set_xlabel({"reach":"Beyond-network availability probability", "topology":"Initial topology", "analysis_round":"Observation round"}[factor])
            if factor == "analysis_round":
                axes[row,col].axvline(levels.index(50), color="#94A0AC", ls="--", lw=.8)
        axes[row,0].set_ylabel(EFFECT_NAMES[row]+"\n(signed-belief units)")
    axes_finish(axes, symmetric=True)
    fig.text(.095,.025,"Signs retain their directional meaning. Intervals describe Monte Carlo variation under fixed model conditions.",fontsize=9)
    return fig


def diagnostic_comparison(frame, factor, levels, metrics, labels, title, subtitle, scenarios=SCENARIOS, initial=None):
    fig, axes = canvas(len(metrics),3,title,subtitle+"\nSmall dots: seeds; larger markers: means. No confidence intervals are drawn.",height=2.55*len(metrics)+2.0)
    for row,(metric,label) in enumerate(zip(metrics,labels)):
        for col,orientation in enumerate(ORIENTATIONS):
            ax=axes[row,col]
            for index,scenario in enumerate(scenarios):
                selected=frame[(frame.scenario==scenario)&frame.orientation.isin([orientation,"none"])]
                means=[]
                offset=(index-(len(scenarios)-1)/2)*.15
                for x,level in enumerate(levels):
                    p=selected[selected[factor]==level].sort_values("seed")
                    require(not p.seed.duplicated().any() and len(p)>0,"Diagnostic seed duplication/missing level")
                    means.append(float(np.mean(p[metric])))
                    ax.scatter(x+offset+np.linspace(-.035,.035,len(p)),p[metric],s=8,alpha=.23,color=COLORS[scenario],linewidths=0)
                ax.plot(np.arange(len(levels))+offset,means,marker=SCENARIO_MARKERS[scenario],linestyle=STYLES[scenario],
                        ms=4,lw=1.25,color=COLORS[scenario])
            if initial is not None:
                baseline=[np.mean(initial.loc[initial[factor]==level,metric]) for level in levels]
                ax.plot(range(len(levels)),baseline,"x--",color="#252525",lw=1,ms=5)
            ax.set_xticks(range(len(levels)),[str(v).upper() if isinstance(v,str) else f"{v:g}" for v in levels])
            ax.set_xlim(-.5,len(levels)-.5)
            if row==0:
                ax.set_title(TITLES[orientation],loc="left",pad=10)
            if row==len(metrics)-1:
                ax.set_xlabel("Initial topology" if factor=="topology" else "Beyond-network availability probability")
        axes[row,0].set_ylabel(label)
    axes_finish(axes)
    for row_index, row in enumerate(axes):
        for ax in row:
            ax.set_ylim(bottom=0)
            if metrics[row_index] == "largest_weak_component_fraction":
                ax.set_ylim(top=1.03)
    handles=[Line2D([0],[0],color=COLORS[s],marker=SCENARIO_MARKERS[s],ls=STYLES[s],label=s) for s in scenarios]
    if initial is not None:
        handles.append(Line2D([0],[0],color="#252525",marker="x",ls="--",label="round 0 (shared network)"))
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.54,.025),ncol=len(handles),frameon=False,fontsize=9)
    return fig


def horizon_structure(frame):
    fig,axes=canvas(4,3,"Network structure through round 100",
                    "Reference: N = 500 · BA · leaders = 3% · reach = 0.04 · 10 seeds · thin lines: seeds; thick lines: means",height=12.2)
    for row,(metric,label) in enumerate(zip(NETWORK,NETWORK_LABELS)):
        for col,orientation in enumerate(ORIENTATIONS):
            ax=axes[row,col]
            for scenario in SCENARIOS:
                selected=frame[(frame.scenario==scenario)&frame.orientation.isin([orientation,"none"])]
                for _,seed in selected.groupby("seed"):
                    ax.plot(seed["round"],seed[metric],lw=.6,alpha=.18,color=COLORS[scenario],ls=STYLES[scenario])
                means=selected.groupby("round")[metric].agg(lambda v: np.mean(v.to_numpy()))
                ax.plot(means.index,means,lw=1.8,color=COLORS[scenario],ls=STYLES[scenario],marker=SCENARIO_MARKERS[scenario],ms=3)
            ax.axvline(50,color="#94A0AC",ls="--",lw=.8)
            ax.set_xlim(0,100)
            ax.set_xticks([0,30,50,75,100])
            if row==0:
                ax.set_title(TITLES[orientation],loc="left",pad=10)
            if row==3:
                ax.set_xlabel("Round (lines join observed checkpoints)")
        axes[row,0].set_ylabel(label)
    axes_finish(axes)
    fig.legend(handles=[Line2D([0],[0],color=COLORS[s],ls=STYLES[s],label=s) for s in SCENARIOS],
               loc="lower center",bbox_to_anchor=(.54,.025),ncol=4,frameon=False)
    return fig


class FigureWriter:
    def __init__(self, output, provenance, dpi=180):
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.provenance=provenance
        self.dpi=dpi
        self.artifacts=[]

    def save(self, name, figure, caption, alt, tables):
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
        titles=[a._left_title for a in figure.axes if a._left_title.get_text()]
        texts=[t for t in figure.texts if t.get_text()]+titles
        boxes=[t.get_window_extent(renderer) for t in texts]
        for i,box in enumerate(boxes):
            require(not any(box.overlaps(other) for other in boxes[i+1:]),f"Overlapping headings in {name}")
            require(figure.bbox.contains(box.x0,box.y0) and figure.bbox.contains(box.x1,box.y1),f"Clipped heading in {name}")
        exports=[]
        for suffix in ("png","pdf","svg"):
            path=self.output/f"{name}.{suffix}"
            figure.savefig(path,dpi=self.dpi,facecolor="white")
            exports.append({"path":str(path.resolve()),"sha256":digest_file(path)})
        plt.close(figure)
        data=[]
        for label,table in tables.items():
            path=self.output/f"{name}__{label}.csv"
            table.to_csv(path,index=False)
            data.append({"path":str(path.resolve()),"sha256":digest_file(path)})
        self.artifacts.append({"name":name,"caption":caption,"alt_text":alt,"exports":exports,"plot_data":data,
                               "layout_check":"Rendered headings contained and nonoverlapping"})

    def finish(self):
        result={**self.provenance,"matplotlib_version":matplotlib.__version__,"dpi":self.dpi,
                "artifacts":self.artifacts,"plotting_source":str(Path(__file__).resolve()),"plotting_source_sha256":digest_file(__file__)}
        (self.output/"study_figure_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        lines=["# Study result figures","","All figures use retained simulations; source tables and reproduction details are in the manifest.",""]
        for item in self.artifacts:
            lines += [f"## {item['name']}","",item["caption"],"",f"![{item['alt_text']}]({item['exports'][0]['path'].replace(chr(92),'/')})",""]
            lines += [" · ".join(f"[{Path(e['path']).suffix[1:].upper()}]({Path(e['path']).as_posix()})" for e in item["exports"]),""]
        (self.output/"figure_index.md").write_text("\n".join(lines),encoding="utf-8")
        return self.output/"figure_index.md"
