"""Audit retained batches and reproduce study-wide result figures without simulation."""
import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import pandas as pd
from opinion_model.experiments.review import audit_study, seed_display, descriptive_summary, digest_file, read_csv
from opinion_model.experiments.analysis import endpoint_metrics, paired_contrasts, summarize_contrasts, OUTCOMES
from opinion_model.visualization.study import (
    FigureWriter, main_effects, supporting_effects, diagnostic_comparison, horizon_structure, NETWORK, NETWORK_LABELS,
)
from opinion_model.visualization.results import load_figure_data, render_figures, ORIENTATIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main",type=Path,default=ROOT/"outputs/main_experiment/main_grid__batch-01")
    for kind in ("reach","topology","horizon"):
        parser.add_argument(f"--{kind}",type=Path,default=ROOT/f"outputs/supporting_experiments/{kind}__batch-01")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/study_results/study__batch-01")
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    paths={k:getattr(args,k) for k in ("main","reach","topology","horizon")}
    batches,audit=audit_study(paths)
    (output/"retained_data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    pd.DataFrame(audit["batches"]).to_csv(output/"batch_coverage.csv",index=False)
    print(f"Audit passed: {audit['unique_trajectories']} unique trajectories",flush=True)
    writer=FigureWriter(output/"figures",{"simulation_fingerprint":audit["simulation_fingerprint"],
        "input_sha256":audit["input_sha256"],"driver_sha256":digest_file(__file__),
        "analysis_source_sha256":digest_file(ROOT/"src/opinion_model/experiments/review.py")})
    p={k:b.contrasts.query("metric == 'mean_signed_belief'") for k,b in batches.items() if b.contrasts is not None}
    s={k:b.summary.query("metric == 'mean_signed_belief'") for k,b in batches.items() if b.summary is not None}
    for k in p:
        p[k].to_csv(output/f"{k}_primary_seed_effects.csv",index=False)
        s[k].to_csv(output/f"{k}_primary_effect_summary.csv",index=False)
    common="Ten paired seeds per condition. Means and two-sided 95% Student-t intervals use seed-level contrasts. Balanced mirrors are averaged within seed. The outcome is population mean signed belief. "
    boundary="Effects are conditional on this model and its fixed parameters; they do not establish empirical validity or identify individual platform rules separately."
    for orientation in ORIENTATIONS:
        writer.save(f"main_round50_effects_{orientation}",main_effects(p['main'],s['main'],orientation),
                    common+"BA, reach 0.04, round 50; populations 500/750/1000 and nominal leader shares 1/3/5%. Small points show all seed effects. "+boundary,
                    f"Three effect rows across three population panels, for {orientation} leaders and three leader shares.",
                    {"seeds":p['main'].query("orientation == @orientation"),"summary":s['main'].query("orientation == @orientation")})
    for kind,factor,levels,title,settings in (
        ("reach","reach",[0.,.04,.08],"Leader effects across platform reach","N = 500 · BA · leaders = 3% · round 50"),
        ("topology","topology",["ba","er","ws","sbm"],"Leader effects across initial topologies","N = 500 · leaders = 3% · reach = 0.04 · round 50")):
        writer.save(f"{kind}_round50_effects",supporting_effects(p[kind],s[kind],factor,levels,title,settings),
                    common+settings+". "+("Reach zero retains finite attention and adaptive ties. " if kind=="reach" else "Topology changes leader identities and initial visibility under a fixed selection rule. ")+boundary,
                    f"Three effect rows and positive, negative and balanced columns across {kind} conditions.",{"seeds":p[kind],"summary":s[kind]})
    metrics=["out_of_network_exposure_count","capacity_binding_rate","accepted_addition_count","accepted_removal_count"]
    reach=seed_display(batches['reach'].endpoints.query("scenario in ['platform','baseline']"),metrics)
    writer.save("reach_round50_process_diagnostics",diagnostic_comparison(reach,"reach",[0.,.04,.08],metrics,
        ["Beyond-network exposures\n(consumer-message events)","Capacity-binding fraction\n(all consumers)","Accepted tie additions\nper round","Accepted tie removals\nper round"],
        "Exposure, attention and tie changes across reach","N = 500 · BA · leaders = 3% · round 50 · 10 seeds",scenarios=("platform","baseline")),
        "Round-50 observations for platform and baseline; ten seeds and within-seed balanced averaging. Thin/small points show seeds; scenario markers and lines show means, without intervals. Exposure channels use pre-update ties; the capacity denominator is all consumers. These diagnostics show realized processes, not separate causal effects.",
        "Four diagnostic rows compare three reach levels for positive, negative and balanced conditions.",{"seeds":reach})
    topo=seed_display(batches['topology'].endpoints,NETWORK)
    initial=seed_display(batches['topology'].rounds.query("round == 0 and scenario == 'null'"),NETWORK)
    writer.save("topology_round50_structure",diagnostic_comparison(topo,"topology",["ba","er","ws","sbm"],NETWORK,NETWORK_LABELS,
        "Initial and round-50 network structure","N = 500 · leaders = 3% · reach = 0.04 · 10 seeds",initial=initial),
        "Seed observations and scenario means at round 50, with shared round-0 network means as black crosses. No intervals. Top 3% is the current structural group, not the fixed leaders. Clustering uses the undirected projection including isolates; largest weak component is divided by population. "+boundary,
        "Four structural indicators across four topologies; three leader orientations; initial networks provide a shared reference.",{"endpoint_seeds":topo,"initial_seeds":initial})
    horizon=batches['horizon']
    hs=seed_display(horizon.rounds.loc[horizon.rounds.structural_top_count.notna()],NETWORK)
    writer.save("horizon_network_structure",horizon_structure(hs),
        "Reference setting N=500, BA, 3% leaders, reach 0.04. Thin lines show ten seeds and thick lines show means at rounds 0,10,30,50,75,100. Connecting lines do not imply observations between checkpoints. The vertical line marks the main endpoint at 50. "+boundary,
        "Four network indicators through round 100 across the four scenarios and three leader orientations.",{"seed_checkpoints":hs})
    comparisons=read_csv(batches['main'].path/"main_comparison_plan.csv")
    ids=set(horizon.plan.run_id)
    comparisons=comparisons.loc[comparisons.baseline_run_id.isin(ids)].copy()
    hpoints=[]
    for r in (30,50,75,100):
        cp=comparisons.assign(experiment="horizon",analysis_round=r)
        hpoints.append(paired_contrasts(endpoint_metrics(horizon.rounds,r),cp))
    hc=pd.concat(hpoints,ignore_index=True)
    hsummary=summarize_contrasts(hc,horizon.manifest['expected_seeds'])
    hc.to_csv(output/"horizon_seed_effects.csv",index=False)
    hsummary.to_csv(output/"horizon_effect_summary.csv",index=False)
    hp=hc.query("metric == 'mean_signed_belief'"); ha=hsummary.query("metric == 'mean_signed_belief'")
    writer.save("horizon_effect_checkpoints",supporting_effects(hp,ha,"analysis_round",[30,50,75,100],
        "Leader effects after the main endpoint","N = 500 · BA · leaders = 3% · reach = 0.04 · vertical line: round 50"),
        common+"Reference trajectories reused at rounds 30,50,75,100; these repeated observations are not new replications. Display positions are categorical checkpoints. Main endpoint remains round 50. "+boundary,
        "Paired leader effects and interaction at four observation rounds; later effects are supporting evidence.",{"seeds":hp,"summary":ha})
    secondary=[]
    for kind,batch in batches.items():
        frame=batch.endpoints if kind!='horizon' else batch.rounds[batch.rounds['round'].isin([30,50,75,100])]
        display=seed_display(frame,OUTCOMES)
        ds=descriptive_summary(display,OUTCOMES)
        ds.insert(0,"experiment",kind)
        secondary.append(ds)
    pd.concat(secondary,ignore_index=True).to_csv(output/"scenario_outcome_summary.csv",index=False)
    writer.finish()
    reference=load_figure_data(batches['main'].path,reference_only=True,through_round=50)
    render_figures(reference,output/"reference_figures")
    extended=load_figure_data(horizon.path,through_round=100)
    render_figures(extended,output/"horizon_figures")
    # Data manifests retain the historical interrupted attempt; review does not rewrite them.
    lines=["# OLIM 2.0 retained results","","Status: Stage IV figures and data audit generated from completed frozen batches.","",
           *( ["- [Stage IV interpretation](results_review.md)"] if (output/"results_review.md").exists() else [] ),
           "- [Study-wide figure index](figures/figure_index.md)",
           "- [Reference condition figures](reference_figures/"+reference.prefix+"__through50__figure_index.md)",
           "- [100-round reference figures](horizon_figures/"+extended.prefix+"__through100__figure_index.md)",
           "- [Data verification](retained_data_audit.json)",
           "- [Batch coverage](batch_coverage.csv)",
           "- [Scenario outcomes and secondary measures](scenario_outcome_summary.csv)",
           "- [Horizon effect summary](horizon_effect_summary.csv)","",
           f"Verified {audit['unique_trajectories']} unique trajectories. Simulation fingerprint: `{audit['simulation_fingerprint']}`.","",
           "Reproduction: `uv run --locked python -B scripts/review_study_results.py` from the opinion-model repository. Optional batch and output arguments appear in `--help`.","",
           "The audit checks source hashes, archived code, configurations, run/round coverage, initialization matching, channel/tie accounting, explicit endpoints, mirror averaging and stored paired intervals. A historical file-access error remains in the completed main manifest; all final source records passed verification.","",
           "The optional round-50 agent belief distribution was not retained and cannot be reconstructed from aggregate outcomes. No new simulations are performed by this review.",""]
    (output/"results_index.md").write_text("\n".join(lines),encoding="utf-8")
    print(output/"results_index.md",flush=True)


if __name__ == "__main__":
    main()
