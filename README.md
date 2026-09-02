# Opinion Leader Influence Model (OLIM)

OLIM is an agent-based model of how opinion leaders, content circulation,
opinion updating, and adaptive follower networks jointly shape online opinion
dynamics. This `legacy` branch preserves the implementation and experiment
artifacts developed for Feiyu Tang's 2026 undergraduate senior thesis at the
ZJU-UIUC Institute, supervised by Pavel Loskot.

This branch is retained as a reproducible thesis-era snapshot. It is not the
active general-framework branch.

## Contents

- Model configuration and execution code in the repository root.
- Behavioral modules under `modules/`.
- Baseline and opinion-leader experiments under `experiments/`.
- Analysis notebooks and archived outputs under `notebooks/` and `outputs/`.
- Sensitivity-analysis code under `Sensitivity_Analysis/`.
- The replication-oriented ODD specification in [`ODD_Protocol.pdf`](./ODD_Protocol.pdf).

## Run

Install the locked environment and run the default model:

```powershell
uv sync --locked
uv run --locked python main.py
```

Run the baseline experiment or a small leader-effects smoke test:

```powershell
uv run --locked python experiments/run_baseline.py
uv run --locked python experiments/run_leader_effects.py --profile trial --scenario core --max-runs 4
```

## Thesis

The thesis PDF is not stored in this GitHub repository. It is available on
[ResearchGate](https://www.researchgate.net/publication/405540855_Undergraduate_Senior_ThesisOpinion_Leader_Influence_in_Agent-Based_Online_Opinion_Dynamics)
and has DOI
[`10.13140/RG.2.2.17881.28001`](https://doi.org/10.13140/RG.2.2.17881.28001).

## Citation

Suggested citation for the thesis and its findings:

> Tang, F. (2026). *Opinion leader influence in agent-based online opinion
> dynamics* [Undergraduate senior thesis, ZJU-UIUC Institute]. ResearchGate.
> https://doi.org/10.13140/RG.2.2.17881.28001

## Scope

OLIM is abstract and theory-driven. It is not calibrated to a single empirical
platform; its parameters operationalize assumptions about visibility, content
circulation, expression, opinion updating, and adaptive network change.
