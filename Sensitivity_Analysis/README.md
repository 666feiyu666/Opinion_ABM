# Leader-Mechanism Robustness Sensitivity Analysis

This directory implements a robustness-oriented sensitivity experiment for the
thesis claims. It does not attempt to rank all OLIM parameters globally.
Instead, it asks whether the reported leader effects survive plausible changes
to the mechanisms that give opinion leaders additional visibility or salience.

## Comparison Logic

Every leader run is paired with a no-leader control using the same topology,
random seed, population size, and number of rounds. Outcomes are reported both
as raw final values and as matched effects:

```text
delta outcome = leader-condition outcome - matched no-leader outcome
```

The main checks are:

- Positive leaders retain a positive effect on final mean opinion, while
  negative leaders retain a negative effect.
- Positive and negative leaders retain positive effects on extremist ratio and
  homophily relative to the no-leader control.
- Balanced leaders remain directionally closer to the control than one-sided
  leaders.
- In the share-gradient study, increasing leader share retains increasing
  one-sided directional, extremity, and homophily effects.

For the benchmark study, the exported checks separately report whether the
mean effect has the expected direction and whether its 95% confidence interval
continues to exclude zero. This prevents a weak mean pattern from being
reported as equally strong robustness evidence.
For the share-gradient study, both strict adjacent-level monotonicity and the
broader `1%`-to-`5%` endpoint increase are exported, so local saturation does
not get conflated with loss of the overall share effect.

## Studies

`core_robustness` fixes `N=1000`, leader share at `3%`, and uses `BA` and
`SBM` networks. It varies one leader-mechanism parameter at a time:

| Parameter | Mechanism | Low | Nominal | High |
| --- | --- | ---: | ---: | ---: |
| `alpha2` | Leader posting-origination advantage | 0.55 | 1.10 | 1.65 |
| `beta1_diff` | Leader follower-based diffusion amplification | 0.40 | 0.80 | 1.20 |
| `w_l` | Leader exposure weight | 1.00 | 1.50 | 2.00 |
| `lambda_L` | Leader attractiveness in network adaptation | 0.00 | 0.50 | 1.00 |

The nominal configuration is simulated once and reused as the common reference
for the four one-at-a-time comparisons.

`share_gradient` fixes `N=1000`, uses `BA` and `SBM` networks, and varies
leader share over `1%`, `3%`, and `5%`. To keep this extension focused, it
checks only the two most direct exposure-visibility mechanisms:
`beta1_diff` and `w_l`, each at low, nominal, and high levels.

## Profiles

| Profile | Purpose | Seeds | Rounds |
| --- | --- | ---: | ---: |
| `smoke` | Reduced execution check, not thesis evidence | 1 | 2 |
| `trial` | Full study structure for preliminary review | 2 | 25 |
| `main` | Thesis-ready robustness analysis | 10 | 50 |

The formal `main` design contains 560 runs for `core_robustness` and 920 runs
for `share_gradient`, including matched controls.

## Running

```bash
python Sensitivity_Analysis/run_sensitivity.py --profile smoke --study core_robustness
python Sensitivity_Analysis/run_sensitivity.py --profile trial --study core_robustness
python Sensitivity_Analysis/run_sensitivity.py --profile main --study core_robustness
python Sensitivity_Analysis/run_sensitivity.py --profile main --study share_gradient
python Sensitivity_Analysis/run_sensitivity.py --profile main --study all --workers 8
python Sensitivity_Analysis/run_sensitivity.py --profile main --study all --workers 8 --resume
```

Outputs are written below `outputs/sensitivity_analysis/` and include:

- `experiment_grid.csv`
- `raw_results.csv`
- `matched_effects.csv`
- `summary_effects.csv`
- `control_summary.csv`
- `robustness_checks.csv`
- `manifest.json`
- `raw_results.partial.csv` as a resume checkpoint during long executions
- `figures/core_matched_effects.png`, a forest plot with 95% confidence intervals
- `figures/share_gradient_effects.png`, an annotated heatmap showing share-gradient magnitude and plateau patterns

For batch efficiency, sensitivity runs disable network layout calculation and
do not retain posts and exposure sets after round-level summaries are formed.
After the combined formal run, open
`notebooks/Sensitivity_Analysis_Results.ipynb` to explore the matched effects,
robustness checks, thesis-facing tables, and figures interactively.
