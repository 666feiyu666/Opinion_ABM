# Full Model Comparison

> **Status:** Implemented exploratory comparison framework
>
> **Scope:** Matched execution of `null`, `opleader`, `platform`, and the
> integrated `baseline`

## Comparison structure

The full model is represented as four explicit scenario assemblies rather than
one scheduler containing scenario flags:

| Scenario | Opinion-leader mechanisms | Platform mechanisms |
|---|---:|---:|
| `null` | No | No |
| `opleader` | Yes | No |
| `platform` | No | Yes |
| `baseline` | Yes | Yes |

Every scenario runs through the same synchronous scheduler and shared state and
event contracts. `baseline` is the integrated reference scenario on `main`; it
is not the shared package or a separate branch.

The comparison loader reads the four scenario configuration files named by
`configs/comparison.toml`. It rejects a run before simulation when the seeds,
shared simulation parameters, ordinary-agent initialization, leader settings,
platform settings, orientations, or extremism threshold have drifted.

## Matched and scenario-specific processes

The following constructs are held constant across all four scenarios:

- population size, rounds, seeds, and random-stream meanings;
- initial directed Barabasi-Albert network realization;
- ordinary-agent belief draws and concentration;
- round-one origination probability and common interest decay;
- evidence weight, self-exposure exclusion, synchronous activation, and
  observation threshold.

The no-platform scenarios use nonbinding capacity because attention competition
is outside their model boundary. The two platform-enabled scenarios use the
same finite capacity and the same platform parameters. The two leader-enabled
scenarios use the same leader selection, orientation assignments, origination
advantage, and source-evidence multiplier.

Consequently, the `opleader - null` contrast represents the joint
opinion-leader package, while `platform - null` represents the joint platform
package. The latter combines out-of-network availability, finite attention, and
exposure-driven network adaptation; it does not separately identify those
three platform mechanisms.

## Execution and retained evidence

Run the complete comparison from the repository root:

```powershell
uv run --locked python scripts/run_comparison.py
```

The runner writes ignored local outputs under `outputs/comparison/`:

- one aligned round-metric table containing scenario and mechanism-factor
  columns;
- common final metrics and across-seed summaries;
- a separate round-metric table for each scenario;
- a manifest containing the current Git revision, dirty state, exact config
  paths, resolved scenario configurations, and interpretation boundaries.

The individual scenario runners remain available for detailed transition and
decision diagnostics. Passing the combined software tests establishes
implementation consistency and reproducibility; it does not by itself provide
calibration, empirical validation, or support for a substantive result claim.
