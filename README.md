# Opinion Model

A shared agent-based framework for studying how agent state shapes information
and how consumed information changes agent state.

## Scope

`main` contains the general contracts and null baseline. Platform behavior,
opinion leadership, action, delivery, consumption, and network adaptation belong
to explicit cases.

- `opleader`: opinion-leader case.
- `platform`: platform-distribution case.
- `legacy`: preserves the implementation and experiment artifacts developed for Feiyu Tang's 2026 undergraduate senior thesis at the ZJU-UIUC Institute, supervised by Prof. Pavel Loskot

The scientific specification will be later provided as an ODD document.

## Current Goal
Establish a minimal shared baseline
The baseline should contain:
- a precisely defined latent attitude;
- a precisely defined observable message;
- a neutral source/message selection process;
- one justified attitude-update mechanism;
- an explicit multi-message aggregation rule;
- a fixed activation and update schedule;
- no opinion-leader advantage;
- no algorithmic platform preference.
This baseline becomes the control model used by both scenario families.

## Tips on Buliding
For every function, record:
- inputs and outputs;
- substantive meaning;
- boundary behavior;
- randomness;
- qualitative expectations;
- empirical or theoretical basis;
- plausible alternative formulation.
For example, before choosing an update equation, decide what should happen under no exposure, unanimous exposure, balanced conflicting exposure, moderate disagreement, and extreme disagreement.

## Quick check

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```
