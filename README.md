# Opinion Model

From Two-Step Flow to Algorithmically Mediated Influence: How Online Platforms Transform Opinion Leadership in Opinion Dynamics

## Scope

This repository provides a shared agent-based modeling framework for comparing three information environments:

- `main`: shared model contracts, common mechanisms, and the stable integration baseline from which the case branches are developed.
- `null`: control case with neither opinion-leader advantage nor algorithmic platform preference.
- `opleader`: opinion-leader case based on interpersonal influence and two-step-flow theory.
- `platform`: platform-distribution case with algorithmically mediated information exposure.
- `legacy`: preserves the implementation and experimental artifacts developed for Feiyu Tang's 2026 undergraduate senior thesis at the ZJU-UIUC Institute, supervised by Prof. Pavel Loskot.

The scientific specification will later be provided as an ODD document.

## Tips on Building

For every model function, record:

- inputs and outputs;
- substantive meaning;
- boundary behavior;
- randomness;
- qualitative expectations;
- empirical or theoretical basis;
- plausible alternative formulations.

...

## Quick Check

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v