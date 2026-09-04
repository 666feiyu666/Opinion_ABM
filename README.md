# Opinion Model

## Null Branch
`null` contains the general contracts and null baseline. Platform behavior, opinion leadership, action, delivery, consumption, and network adaptation belong to explicit cases.

## Goal of Null Branch
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

## Quick check

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```
