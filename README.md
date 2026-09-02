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

## Quick check

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```
