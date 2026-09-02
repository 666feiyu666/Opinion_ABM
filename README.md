# Opinion Model

A shared agent-based framework for studying how agent state shapes information
and how consumed information changes agent state.

## Scope

`main` contains the general contracts and null baseline. Platform behavior,
opinion leadership, action, delivery, consumption, and network adaptation belong
to explicit cases.

- `opleader`: opinion-leader case.
- `platform`: platform-distribution case.
- `legacy`: frozen OLIM implementation and historical results.

The scientific specification will be provided as an ODD document rather than maintained under `docs/`.

## Quick check

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```
