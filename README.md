# Opinion Model

A shared framework for agent-based research on how agent state shapes
information and how consumed information changes agent state.

## Scientific boundary

The `main` branch owns the general framework, shared contracts, and a
transparent null baseline. It does not decide which platform-specific action a
user takes, how a platform allocates attention, who counts as an opinion
leader, or how platform-specific social connections change.

The general causal loop is:

```text
agent state
    -> information formation, conditional on an external expression event
    -> information item
    -> externally supplied consumption event
    -> information effect
    -> next agent state
```

Concrete action, delivery, consumption, role, and network mechanisms belong to
explicit case branches. The null baseline supplies declared experimental
boundary conditions; it is not a theory of platform behavior.

## Branches

- `main`: general framework, core contracts, and null baseline.
- `opleader`: opinion-leader case development.
- `platform`: platform-mediated information-allocation case development.
- `legacy`: frozen OLIM implementation and historical results.

## Layout

- `docs/framework.md`: canonical scientific scope and framework contract.
- `docs/model-map.md`: ownership, interfaces, and branch dependency map.
- `docs/roadmap.md`: evidence-gated development sequence.
- `docs/changes/`: scientific model change-impact records.
- `src/opinion_model/core/`: information-formation and information-effect
  contracts.
- `src/opinion_model/interfaces/`: boundary between the core and case-specific
  event providers.
- `src/opinion_model/baseline/`: transparent null boundary conditions and
  framework checks.

Case branches extend these contracts without redefining their canonical
meaning. A discovery that changes shared state, information semantics, or the
update schedule must first be proposed as a framework change on `main`.
