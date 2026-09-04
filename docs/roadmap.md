# General Model Development Roadmap

> Status: draft v0.2
> Date: 2026-08-23
> Compatible framework: `GOM-FW 0.2-draft`
> Legacy reference: OLIM on `legacy`

## Purpose

Develop the smallest interpretable framework for how agent state forms
information and how consumed information changes agent state. Platform-specific
actions, allocation, consumption, special roles, and network adaptation are
case mechanisms rather than default components of the general model.

## Development sequence

### Phase C — Shared contracts

1. Define the minimal Agent-state snapshot.
2. Define the minimal Information item.
3. Define expression-context and consumed-information interfaces.
4. Define synchronous state proposal and commit semantics.
5. Define event and state observations needed to reconstruct a round.

Exit requires canonical definitions, owners, domains, provenance, and V0
contract tests. It does not require a full simulation.

### Phase F — Information Formation

Given a controlled state and an externally supplied expression context, test
how information is formed. Action opportunity and platform action type remain
fixed inputs.

Required evidence includes domain bounds, sign behavior, deterministic and
stochastic reproducibility, controlled response to each retained state, and
explicit behavior for neutral and boundary cases.

### Phase E — Information Effect

Given a controlled state and an externally supplied consumed-information
batch, test the proposed state transition. Delivery and consumption choice
remain fixed inputs.

Required evidence includes no-consumption invariance, state bounds,
dose-response behavior, repeated-consumption trajectories, competing
mechanisms, and explicit behavior near neutral and state boundaries.

### Phase B — Null baseline

Couple Formation and Effect through a declared null boundary. Begin with fixed
events; use deterministic pairing or uniform random matching only when the
boundary is recorded and the additional stochasticity answers a test question.

The null baseline must support event reconciliation, synchronous updates,
independent random streams, reproducibility, module-off checks, and transparent
micro-to-macro aggregation. It is not a platform model.

### Case development

- `opleader`: add and isolate opinion-leader roles, visibility, credibility,
  action, susceptibility, and network channels as explicit treatments.
- `platform`: add and isolate eligibility, ranking, delivery, diversity,
  capacity, recency, and consumption policies as explicit treatments.

Case mechanisms may not silently redefine shared state or information
semantics. Framework-relevant discoveries return to `main` through a change
record.

## General modeling principles

- mechanism before scenario;
- minimal state before convenient state;
- controlled inputs before endogenous event generation;
- micro behavior before macro outcomes;
- deterministic transformations separated from stochastic realization;
- independent reproducible random streams;
- synchronous state transitions;
- evidence gates rather than preferred-trajectory gates;
- lower-level evidence does not establish a higher-level claim.

## Current non-goals

- reproduce every legacy OLIM trajectory;
- select a preferred social-media action taxonomy;
- define a universal platform selection mechanism;
- evolve a social network in the general baseline;
- calibrate the full framework to one platform;
- make causal claims about opinion leaders or recommender systems;
- optimize performance before contracts stabilize.

## Immediate deliverable

Complete Phase C and pre-register the first Formation and Effect micro-cases.
No opinion-leader, platform allocation, or network-adaptation implementation is
required.
