# General Opinion Model Framework

> Framework ID: `GOM-FW`
> Framework version: `0.2-draft`
> Date: 2026-08-23
> Status: draft
> Compatibility status: current

## 1. Research problem

The generalized model asks whether explicit, bounded, and testable rules for
information formation and information effect produce interpretable individual
state transitions and an interpretable aggregate baseline.

The intended inference is limited to these behavioral transformations:

1. conditional on an externally supplied expression event, how does an agent's
   current state shape the information it forms; and
2. conditional on an externally supplied consumption event, how does consumed
   information transform the consumer's state.

The unit of explanation is the agent-level state transition. Population-level
opinion distributions are emergent diagnostics, not substitutes for
mechanism-level evidence.

### Scope

- shared ontology and state-transition contracts;
- information formation conditional on expression;
- information effect conditional on consumption;
- synchronous round closure and observation;
- transparent null boundary conditions used to probe the coupled core.

### Non-goals

The generalized core does not decide:

- whether a user posts, comments, shares, likes, follows, or unfollows;
- which information is eligible, ranked, delivered, noticed, or consumed;
- how an opinion leader is identified or advantaged;
- how a platform or network policy allocates attention;
- how platform-specific social connections adapt;
- whether a legacy OLIM trajectory must be reproduced.

These mechanisms belong to explicit case environments. The `opleader` and
`platform` branches are the first two such cases.

### Why retain an ABM framework

Agent-level representation remains useful because heterogeneous state,
repeated consumption, stochastic response, and path dependence can generate
population distributions that are not recoverable from a single mean. Simple
analytical micro-cases and mean-field calculations remain required benchmark
models whenever they can answer the same question.

## 2. Evidence status

The legacy OLIM implementation, thesis experiments, and Zhihu analysis motivate
the broader research program. They do not validate this generalized core.
Their mechanisms and results are retained as case evidence and historical
benchmarks until compatibility with this framework is demonstrated.

The first framework version is assumption-led. Every substantive formation or
effect rule must identify its provenance as literature-derived, observed,
assumed, calibrated, implementation-derived, or simulation result.

## 3. Canonical ontology

| Concept | Canonical definition | Owner | Provenance/status |
|---|---|---|---|
| Agent | Persistent individual whose internal state can condition information and be changed by consumed information. | Framework | Assumed; current |
| Agent state | The minimal snapshot read by formation and effect mechanisms. Exact fields and scales remain open until required by a tested claim. | Framework | Open |
| Expression event | An external declaration that an agent is to form information in a stated context. It does not encode why the action occurred. | Case environment | User decision; current |
| Information item | The abstract content produced from state and expression context. Platform-specific metadata is optional case data. | Information Formation | Draft |
| Consumption event | A declaration that an agent actually consumed a stated information batch. Availability, ranking, delivery, and consumption choice occur upstream. | Case environment | User decision; current |
| Proposed next state | The result of applying consumed information to the current snapshot before synchronous commit. | Information Effect | Draft |
| Null boundary | A transparent experimental provider of expression and consumption events with no claim to represent a real platform. | Baseline harness | User decision; current |

## 4. State and process ownership

| State or object | Owner | Readers | Writers | Update phase |
|---|---|---|---|---|
| Agent identity | Framework | All components | Initialization only | Initialization |
| Agent state at `t` | Framework | Formation, Effect, observation | None within the round | Round start |
| Expression context | Case or null boundary | Formation | Case or null boundary | Before formation |
| Information item | Formation | Case environment, Effect, observation | Formation | Formation |
| Consumed-information batch | Case or null boundary | Effect, observation | Case or null boundary | Before effect |
| Proposed state at `t+1` | Effect | Round closure, observation | Effect | Effect |
| Committed state at `t+1` | Framework | Next round | Round closure | Synchronous closure |

Opinion-leader status, follower count, centrality, recommendation score,
platform action history, and delivery channel are not required shared states.
A case may introduce them without changing the core unless a shared contract
must read or write them.

## 5. Integrated process schedule

| Order | Owner | Process | Reads | Writes | Boundary status |
|---:|---|---|---|---|---|
| 1 | Case/null environment | Supply expression events | `state_t`, case context | expression contexts | External boundary |
| 2 | Information Formation | Form information | `state_t`, expression context | information items | Core mechanism |
| 3 | Case/null environment | Supply consumption events | information items, case context | consumed batches | External boundary |
| 4 | Information Effect | Compute state effects | `state_t`, consumed batches | proposed `state_t1` | Core mechanism |
| 5 | Framework | Commit and observe | proposed `state_t1`, event log | committed `state_t1`, diagnostics | Core schedule |

Every process reads a common `state_t`. Effects are committed synchronously so
agent iteration order cannot create an undocumented advantage.

## 6. Component register

| ID | Component | Role | Owns | Direct dependencies | Status |
|---|---|---|---|---|---|
| `IF-01` | Information Formation | Map state and expression context to information. | Formation rule and information semantics | Shared state contract | Specifying |
| `IE-01` | Information Effect | Map state and consumed information to proposed next state. | Effect rule and state transition | Shared state and information contracts | Specifying |
| `NB-01` | Null Boundary | Provide transparent events for isolated and coupled checks. | Experimental boundary conditions | Framework contracts | Planned; not a scientific mechanism |
| `CASE-OL` | Opleader case | Test explicit opinion-leader channels. | Leader roles and case mechanisms | Compatible framework version | Branch `opleader` |
| `CASE-PL` | Platform case | Test explicit allocation and consumption policies. | Platform mechanisms | Compatible framework version | Branch `platform` |

## 7. Dependency and feedback map

The minimal closed loop is:

```text
state_t
  -> Information Formation
  -> information items
  -> external consumption boundary
  -> Information Effect
  -> state_t1
```

The environment controls when formation and consumption are invoked. The core
controls the semantics of information and state transition. A case may create
feedback from prior outcomes to later action or allocation, but must declare
that feedback and its activation time rather than embedding it in the core.

## 8. Baseline interpretation

The null baseline is a diagnostic boundary condition. It may use fixed events,
deterministic pairing, or explicitly uniform random matching. Its results are
always conditional on that declared boundary and are not claims about a real
platform.

A baseline is interpretable when:

- no consumed information produces no unexplained state change;
- state and information values remain within declared domains;
- symmetric inputs produce symmetric results when the mechanism is specified
  as symmetric;
- repeated information produces a documented trajectory, including any
  saturation, convergence, oscillation, or boundary behavior;
- aggregate diagnostics can be reconstructed from agent transitions and event
  records;
- node identity, insertion order, and unrelated random draws do not change the
  intended result.

## 9. Validation and claim limits

- V0 checks establish equations, domains, transformations, and deterministic
  invariants.
- V1 checks establish one mechanism only under declared synthetic inputs.
- V2 checks establish compatibility between Formation and Effect through a
  null boundary.
- V3/V4 evidence is required before making meso- or system-level claims.

No successful baseline run validates an opinion-leader or platform claim.

## 10. Open decisions

- Which fields are necessary in the minimal Agent state?
- Is confidence a required state or an optional mechanism?
- Is expression always externally triggered, or is a generic express/abstain
  decision eventually needed in the core?
- What minimal information dimensions are required beyond expressed position?
- What competing Information Effect mechanisms should be tested first?
- Which null boundary is most diagnostic for the first coupled baseline?
- Which observation metrics are required before case development begins?

## 11. Smallest useful next iteration

Specify one minimal Agent-state contract and one Information-item contract,
then run controlled micro-cases for Information Formation and Information
Effect separately. Do not implement opinion-leader, platform-allocation, or
network-adaptation mechanisms in this iteration.
