# Change 0001 — Separate the General Core from Case Mechanisms

> Project: Opinion Model
> Change ID: `GOM-MC-001`
> Origin: generalized model boundary
> Date: 2026-08-23
> Source framework: refinement draft v0.1
> Target framework: `GOM-FW 0.2-draft`
> Status: validating

## 1. Scientific delta

The previous plan treated Message, Selection, Update, and Network as four
general submodels. The revised framework retains only Information Formation
and Information Effect as candidate general mechanisms. Action triggers,
allocation, delivery, consumption, opinion-leader roles, platform policy, and
network adaptation become explicit case or experimental boundary mechanisms.

The change was triggered by recognizing that legacy opinion-leader and platform
assumptions had entered the general baseline without being required by its
intended inference.

Change class: ontology and causal structure, with interface and boundary
changes.

| Element | Before | After | Provenance |
|---|---|---|---|
| Message mechanism | Included posting opportunity, creation decision, stance, and style | Conditional Information Formation only | User decision |
| Selection | General submodel | Case mechanism or declared null boundary | User decision |
| Exposure | Delivery and consumption combined | Core receives only information declared consumed | User decision |
| Network update | General submodel | Optional case mechanism | User decision |
| Opinion leader | Later extension to a four-module baseline | Dedicated `opleader` case | User decision |
| Platform | Later extension of Selection | Dedicated `platform` case | User decision |

## 2. Changed contracts

| Contract | Change | Owner | Compatibility break |
|---|---|---|---|
| Agent state | Remove default dependence on leader, followers, activity, and network position | Framework | Yes for legacy |
| Information item | Retain only dimensions required by core effects; case metadata remains optional | Formation | Potential |
| Expression | Trigger and action type supplied externally | Case/null environment | Yes |
| Consumption | Case supplies actually consumed information | Case/null environment | Yes |
| State update | Core owns effect and synchronous proposed next state | Effect/framework | Clarified |
| Network state | No default endogenous writer | Case when present | Yes |

Removed feedbacks include automatic posting-selection-update-network closure in
the general baseline. A case may restore any feedback explicitly and must state
its schedule and compatibility.

## 3. Dependency impact

| Artifact | Impact | Status/action |
|---|---|---|
| `Model-Refinement/README.md` v0.1 | Direct; old causal decomposition | Superseded by `docs/roadmap.md` |
| Package `extensions` | Direct; boundary was ambiguous | Replaced by `interfaces` and `baseline` |
| `main` README | Direct; branch and scope names | Revised |
| `study-a` branch | Direct naming and scope | Rename to `opleader` |
| `study-b` branch | Direct naming and scope | Rename to `platform` |
| `legacy` implementation | Scientific incompatibility with general core | Retain unchanged as historical case |
| Thesis and empirical results | Interpretation only | Retain; do not use as general-core validation |

Known unaffected artifacts are the archived raw Zhihu data and legacy outputs;
this change neither rewrites nor deletes them.

## 4. Evidence and claim impact

- Legacy results remain simulation results for the bundled OLIM case.
- They do not validate the new core or null baseline.
- Previous claims that Selection and Network are required general mechanisms
  are superseded.
- Core Formation and Effect mechanisms remain open until competing rules and
  micro-tests are specified.

## 5. Validation plan

| Level | Check | Expected evidence | Status |
|---|---|---|---|
| V0 | Package and contract imports | Core/interfaces/baseline load without case code | Pending |
| V0 | Repository search | No active `study-a`/`study-b` branch documentation remains | Pending |
| V0 | Git branch verification | `opleader` and `platform` preserve former branch tips | Pending |
| V1+ | Formation and Effect probes | Mechanism evidence under controlled inputs | Future work |
| V2 | Null-baseline coupling | Reconstructable, synchronous core loop | Future work |

The current change can close after the documentation, package structure,
cache cleanup, tests, and branch references are verified. Scientific mechanism
validation remains a later framework task.
