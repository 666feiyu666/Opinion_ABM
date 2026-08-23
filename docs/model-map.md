# Model and Branch Map

> Compatible framework: `GOM-FW 0.2-draft`
> Status: current

## Scientific layers

```text
CASE ENVIRONMENT
  owns action triggers, allocation, delivery, consumption,
  special roles, platform policy, and optional network adaptation
                         |
                         v
SHARED INTERFACES
  expression context + consumed-information batch
                         |
                         v
GENERAL CORE
  Information Formation + Information Effect + synchronous state commit
                         |
                         v
OBSERVATION
  event records, agent transitions, and aggregate diagnostics
```

The null baseline occupies the case-environment position only for testing. It
does not become part of the substantive causal core.

## Ownership map

| Concern | `main` | `opleader` | `platform` | `legacy` |
|---|---|---|---|---|
| Shared ontology and state contracts | Owns | Reads | Reads | Historical/incompatible until mapped |
| Information Formation semantics | Owns | Uses or proposes explicit extension | Uses | Historical implementation |
| Information Effect semantics | Owns | Uses or proposes explicit extension | Uses | Historical implementation |
| Expression action and opportunity | Interface only | May implement leader action channels | May implement platform affordances | Implements OLIM posting logic |
| Delivery and consumption | Interface/null fixture only | May implement leader visibility | Owns platform allocation policies | Implements OLIM diffusion |
| Opinion-leader identity and advantage | Excluded | Owns | Excluded unless interaction is explicit | Bundled historical treatment |
| Network adaptation | External/optional interface | Case-specific if required | Case-specific if required | Implements OLIM follow updates |
| Platform ranking/policy | Excluded | Excluded unless declared | Owns | Contains historical exposure rules |

## Branch workflow

1. Shared ontology, contracts, clocks, and observation changes originate on
   `main`.
2. `opleader` and `platform` record their compatible framework revision.
3. A case may add state or metadata that remains case-owned.
4. If the core must read or write that addition, open a framework change on
   `main` before merging the new dependency.
5. Case results remain conditional on their action, allocation, consumption,
   and feedback boundaries.

## Compatibility rule

The `legacy` branch is preserved as the frozen OLIM case and provenance record.
Its outputs do not serve as regression targets for the general baseline unless
a later compatibility study explicitly reconstructs the relevant boundary and
mechanisms.
