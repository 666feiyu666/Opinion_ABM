# Platform-Mediated Opinion Dynamics: Rough Design

> **Status:** Implemented connected platform-only scenario
>
> **Case:** `platform`
>
> **Scope:** Uniform out-of-network message availability, finite attention, and
> exposure-driven adaptation of a directed following network without opinion
> leaders

## Purpose and model boundary

This model examines how platform-mediated access to messages outside existing
social ties changes opinion dynamics and the following network. It represents a
minimal platform environment in which messages can escape the current network,
compete for finite attention, affect private Beta beliefs, and create
opportunities for agents to form or dissolve following ties.

The `platform` case contains no opinion leaders. All agents use ordinary message
origination, and all processed messages contribute homogeneous evidence during
aggregation. The case-specific mechanisms are platform message selection and
exposure-driven network adaptation.

The platform is not represented as an autonomous agent and does not optimize an
objective. Its initial role is limited to supplying uniform out-of-network
message availability. The model does not yet claim to reproduce a realistic
recommendation algorithm.

As in the shared model, information about the focal issue is assumed to have
entered the social system before the simulation begins. Its prior influence is
represented by the initialized Beta beliefs. The simulated process begins when
agents may originate messages from those beliefs.

## Mechanism ownership

| Process | Current owner | Role in the `platform` case |
|---|---|---|
| Message origination | Shared mechanisms | Ordinary agents may originate one message whose stance is sampled from their Beta belief. |
| Message selection | `platform` | Existing-tie messages become available deterministically; out-of-network messages may become available through a Bernoulli draw; finite capacity determines processed exposures. |
| Message aggregation and opinion update | Shared mechanisms | Every processed message contributes the same base evidence weight to the recipient's Beta update. |
| Network update | `platform` | Processed messages create opportunities to form or dissolve directed following ties according to degree and belief--message alignment. |

Platform mechanisms do not silently change shared message semantics or the
meaning of the Beta belief. The full comparison loader now enforces matched
shared parameters and identical platform parameters in `platform` and
`baseline`; their numerical values remain exploratory rather than calibrated.

The executable platform-only scenario is assembled in
`src/opinion_model/scenarios/platform/` and configured by
`configs/platform.toml`. Its numerical values are exploratory settings matched
to the integrated baseline; they are not calibrated estimates. The scenario
contains no opinion-leader roles or mechanisms.

## Entities, states, and interaction roles

- **Adaptive social agents:** all agents hold a private Beta belief, may
  originate messages, process selected messages, update their beliefs, and
  adapt whom they follow.
- **Messages:** transient observable expressions containing an originator
  identity and a binary supportive or opposing stance.
- **Directed following network:** a tie $i\rightarrow j$ means that consumer
  $i$ follows producer $j$. The network used for selection is fixed within a
  round but may change between rounds.
- **Candidate messages:** messages made available to an agent through either an
  existing tie or the platform's out-of-network channel.
- **Exposures:** candidate messages retained after the agent's processing
  capacity is applied. Only exposures affect beliefs or create network-update
  opportunities.
- **Platform allocation rule:** a case mechanism that supplies uniform
  out-of-network availability; it is not an agent with its own state or goal.

`originator` and `consumer` are temporary communication roles. The same agent
may originate a message and process messages from other agents in one round.
Existing-tie and out-of-network candidates remain distinguishable by their
relationship to the start-of-round network, even though they compete equally
for attention.

## Shared message origination

Every ordinary agent receives one opportunity to originate a message in each
round. A successful origination draw produces exactly one message. Conditional
on origination, the probability of a supportive stance is the probability mass
that the agent's start-of-round Beta belief assigns above the neutral threshold
of one half.

The platform does not change origination probability or the mapping from belief
to message stance. There is no platform-conditioned posting incentive,
engagement feedback, or source-role advantage in the current case.

## Platform message selection

Messages from currently followed producers become candidates deterministically.
For each message from an untied producer, the platform independently draws
whether it becomes available to the consumer using one out-of-network
availability probability $\rho$.

Tied and out-of-network candidates are then combined in one attention pool. An
agent processes at most $K=10$ messages per round. If more than ten candidates
are available, ten are sampled uniformly without replacement. Neither source
channel receives priority after entering the common pool.

The mechanism therefore represents uniform platform-mediated reach rather than
personalized or engagement-optimized ranking. Its detailed definition is
maintained in [Platform Message Selection](platform-message-selection.md).

## Homogeneous aggregation and Beta-belief updating

Each recipient aggregates all retained exposures once per round. Supportive and
opposing messages contribute the same common evidence weight regardless of
their originator or whether they arrived through a tie or the out-of-network
channel. The resulting evidence updates the two shape parameters of the
recipient's Beta belief.

Out-of-network messages can consequently change opinions without receiving a
special cognitive weight. Their distinct effect comes from changing which
messages enter the finite exposure set.

## Exposure-driven network adaptation

Every unique producer in an agent's retained exposure set may create one
network-update decision. A producer who was not followed in the start-of-round
network is a candidate for tie formation. A producer who was followed is a
candidate for tie dissolution.

The formation and dissolution probabilities depend on two factors:

1. the consumer's start-of-round number of following ties; and
2. alignment between the consumer's Beta belief and the observed message
   stance.

Lower-degree agents are more likely to form ties and less likely to dissolve
them. Greater alignment increases formation probability and decreases
dissolution probability. Alignment is calculated from the Beta probability mass
on the supportive side and therefore uses both the belief mean and
concentration. Formation and dissolution use logistic probabilities with degree
normalized by the maximum possible following degree. Their intercepts describe
the probability at half degree and neutral alignment, while positive degree and
alignment coefficients act on the log-odds scale.

Isolation remains possible rather than being prohibited. Low degree creates a
soft tendency toward connection, and platform-mediated out-of-network exposures
give isolated agents opportunities to reconnect.

The model imposes no one-follow or one-unfollow limit. Producer-level decisions
are evaluated from the same start-of-round degree and network, then committed
together. Detailed candidate sets, alignment, probability constraints, and
boundary behavior are maintained in
[Platform Network Update](platform-network-update.md).

## Round schedule

Each round follows one synchronous sequence:

1. Read all Beta beliefs and the directed following network $G^t$ from the
   start-of-round snapshot.
2. Draw whether each ordinary agent originates a message and, conditional on
   origination, sample its stance from the agent's Beta belief.
3. For every potential consumer, make existing-tie messages available and draw
   out-of-network candidate availability with probability $\rho$.
4. Combine both candidate sources and retain at most ten messages through the
   platform's uniform finite-attention rule.
5. Aggregate each consumer's retained exposures and propose its next Beta
   belief.
6. Use the same retained exposures, start-of-round belief, following degree,
   and $G^t$ to propose tie additions and removals.
7. Commit all proposed beliefs and the proposed network together as the state
   for round $t+1$.

An updated belief and a changed tie can affect message origination, selection,
or network adaptation only from the next round onward. No agent-processing order
can change another agent's inputs within the current round.

## Connected feedbacks

The start-of-round network determines which messages have deterministic access
to each consumer. The platform adds probabilistic access outside that network,
and finite attention determines actual exposures. Those exposures affect the
next Beta beliefs and create opportunities to adapt ties. The updated beliefs
and network then change the communication environment of later rounds.

The minimal connected feedback is therefore:

```text
Beta beliefs_t + network_t
  -> ordinary message origination
  -> tied and out-of-network candidate availability
  -> capacity-limited exposures
  -> Beta-belief proposals + network proposals
  -> Beta beliefs_t+1 + network_t+1
```

Message selection and network update remain distinct mechanisms within this
loop. Selection determines what is processed now; network update changes future
access.

## Current boundaries and open decisions

The current platform case includes ordinary message origination, deterministic
existing-tie candidate availability, uniform Bernoulli out-of-network
availability, a common processing capacity of ten, homogeneous evidence
aggregation, Beta-belief updating, and exposure-driven formation and dissolution
of directed following ties.

The current working decisions are the single out-of-network parameter $\rho$,
equal attention competition between source channels, $K=10$, Beta-tail
alignment, soft degree regulation, producer-level network decisions, possible
isolation, and synchronous next-round commitment.

Calibration or empirical justification of $\rho$, the formation and dissolution
parameters, and the initialization and scale of the directed network remain
open. The executable exploratory scenario chooses transparent working values in
`configs/platform.toml` solely to match the integrated baseline. Degree
normalization removes a mechanical dependence on population size, but whether
substantive parameter values should differ across population sizes remains an
empirical question. These questions are not filled by the historical OLIM
implementation automatically.

The current case excludes opinion leaders, source-dependent evidence weight,
personalization, stance or similarity ranking, popularity and engagement
signals, recency weighting, message forwarding, same-round cascades,
platform-conditioned origination, guaranteed connectivity, forced one-for-one
rewiring, triadic closure, reciprocity preference, and network adaptation from
messages the agent did not process.
