# Mass-Communication Opinion-Leader Model: Rough Design

> **Status:** Implemented exploratory connected framework
>
> **Case:** `opleader`
>
> **Scope:** Two-step-flow-inspired opinion leadership after information about a focal issue has entered the social system, using a fixed directed information-access network

## Purpose and model boundary

This model examines how opinion leaders shape information diffusion and opinion updating after a topic has entered a social system. It adapts selected ideas from Katz's two-step-flow research to the interpersonal communication process that follows.

The boundary assumption is that information about the focal issue has already circulated before the simulation begins. Its prior influence is represented through the agents' initialized beliefs and other initial conditions. The simulated process begins when social agents may originate messages within their network.

Within this boundary, an **originator** is not necessarily the original creator of the underlying information. It is an agent selected to produce a message that can initiate a diffusion episode in the modeled network.

## Theoretical interpretation

For one focal topic, opinion leadership is represented through the following provisional mapping:

| Katz-related property | Candidate representation |
|---|---|
| Greater involvement in the topic | Leaders have a higher probability of originating a message. |
| Greater contact with prior information | If retained, leaders may begin with different belief or information initial conditions. |
| Strategic social location | Leaders may occupy more connected positions in the fixed interpersonal network. |
| Competence or recognized authority | Messages from leaders may receive greater evidence weight. |
| Domain-specific leadership | Leader status applies only to the focal topic. |

These representations adapt selected ideas from two-step-flow theory to the current framework. Because ordinary agents may also originate posts, the model represents leader-amplified interpersonal diffusion within the post-entry stage rather than a literal two-step sequence.

## Entities, states, and communication roles

- **Adaptive social agents:** all agents hold a private Beta belief, may originate a message, may receive messages, and update their beliefs.
- **Opinion leaders:** adaptive social agents whose leader status can modify particular mechanisms, including origination probability, initialized structural reach, and message weight.
- **Ordinary agents:** adaptive social agents governed by the same basic process but without the corresponding leader advantages.
- **Messages:** observable expressions produced by agents. Each message retains its producer identity and stance so that exposure and influence can depend on source role.
- **Social network:** fixed directed ties recording which producers are available to each consumer for communication about the focal topic.

`leader` and `ordinary` are agent roles. `originator` and `consumer` describe communication roles within a round:

- an **originator** has been selected to produce one message;
- a **consumer** is an agent exposed to a produced message.

The same agent can occupy different communication roles within and across rounds.

## Message origination

At the beginning of each round, every adaptive agent is eligible to originate a message. The origination rule is shared across roles, but opinion leaders receive a higher origination probability through a multiplicative advantage on the odds scale. A common decay term represents declining interest after the external-information stage. A successful origination draw produces exactly one message; there is no second posting or production-probability gate. Activity heterogeneity, network position, and exposure-dependent extensions are deferred.

The message stance is sampled from the originator's start-of-round Beta belief. The probability of a supportive message is the Beta probability mass above the neutral threshold of one half. This mapping uses both the belief's mean and its concentration without adding a second behavioral decision.

Keeping the origination draw separate from conditional stance formation prevents a leader's greater likelihood of speaking from being conflated with opinion direction. For the bounded mechanism definition, see [Posting Origination](posting-origination.md).

## Diffusion, exposure, and influence

Produced messages diffuse through the fixed directed social network. Every message originated in the current round is delivered deterministically to consumers for whom that originator is an eligible producer. Messages from both leaders and ordinary agents obey the same tie-bound rule; no message is delivered to an untied agent. Selection is stance-blind and role-blind.

Opinion leaders can have greater expected reach through a more connected initialized network position and through their higher probability of originating messages. Selection adds no separate leader-conditioned delivery advantage. Attention competition is omitted, so configured capacity must be large enough to retain all messages eligible through social ties. For the bounded mechanism definition, see [Message Selection](message-selection.md).

This case does not include algorithmic ranking, recommendation, or out-of-network platform amplification. Those are platform mechanisms rather than opinion-leader mechanisms.

Unequal influence has two separable components:

1. **Exposure or reach:** whether and how widely a message is received.
2. **Evidence weight:** how strongly a received message affects the recipient's belief update.

Messages retain producer identity so that leader and ordinary messages can receive different weights. Ordinary messages provide the reference weight, and opinion-leader messages receive a relative evidence multiplier. Recipient role does not alter this multiplier. The substantive interpretation of a larger leader weight—such as credibility, persuasion, or evidential strength—and its numerical magnitude remain open.

## Opinion updating

Each recipient aggregates the messages received during the round and updates its private Beta belief once. Raw exposure counts and weighted evidence remain distinguishable so that message availability is not conflated with influence. For the bounded mechanism definition, see [Message Aggregation](message-aggregation.md).

## Round schedule

Each round follows one synchronous sequence:

1. Read agent beliefs and the fixed social network from the state at the start of round `t`.
2. Draw whether each agent originates a message, with a higher origination probability for opinion leaders. Each successful draw produces one message whose stance is sampled from the agent's start-of-round Beta belief.
3. Deliver each originated message to all of the originator's regular social contacts and determine each agent's exposures.
4. Aggregate received messages, including any source-dependent evidence weights.
5. Propose one updated Beta belief for each agent.
6. Commit the proposed beliefs as the state for round `t+1` while retaining the same social network.

An updated belief can affect message origination only from the next round onward.

## Current boundaries and open decisions

The current case includes ordinary-agent and leader message origination, deterministic tie-bound diffusion through a fixed directed network, source-dependent evidence aggregation, and opinion updating. It excludes a separate message-production decision, stochastic or out-of-tie delivery, attention competition, message forwarding, network adaptation, algorithmic ranking, out-of-network recommendation, and engagement-driven platform feedback.

The exploratory implementation uses the same directed Barabasi-Albert realization, ordinary-belief draws, leader selection, and orientation assignment as `main@1d8eaac`. Leaders are the top three percent of agents by producer in-degree. The current basic decisions are the decaying logistic origination probability, the leader log-odds advantage, the Beta-tail stance mapping documented in [Posting Origination](posting-origination.md), the deterministic tied-delivery rule documented in [Message Selection](message-selection.md), and the source-only relative weighting rule documented in [Message Aggregation](message-aggregation.md). Parameter values remain transparent working assumptions rather than calibrated estimates.

One comparison limitation remains explicit: interest decay applies to both roles in this scenario but is absent from `null@c142c46`. Until decay is made shared across scenarios or disabled throughout the matched design, an `opleader - null` contrast includes that common temporal decline as well as leader-specific advantages.

Leader initialization, origination advantage, structural reach, and evidence weight should remain separable so that their individual and combined effects can later be examined.
