# Mass-Communication Opinion-Leader Model: Rough Design

> **Status:** Revised rough design
>
> **Case:** `opleader`
>
> **Scope:** Two-step-flow-inspired opinion leadership after external information has entered the social system, without a modeled press or online platform

## Purpose and model boundary

This model examines how opinion leaders shape information diffusion, opinion updating, and network adaptation after a topic has entered a social system. It is inspired by Katz's two-step-flow research, but it does not model the press as an entity or reproduce the complete press-to-leader-to-public sequence.

The boundary assumption is that external information providers have already done their work before the simulation begins. Their prior influence is represented only through the agents' initialized beliefs and other initial conditions. The simulated process begins when social agents may originate messages within their network.

Within this boundary, an **originator** is not necessarily the original creator of the underlying information. It is an agent selected to produce a message that can initiate a diffusion episode in the modeled network.

## Theoretical interpretation

For one focal topic, opinion leadership is represented through the following provisional mapping:

| Katz-related property | Candidate representation |
|---|---|
| Greater involvement in the topic | Leaders have a higher probability of originating a message. |
| Greater contact with external information | If retained, leaders may begin with different belief or information initial conditions; the external source itself remains outside the model. |
| Strategic social location | Leaders may occupy more connected positions or acquire greater reach through the adaptive network. |
| Competence or recognized authority | Messages from leaders may receive greater evidence weight. |
| Domain-specific leadership | Leader status applies only to the focal topic. |

These representations adapt selected ideas from two-step-flow theory to the current framework. Because ordinary agents may also originate posts and the external-information stage is not simulated, the model represents leader-amplified social diffusion rather than a literal or complete two-step flow.

## Entities, states, and communication roles

- **Adaptive social agents:** all agents hold a private Beta belief, may originate a message, may receive messages, update their beliefs, and participate in network adaptation.
- **Opinion leaders:** adaptive social agents whose leader status can modify particular mechanisms, including origination probability, structural reach, message weight, and network evaluation.
- **Ordinary agents:** adaptive social agents governed by the same basic process but without the corresponding leader advantages.
- **Messages:** observable expressions produced by agents. Each message retains its producer identity and stance so that exposure and influence can depend on source role.
- **Communication network:** a directed and potentially adaptive relation indicating which producers are available to each consumer.

`leader` and `ordinary` are agent roles. `originator` and `consumer` describe communication roles within a round:

- an **originator** has been selected to produce one message;
- a **consumer** is an agent exposed to a produced message.

The same agent can occupy different communication roles within and across rounds.

## Message origination

At the beginning of each round, every adaptive agent is eligible to originate a message. The origination rule is shared across roles, but opinion leaders receive a higher origination probability through a multiplicative advantage on the odds scale. A common decay term represents declining interest after the external-information stage. A successful origination draw produces exactly one message; there is no second posting or production-probability gate. Activity heterogeneity, network position, and exposure-dependent extensions are deferred.

The message stance is sampled from the originator's start-of-round Beta belief. The probability of a supportive message is the Beta probability mass above the neutral threshold of one half. This mapping uses both the belief's mean and its concentration without adding a second behavioral decision.

Keeping the origination draw separate from conditional stance formation prevents a leader's greater likelihood of speaking from being conflated with opinion direction. For the bounded mechanism definition, see [Posting Origination](posting-origination.md).

## Diffusion, exposure, and influence

Produced messages diffuse through the directed social network. All produced messages, whether from leaders or ordinary agents, are eligible for network-based transmission. Opinion leaders can have greater reach through their structural position or through an explicit leader-conditioned delivery rule, but the chosen representation remains open.

This case does not include algorithmic ranking, recommendation, or out-of-network platform amplification. Those are platform mechanisms rather than opinion-leader mechanisms.

Unequal influence has two separable components:

1. **Exposure or reach:** whether and how widely a message is received.
2. **Evidence weight:** how strongly a received message affects the recipient's belief update.

Messages retain producer identity so that leader and ordinary messages can receive different weights. The substantive interpretation of a larger leader weight—such as credibility, persuasion, or evidential strength—remains open.

## Opinion updating and network adaptation

Each recipient aggregates the messages received during the round and updates its private Beta belief once. Raw exposure counts and weighted evidence remain distinguishable so that message availability is not conflated with influence.

After exposure and belief updating, agents may revise their directed network relations using information from the messages and producers encountered during the round. Leader status may affect creator evaluation or tie formation, but the exact network-adaptation rule and whether it evaluates pre-update or proposed post-update beliefs remain open.

## Round schedule

Each round follows one synchronous sequence:

1. Read agent beliefs and the network from the state at the start of round `t`.
2. Draw whether each agent originates a message, with a higher origination probability for opinion leaders. Each successful draw produces one message whose stance is sampled from the agent's start-of-round Beta belief.
3. Diffuse originated messages through the current directed network and determine each agent's exposures.
4. Aggregate received messages, including any source-dependent evidence weights.
5. Propose one updated Beta belief for each agent.
6. Propose network changes from the round's encounters under the specified network-adaptation rule.
7. Commit the proposed beliefs and network as the state for round `t+1`.

An updated belief or network position can affect message origination only from the next round onward.

## Current boundaries and open decisions

The current case includes ordinary-agent and leader message origination, network-based diffusion, opinion updating, and network adaptation. It excludes a separate message-production decision, a press entity, externally scheduled press messages, algorithmic ranking, out-of-network recommendation, and engagement-driven platform feedback.

The current basic decisions are the decaying logistic origination probability, the leader log-odds advantage, and the Beta-tail stance mapping documented in [Posting Origination](posting-origination.md). Open decisions include parameter justification or calibration, the representation of leader reach, the meaning and magnitude of leader evidence weight, the network-adaptation rule, and whether leader access to prior external information requires a distinct initialization mechanism.

Leader initialization, origination advantage, reach, evidence weight, and network attractiveness should remain separable so that their individual and combined effects can later be examined.
