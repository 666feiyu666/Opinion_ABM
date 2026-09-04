# Mass-Communication Opinion-Leader Model: Rough Design

> Status: Reviewed  
> Case: `opleader`  
> Scope: Mass-media-era opinion leadership without an online platform

## Purpose

This model examines how opinion leaders mediate influence between mass media and a less-active audience. It is a topic-specific reconstruction inspired by Katz's two-step-flow research, not a complete reproduction of that theory. The model will separate leaders' access, reach, and influence so that their effects can later be examined independently.

## Theoretical interpretation

For one focal topic, opinion leadership is represented through the following provisional mapping:

| Katz-related property | Candidate representation |
|---|---|
| Greater involvement in the topic | Leaders are eligible to produce topic messages |
| Greater contact with external information | Leaders may receive press messages with a different probability |
| Strategic social location | Leaders have larger fixed outgoing neighborhoods |
| Competence or recognized authority | Leader messages may receive greater evidence weight |
| Domain-specific leadership | Leader status applies only to the focal topic |

## Entities and roles

- **Press:** a non-adaptive originator that produces externally specified messages. It does not receive messages or update an opinion.
- **Opinion leaders:** adaptive agents that receive messages, update their opinions, and produce messages about the focal topic.
- **Ordinary agents:** adaptive agents that receive messages and update their opinions but do not produce topic messages.

"Originator" and "recipient" are communication roles rather than mutually exclusive entity types. The press is only an originator, opinion leaders are both originators and recipients, and ordinary agents are only recipients.

## Communication structure and influence

The communication network is directed and stable. No new ties are formed and no existing ties are removed. Press messages may reach leaders and ordinary agents. Leader messages may reach ordinary agents and other leaders through fixed outgoing ties. Leaders have wider interpersonal reach than ordinary agents, although the rule used to construct this advantage remains open.

Delivery is probabilistic even when a fixed connection exists. Unequal influence has two separate components:

1. **Exposure or reach:** whether and how widely a message is received.
2. **Evidence weight:** how strongly a received message affects the recipient.

Messages must retain their originator identity so that press and leader messages can receive different weights. A leader message may carry greater evidence weight than a press message, but the substantive meaning of this difference—credibility, persuasion, or evidential strength—has not yet been settled.

## Round schedule

Each round follows one synchronous sequence:

1. The press and currently informed leaders produce messages from their states at the start of round `t`.
2. Messages are probabilistically delivered through the fixed communication structure.
3. Each recipient assigns source-dependent weights and aggregates all messages received in the round.
4. Leaders and ordinary agents update once, producing their states for `t+1`.
5. A leader's newly updated state can affect message production only at `t+1`.

Two initial conditions are retained. For a **new topic**, leaders begin without topic knowledge, so the press is initially the only producer; leaders may begin producing after becoming informed. For an **established topic**, leaders begin with prior knowledge and may produce alongside the press from the first round.

## Current boundaries and open decisions

The first model excludes ordinary-agent production, network adaptation, algorithmic ranking, engagement feedback, and same-round influence on leader production.

Open decisions include how to represent absence of knowledge versus a neutral opinion; how leaders form messages from their updated opinions; how press exposure and leader reach are assigned; and whether evidence weight changes persuasion, confidence, or both. Leader production, media exposure, reach, and evidence weight should remain separable in later simulation experiments even if the complete model combines them.
