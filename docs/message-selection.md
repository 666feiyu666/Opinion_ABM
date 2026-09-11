# Message Selection

> **Status:** Implemented mechanism definition
>
> **Case:** `opleader`

## Purpose and boundary

Message selection determines which messages originated in the current round become exposures for each agent. The mechanism operates within a fixed directed information-access network containing opinion-leader and ordinary-agent nodes. A recorded tie identifies a producer whose current-round message is available to a particular consumer, and the network does not adapt during the simulation.

The current rule is deliberately narrow: messages travel only through regular interpersonal relationships, and delivery through those relationships is deterministic. Opinion-leader status does not create a separate delivery privilege inside the selection function.

## Social-network meaning

Let $G=(V,E)$ be the social network initialized before the first round. A directed tie $(i,j)\in E$ means that consumer $i$ can receive a current-round message originated by producer $j$. The same network is retained in every round; messages and opinion changes do not create, remove, or rewire ties.

The exploratory scenario begins with an undirected Barabasi-Albert graph and assigns one random direction to each edge using a seed-matched initialization stream. The software representation then records eligible producers for each consumer. This working representation is shared with `main` and `null`; a reciprocal-network alternative is parked for later robustness work rather than mixed into the matched comparison.

Opinion leaders are selected as the top three percent of agents by producer in-degree. Their structural reach is therefore an exogenous initial condition rather than an outcome of communication during the simulation.

## Working definition

Let $\mathcal{M}^t$ be the set of messages originated in round $t$, and let $N_i$ be agent $i$'s fixed set of social contacts. For recipient $i$, the eligible message set is

$$
\mathcal{C}_i^t
=
\left\{
m_j^t\in\mathcal{M}^t
\;\middle|\;
j\in N_i,
\;j\neq i
\right\}.
$$

Every eligible message is delivered:

$$
\Pr\!\left(D_{ij}^t=1\mid j\in N_i\right)=1,
\qquad
\Pr\!\left(D_{ij}^t=1\mid j\notin N_i\right)=0.
$$

The exposures received by $i$ are therefore

$$
\mathcal{E}_i^t
=
\left\{
\operatorname{Exposure}(i,m_j^t)
\;\middle|\;
m_j^t\in\mathcal{C}_i^t
\right\}.
$$

Selection does not inspect the producer's role or the message stance. Conditional on originating a message, an opinion leader and an ordinary agent with the same social contacts have the same delivery pattern. Each exposure retains the message identity, originator identity, round, and stance for the source-dependent rule defined in [Message Aggregation](message-aggregation.md).

## Attention and capacity

Attention competition is not part of the current mechanism. Any configured consumption capacity must be large enough to retain every eligible message for every agent. A binding capacity must not silently truncate or rank messages, because either behavior would introduce an additional attention-selection mechanism that has not been specified.

For a fixed network in which each originator can produce at most one message per round, a sufficient capacity is at least the largest number of eligible producers available to any one agent. The repository configuration uses `agent_count - 1`, guaranteeing that capacity never binds for any valid non-self network.

## Timing and propagation

Selection reads the fixed network and the complete message pool produced from start-of-round beliefs in round $t$. It creates exposures before aggregation and opinion updating. Receiving a message does not cause the same message to be forwarded again within the round, and an updated belief can affect an agent's own message origination only from round $t+1$ onward.

The model can consequently transmit influence across several parts of the network over several rounds through newly originated expressions, but it does not simulate a same-message cascade or same-round rebroadcast process.

## Role of opinion leadership

The current model keeps three possible sources of leader advantage separate:

1. opinion leaders have a higher probability of originating a message;
2. opinion leaders may have greater structural reach through their initialized network position;
3. opinion-leader messages receive a relative evidence multiplier during aggregation, as defined in [Message Aggregation](message-aggregation.md).

Message selection itself adds no fourth advantage. In particular, it does not preferentially deliver leader messages and does not send them to untied agents.

## Excluded extensions

The current selection mechanism excludes stochastic within-tie delivery, out-of-tie leader outreach, global random mixing, algorithmic recommendation, role- or stance-based ranking, binding attention capacity, message forwarding, and network adaptation. Any of these would require a separate substantive interpretation rather than being introduced as implementation detail.

The fixed-network boundary, directed matched initialization, and deterministic tied delivery are researcher-approved current decisions. The selection rule is a model operationalization inspired by interpersonal mediation in two-step-flow research; it is not claimed to be a uniquely specified rule from that literature.
