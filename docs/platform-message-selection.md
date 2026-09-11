# Platform Message Selection

> **Status:** Implemented working mechanism definition
>
> **Case:** `platform`

## Purpose and boundary

Platform message selection determines which messages originated in the current
round become available to each agent and, subject to a finite processing
capacity, which of those messages become exposures. The mechanism extends
communication beyond existing network ties while retaining the same ordinary
message-origination and homogeneous aggregation mechanisms as the shared
baseline.

The current version represents the platform through uniform out-of-network
reach. It does not rank messages by stance, source role, popularity, predicted
engagement, or similarity to the recipient. Opinion leaders are absent from the
`platform` case, and leader status is not an input to this mechanism.

Message selection does not create or remove network ties. It records the
exposures that a later network-update mechanism may use when proposing ties for
the next round.

## Network and message-pool meaning

Let $G^t=(V,E^t)$ be the network at the beginning of round $t$, and let $N_i^t$
be the set of producers connected to potential consumer $i$. A producer
$j\in N_i^t$ is an existing-tie source for $i$; a producer $j\notin N_i^t$ is an
out-of-network source for $i$.

Let $\mathcal{M}^t$ be the complete set of messages originated in round $t$.
Every message retains its originator identity and stance. An agent cannot
receive its own message.

The current rule separates two concepts:

- **candidate availability:** a message enters the set available for the agent's
  attention;
- **exposure:** the message is retained within the agent's processing capacity
  and passed to message aggregation.

This distinction is necessary because an out-of-network message can become
available but still not be processed when more than ten messages compete for
attention.

## Candidate-availability rule

Let $m_j^t\in\mathcal{M}^t$ be a message originated by agent $j$, with $j\neq i$.
Define $A_{ij}^t=1$ when that message becomes available to consumer $i$. The
working rule is

$$
A_{ij}^{t}
=
\begin{cases}
1, & j\in N_i^t,\\[4pt]
\operatorname{Bernoulli}(\rho), & j\notin N_i^t,
\end{cases}
$$

where $\rho\in[0,1]$ is the **out-of-network availability probability**.
Existing-tie messages therefore become available deterministically. Each
out-of-network message--consumer pair receives an independent Bernoulli draw.

The candidate set for consumer $i$ is

$$
\mathcal{C}_i^t
=
\left\{
m_j^t\in\mathcal{M}^t
\;\middle|\;
j\neq i,\; A_{ij}^t=1
\right\}.
$$

For interpretation, this set can be decomposed into

$$
\mathcal{C}_{i,\mathrm{tie}}^t
=
\left\{
m_j^t\in\mathcal{M}^t
\;\middle|\;
j\neq i,\;j\in N_i^t
\right\}
$$

and

$$
\mathcal{C}_{i,\mathrm{out}}^t
=
\left\{
m_j^t\in\mathcal{M}^t
\;\middle|\;
j\neq i,\;j\notin N_i^t,\;A_{ij}^t=1
\right\},
$$

so that

$$
\mathcal{C}_i^t
=
\mathcal{C}_{i,\mathrm{tie}}^t
\cup
\mathcal{C}_{i,\mathrm{out}}^t.
$$

The two subsets preserve how each candidate reached the consumer, which is
relevant to interpretation and to the later network-update mechanism. They do
not define different attention priorities. Once the subsets are combined in
$\mathcal{C}_i^t$, tied and out-of-network candidates are equal competitors for
the agent's processing capacity.

Selection does not inspect message stance or any opinion-leader role. Conditional
on their network relation to the consumer, all messages use the same
availability rule.

## Processing capacity and exposure

Each agent can process at most

$$
K=10
$$

messages per round. This finite capacity is part of the platform package;
the two no-platform scenarios use nonbinding capacity. The final exposure set is

$$
\mathcal{E}_i^t
=
\begin{cases}
\mathcal{C}_i^t, & |\mathcal{C}_i^t|\leq K,\\[4pt]
\operatorname{UniformSample}(\mathcal{C}_i^t,K),
& |\mathcal{C}_i^t|>K,
\end{cases}
$$

where sampling is without replacement. When capacity binds, existing-tie and
out-of-network candidates compete in the same uniform attention pool. The rule
does not reserve capacity for existing ties or give out-of-network messages a
ranking advantage after they enter the candidate set.

Consequently, $\rho$ controls the probability that an out-of-network message
becomes available, not its unconditional probability of final exposure. When
the candidate pool exceeds ten messages, final exposure also depends on the
capacity draw.

## Expected behavior

If consumer $i$ has $u_i^t$ current-round messages from untied producers, the
expected number of out-of-network candidates before capacity is

$$
\mathbb{E}[C_{i,\mathrm{out}}^t]=\rho u_i^t.
$$

If a message from producer $j$ has $d_j^t$ existing-tie recipients in a
population of size $N$, its expected candidate reach before capacity is

$$
\mathbb{E}[R_j^t]
=
d_j^t+\rho\left(N-1-d_j^t\right).
$$

Capacity can reduce realized exposure below these candidate-reach quantities.
Increasing $\rho$ weakly increases expected candidate availability, although it
does not guarantee that every additional candidate will survive attention
competition.

## Boundary behavior

- $\rho=0$ reduces candidate availability to existing-tie messages.
- $\rho=1$ makes every non-self message available, regardless of network ties.
- An empty message pool produces no candidates and no exposures.
- If $|\mathcal{C}_i^t|\leq10$, every candidate becomes an exposure.
- If $|\mathcal{C}_i^t|>10$, exactly ten distinct candidates become exposures.
- A complete network contains no out-of-network source pairs, so changing
  $\rho$ has no effect.
- The producer of a message is always excluded from receiving that message.

The current $N=11$ complete-network verification configuration cannot exhibit
out-of-network reach. Mechanism checks therefore require a declared sparse
network, even if the capacity remains fixed at ten.

## Timing and network-update handoff

Selection reads the network $G^t$ and the complete message pool produced from
start-of-round beliefs. It first constructs candidate sets, then applies the
capacity rule, and finally records exposures. Message aggregation and opinion
updating operate on the retained exposure sets.

A later network-update mechanism may read these exposures and propose tie
formation or dissolution. Any proposed network is committed as $G^{t+1}$ and
affects message selection only from the next round onward. Selection in round
$t$ never reads or modifies a partially updated network.

## Randomness and reproducibility

The Bernoulli availability draws and uniform capacity sampling are separate
stochastic operations. Implementations should use stable message ordering and
reproducible random streams so that results do not depend on dictionary or agent
iteration order. Matched scenario runs retain the same initialization and
message-origination streams. `platform` and `baseline` also retain the same
finite capacity, while `null` and `opleader` use nonbinding capacity because
attention competition is outside their boundary.

## Interpretation and current status

This mechanism represents **uniform platform-mediated reach**, not a realistic
or optimized recommendation algorithm. It adds one platform channel: messages
may become available beyond existing ties. The platform capacity rule then models
finite attention without introducing a preference over message content or
source identity.

The use of one Bernoulli parameter, deterministic existing-tie availability,
uniform truncation, $K=10$, self-exclusion, and next-round network changes are
current working decisions. The mechanism itself does not prescribe a calibrated
value of $\rho$. The exploratory platform scenario currently uses $\rho=0.04$
to match the integrated baseline configuration. Formation and dissolution of
network ties are specified separately.

The current mechanism excludes personalization, stance-based ranking,
opinion-similarity ranking, popularity or engagement signals, source-role
preference, recency weighting, repeated recommendation, message forwarding,
same-round cascades, and endogenous adjustment of $\rho$. These are possible
extensions rather than hidden parts of the initial platform case.
