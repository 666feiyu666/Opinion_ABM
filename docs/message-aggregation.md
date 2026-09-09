# Message Aggregation

> **Status:** Working mechanism definition and tested implementation
>
> **Case:** `opleader`

## Purpose and boundary

Message aggregation converts the messages received by one agent in one round into explicit supportive and opposing evidence for that agent's next belief state. It reads the exposures produced by message selection, the stance of each exposed message, the role of its originator, and a common base evidence weight. It does not determine which messages are delivered or update the agent directly.

The current mechanism distinguishes messages from opinion leaders and ordinary agents. Ordinary messages provide the reference contribution. Opinion-leader messages may contribute more evidence through one relative source multiplier. Recipient role is not an input to the weighting rule.

## Working definition

Let $\mathcal{E}_i^t$ be the exposures received by agent $i$ in round $t$. For each exposure $e\in\mathcal{E}_i^t$, let $x_e\in\{-1,+1\}$ be the message stance and let $L_e=1$ when the originator is an opinion leader and $L_e=0$ when the originator is an ordinary agent.

Let $\eta\geq0$ be the common base evidence weight and let $\omega_L\geq0$ be the relative opinion-leader multiplier. The source multiplier for an exposure is

$$
w_e
=
\begin{cases}
\omega_L, & L_e=1,\\
1, & L_e=0.
\end{cases}
$$

The supportive and opposing weighted evidence received by agent $i$ are

$$
W_{i,+}^t
=
\eta
\sum_{e\in\mathcal{E}_i^t}
w_e\,\mathbf{1}(x_e=+1),
$$

$$
W_{i,-}^t
=
\eta
\sum_{e\in\mathcal{E}_i^t}
w_e\,\mathbf{1}(x_e=-1).
$$

Equivalently, if $n_{i,O,+}^t$ and $n_{i,L,+}^t$ count supportive ordinary and leader messages, and $n_{i,O,-}^t$ and $n_{i,L,-}^t$ count opposing ordinary and leader messages, then

$$
W_{i,+}^t
=
\eta\left(n_{i,O,+}^t+\omega_L n_{i,L,+}^t\right),
$$

$$
W_{i,-}^t
=
\eta\left(n_{i,O,-}^t+\omega_L n_{i,L,-}^t\right).
$$

Raw supportive and opposing exposure counts remain unweighted. Each exposure contributes once, and contributions add linearly without depending on exposure order. This keeps message availability distinguishable from the evidential influence assigned to those messages.

## Effect on belief updating

Aggregation returns the raw counts together with $W_{i,+}^t$ and $W_{i,-}^t$. The existing opinion-update rule uses those weighted quantities as Beta evidence:

$$
a_i^{t+1}=a_i^t+W_{i,+}^t,
\qquad
b_i^{t+1}=b_i^t+W_{i,-}^t.
$$

The updated belief concentration is therefore

$$
a_i^{t+1}+b_i^{t+1}
=
a_i^t+b_i^t+W_{i,+}^t+W_{i,-}^t.
$$

Positive-weight messages increase concentration as well as potentially changing the belief mean. The mean remains unchanged when the balance of newly weighted evidence matches the prior mean; for example, equal supportive and opposing evidence leave a prior mean of one half unchanged while increasing its concentration.

## Interpretation and status

Two-step-flow research motivates examining whether influence depends on who communicates a message. It does not uniquely specify a linear evidence multiplier or determine its numerical value. Using $\omega_L$ is therefore a model operationalization rather than a direct reproduction of a source-defined equation.

The homogeneous null is $\omega_L=1$. A value $\omega_L>1$ represents the working hypothesis that an opinion-leader message contributes more evidence than an otherwise comparable ordinary-agent message. Whether that greater contribution should be interpreted as credibility, persuasiveness, evidential strength, or another construct remains open, and no value is currently calibrated or empirically validated.

Because recipient role does not enter the rule, leader and ordinary recipients with identical prior beliefs and identical exposures receive identical evidence and update identically. Any difference caused by role-specific initial beliefs remains a separate initialization mechanism.

## Timing and interfaces

Aggregation occurs after all current-round messages have been originated and delivered through the fixed social network. It operates separately for each recipient on that recipient's complete exposure set and produces one aggregate before opinion updating. All proposed beliefs are then committed synchronously as the state for round $t+1$.

Message selection remains stance-blind and role-blind. It preserves the originator identity and message stance in each exposure so that aggregation can recover the source role and place the contribution in supportive or opposing evidence.

## Boundary behavior and deferred extensions

The current definition has the following boundary behavior:

- no exposures produce zero counts and zero weighted evidence;
- $\eta=0$ preserves raw exposure counts but produces no weighted evidence;
- $\omega_L=1$ reproduces homogeneous aggregation in which every message has the same weight;
- aggregation is invariant to exposure order;
- originator membership in the configured leader set determines leader status, while other valid social-agent producers are ordinary; and
- exposures for different recipients are aggregated separately.

The basic rule does not include recipient-specific susceptibility, nonlinear saturation, repetition discounting, dependence among messages, forgetting, stance-dependent credibility, or endogenous changes in source weight. These are distinct mechanisms that would require their own substantive interpretation.

The source-only structure and the use of ordinary messages as the reference are researcher-approved current decisions. The interpretation and magnitude of $\omega_L$ remain open. The tested implementation and mechanism probe exercise this definition at the isolated aggregation-and-update boundary.
