# Platform Network Update

> **Status:** Rough mechanism definition for review
>
> **Case:** `platform`

## Purpose and boundary

Platform network update describes how an agent may change whom it follows after
processing messages in one round. Out-of-network exposures provide opportunities
to form new following ties, while exposures from currently followed producers
provide opportunities to dissolve existing ties.

The mechanism combines two working ideas:

1. **degree regulation:** agents with fewer current following ties are more
   likely to form ties and less likely to dissolve them;
2. **alignment-based homophily:** agents are more likely to follow producers
   whose observed messages align with their beliefs and more likely to unfollow
   producers whose messages oppose their beliefs.

Isolation is possible. The model does not impose a hard minimum number of
following ties. Instead, low degree creates a soft tendency toward forming ties
and retaining existing ones. Platform-mediated out-of-network exposure gives an
isolated agent opportunities to reconnect in later rounds.

Network update does not decide which messages become exposures. It reads the
retained exposures produced by [Platform Message Selection](platform-message-selection.md)
and proposes a network for the next round.

## Directed-tie meaning

Let $G^t=(V,E^t)$ be the directed network at the beginning of round $t$. A tie

$$
i\rightarrow j
$$

means that consumer $i$ follows producer $j$ and is therefore eligible to
receive $j$'s messages through an existing tie. Let

$$
N_i^t=\{j\mid(i,j)\in E^t\}
$$

be the set of producers followed by $i$, and let

$$
k_i^t=|N_i^t|
$$

be the agent's start-of-round following degree.

All network-update decisions in round $t$ classify relationships using $G^t$.
No proposed addition or removal changes another decision within the same round.

## Exposure-based candidate sets

Let $\mathcal{E}_i^t$ be the messages retained by consumer $i$ after platform
message selection and the processing-capacity limit. Each unique exposed
producer is evaluated at most once during the round.

The candidate set for tie formation is

$$
\mathcal{F}_i^t
=
\left\{
j
\;\middle|\;
i\text{ processed a message from }j,
\;j\notin N_i^t,
\;j\neq i
\right\}.
$$

These are producers encountered through retained out-of-network messages.

The candidate set for tie dissolution is

$$
\mathcal{U}_i^t
=
\left\{
j
\;\middle|\;
i\text{ processed a message from }j,
\;j\in N_i^t
\right\}.
$$

An existing tie is therefore considered for dissolution only when the agent
actually processes a message from that producer in the current round. A message
that entered the candidate-availability pool but was removed by the capacity
limit does not create a network-update opportunity.

Although tied and out-of-network messages compete equally for attention during
selection, their relationship to the start-of-round network remains identifiable
and determines which network action is available after exposure.

## Alignment representation

Let consumer $i$'s private belief at the beginning of round $t$ be

$$
B_i^t=\operatorname{Beta}(a_i^t,b_i^t).
$$

The probability mass that this belief assigns to the supportive side of the
neutral threshold is

$$
q_i^t
=
\Pr(\theta_i^t>0.5)
=
1-F_{\operatorname{Beta}}(0.5;a_i^t,b_i^t).
$$

For a processed message with stance $x_j^t\in\{-1,+1\}$, the working alignment
score is

$$
a_{ij}^t
=
x_j^t\left(2q_i^t-1\right)
\in[-1,1].
$$

Equivalently,

$$
a_{ij}^t
=
\begin{cases}
2q_i^t-1, & x_j^t=+1,\\[4pt]
1-2q_i^t, & x_j^t=-1.
\end{cases}
$$

A positive value indicates alignment, a negative value indicates opposition,
and a value near zero indicates that the recipient's belief does not strongly
favor either stance. A larger absolute value represents stronger directional
confidence and therefore a stronger alignment or opposition signal for network
adaptation.

This representation uses both Beta shape parameters rather than only the belief
mean. Consequently, belief concentration can affect tie decisions: when a Beta
belief becomes more concentrated on one side of the neutral threshold, messages
on that side receive a stronger positive alignment score and messages on the
other side receive a stronger negative score. The definition is also consistent
with the Beta-tail mapping used in message origination. It does not require the
consumer to originate a message or possess a separately stored public stance.

## Tie-formation probability

For each $j\in\mathcal{F}_i^t$, the probability of proposing a new tie is

$$
p_{ij,+}^t=f_{+}(k_i^t,a_{ij}^t).
$$

The working qualitative requirements are

$$
\frac{\partial f_{+}}{\partial k_i^t}<0,
\qquad
\frac{\partial f_{+}}{\partial a_{ij}^t}>0.
$$

Agents with fewer following ties are therefore more likely to follow an
encountered out-of-network producer, and greater alignment increases that
probability. The proposed tie is sampled as

$$
Z_{ij,+}^t\sim\operatorname{Bernoulli}(p_{ij,+}^t).
$$

The exact functional form and parameter values of $f_{+}$ remain open.

## Tie-dissolution probability

For each $j\in\mathcal{U}_i^t$, the probability of proposing removal of the
existing tie is

$$
p_{ij,-}^t=f_{-}(k_i^t,a_{ij}^t).
$$

The working qualitative requirements are

$$
\frac{\partial f_{-}}{\partial k_i^t}>0,
\qquad
\frac{\partial f_{-}}{\partial a_{ij}^t}<0.
$$

Agents with more following ties are therefore more likely to unfollow an
encountered producer, while greater alignment reduces that probability. The
proposed removal is sampled as

$$
Z_{ij,-}^t\sim\operatorname{Bernoulli}(p_{ij,-}^t).
$$

The exact functional form and parameter values of $f_{-}$ remain open. Even at
low degree, dissolution is not prohibited; its probability is only expected to
be lower.

## Multiple decisions within one round

The model does not impose a one-follow or one-unfollow limit. Conditional on the
start-of-round state, one Bernoulli decision is made for every unique producer
in $\mathcal{F}_i^t$ and $\mathcal{U}_i^t$. An agent may therefore form several
ties, dissolve several ties, perform both kinds of change, or make no change in
one round.

The message-processing capacity already bounds the number of encountered
producers. With the current capacity $K=10$, an agent can evaluate at most ten
unique producers in a round. All probabilities use the same $k_i^t$ rather than
an incrementally updated degree, so the result does not depend on the order in
which exposures are processed.

Conditional on the common start-of-round state, the producer-level Bernoulli
decisions are treated as independent in the initial mechanism.

## Synchronous network commitment

Let $A^t$ be the complete set of accepted tie additions and let $R^t$ be the
complete set of accepted tie removals across all agents. The next network is

$$
E^{t+1}=(E^t\setminus R^t)\cup A^t.
$$

Every decision reads the start-of-round network, degree, belief, and current
exposures. Additions and removals are committed together after the current-round
message selection and agent-level proposals have been computed. A newly formed
tie cannot deliver a message until round $t+1$, and a tie proposed for removal
remains part of the round-$t$ selection environment.

## Boundary behavior

- An agent with no retained exposures proposes no network changes.
- An agent with $k_i^t=0$ has no existing tie to dissolve but may form ties to
  retained out-of-network producers.
- An isolated agent is not guaranteed to reconnect in a particular round.
- An agent with no retained out-of-network exposure cannot form a new tie in
  that round.
- An existing tie whose producer is not encountered is not considered for
  dissolution in that round.
- Self-ties, duplicate ties, additions of existing ties, and removals of absent
  ties are invalid.
- A complete network supplies no formation candidates but may supply
  dissolution candidates.
- An empty network supplies no dissolution candidates, while platform-mediated
  out-of-network exposure may still supply formation candidates.

## Interpretation and current status

Tie formation and dissolution are agent adaptation mechanisms enabled by
platform-mediated encounters; they are not direct platform ranking decisions.
They are included in the `platform` mechanism family because out-of-network
selection creates the encounters through which the network can change.

The exposure-based candidate sets, degree and alignment directions, soft
isolation tendency, producer-level decisions, $K=10$ opportunity bound, and
synchronous commitment are current working decisions. The alignment score is a
working representation for review. The forms and parameters of $f_{+}$ and
$f_{-}$ remain unresolved.

The current mechanism excludes a fixed target degree, forced one-for-one
rewiring, guaranteed connectivity, triadic closure, reciprocity preference,
preferential attachment, global network optimization, cumulative relationship
memory, popularity-based following, opinion-leader status, and network changes
caused by messages the agent did not process. These are possible extensions
rather than hidden parts of the initial platform case.
