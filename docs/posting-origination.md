# Posting Origination

> **Status:** Working mechanism definition
>
> **Case:** `opleader`

## Purpose and boundary

Before the simulation begins, information about the focal issue has already circulated and agents hold heterogeneous, non-consensual Beta beliefs. These initial beliefs represent the state from which interpersonal communication begins. Posting origination describes how agents introduce new expressions of those beliefs into the simulated interaction process.

## Working definition

In each round `t`, every agent `i` is eligible to originate a message. Let $L_i=1$ for an opinion leader and $L_i=0$ for an ordinary agent. The current basic origination probability is:

$$
\pi_i^t
=
\sigma\!\left[
\operatorname{logit}(\pi_0)
-\lambda(t-1)
+\beta_L L_i
\right],
$$

where $\pi_0\in[0,1]$ is an ordinary agent's origination probability in round 1, $\lambda\geq0$ is the common rate of interest decay, $\beta_L\geq0$ is the opinion-leader advantage, and $\sigma(x)=1/(1+e^{-x})$. For interior values, leader status therefore multiplies the odds of origination by $e^{\beta_L}$ while both roles experience the same temporal decline. The endpoints use the corresponding limiting cases: $\pi_0=0$ remains zero and $\pi_0=1$ remains one for finite decay and leader-advantage parameters.

The origination event is then sampled as:

$$
O_i^t \sim \operatorname{Bernoulli}(\pi_i^t).
$$

If $O_i^t=1$, agent $i$ produces exactly one message. If $O_i^t=0$, it produces none. Origination and message production are therefore the same event in this model; there is no second posting-probability gate. An agent is called an originator only for a message produced in that round, not because it created the underlying issue or information.

Conditional on origination, the message stance $X_i^t\in\{-1,+1\}$ is sampled from the agent's start-of-round Beta belief using the upper-tail probability at the neutral threshold:

$$
B_i^t=\operatorname{Beta}(a_i^t,b_i^t),
\qquad
p_{i,+}^t
=
1-F_{\operatorname{Beta}}(0.5;a_i^t,b_i^t),
$$

$$
\Pr(X_i^t=+1\mid O_i^t=1)=p_{i,+}^t,
\qquad
\Pr(X_i^t=-1\mid O_i^t=1)=1-p_{i,+}^t.
$$

Leader status affects whether a message is originated, not its stance conditional on the agent's belief.

## Extensions and future work

The current mechanism reads the round, leader role, and start-of-round Beta belief. It returns either no message or one message retaining the round, originator identity, and stance. It does not update beliefs, select recipients, diffuse messages, or change network ties.

Later versions may extend $\pi_i^t$ with baseline activity heterogeneity, social-network position, previous-round exposure volume, or exposure--belief alignment. These factors are intentionally excluded from the basic function. A separate production or abstention gate should be introduced only if it receives a distinct substantive interpretation, such as expression cost, strategic silence, or motivation. If accumulated Beta evidence makes stance selection prematurely deterministic, evidence discounting or forgetting may also be considered.

The model boundary and one-stage origination decision are researcher-defined. The logistic probability and Beta-tail stance mapping are the current provisional formulation for review.
