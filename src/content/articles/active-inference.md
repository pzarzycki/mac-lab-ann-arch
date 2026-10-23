## Infer hidden causes
Let $o$ be observed data, $s$ hidden states, $p(o,s)$ a generative model and $q(s)$ an approximate posterior. Variational free energy is

$$
F[q]=\mathbb E_q[\log q(s)-\log p(o,s)]
=D_{\mathrm{KL}}(q(s)\|p(s\mid o))-\log p(o).
$$

For a valid model with finite quantities, $F$ bounds negative log evidence. Minimizing it with respect to $q$ improves the approximate posterior; this does not make every numerical quantity called “energy” physical energy in joules. [1](#ref-active)

## Act to shape future observations
Active inference extends belief updating to action or policy selection. It uses a generative model of state transitions and observations, with preferences and uncertainty contributing to evaluation of possible futures. Expected free energy is a policy-dependent quantity; its exact form and decomposition depend on the model and conventions. It should not be substituted blindly for the current-observation $F[q]$ above. [2](#ref-active-process)

## Relation to predictive coding
Predictive coding can implement parts of inference under particular generative models and approximations. Active inference supplies a broader account of inference and action. PC-ALM instead studies an augmented-Lagrangian route to local credit assignment for a constrained supervised network. Shared words such as “prediction error,” “inference” and “free energy” do not make these objectives identical. [3](#ref-pcalm)

| Quantity | What it evaluates | Essential specification |
| --- | --- | --- |
| Variational free energy | Approximate posterior for observations | Generative model and posterior family |
| Expected free energy | Candidate futures/policies | Preferences, transitions and beliefs |
| PC constraint energy | Consistency of neural activities | Layer functions, clamping and penalties |

## A two-state, two-action instance
Here is a synthetic one-step partially observed model. Hidden state $s\in\{\mathrm{good},\mathrm{bad}\}$ is never shown to the agent. Outcomes are $o\in\{\mathrm{bright},\mathrm{dim}\}$. Start from prior $q_0=(0.5,0.5)^\top$. The likelihood and two action transitions are column-stochastic matrices (columns are the previous state):

$$
A=\begin{bmatrix}a&1-a\\1-a&a\end{bmatrix},\quad
B_{\mathrm{stay}}=I,\quad
B_{\mathrm{reset}}=\begin{bmatrix}0.85&0.85\\0.15&0.15\end{bmatrix}.
$$

$A$ has outcome rows and state columns. Observing bright gives $q(s\mid\mathrm{bright})\propto A_{\mathrm{bright},s}q_0(s)$; with $a=0.7$, the posterior is $(0.7,0.3)$. Each policy is just one action, with predicted state $q_\pi(s')=B_\pi q(s)$ and outcome $q_\pi(o')=Aq_\pi(s')$. Stay predicts bright with probability $0.58$; Reset predicts $0.64$. These are model predictions, not measurements of the actual next state.

Choose preferred outcome distribution $C=(0.8,0.2)$ and natural logarithms. For this finite-state, known-likelihood example, use the risk-plus-ambiguity expected-free-energy convention from the process theory: [2](#ref-active-process)

$$
G(\pi)=D_{\mathrm{KL}}(q_\pi(o')\|C)
+\mathbb E_{q_\pi(s')}H[A_{\cdot,s'}]
=\underbrace{-\mathbb E_{q_\pi(o')}\log C(o')}_{\text{preference cost}}
-\underbrace{I_{q_\pi}(s';o')}_{\text{information gain}}.
$$

Here $H$ is categorical entropy and $I$ is mutual information under $q_\pi(s')A(o'\mid s')$. The information gain is about the **future state**, after the action; it is not information about the pre-reset state. With a uniform policy prior and fixed precision $\gamma=4$, set $q(\pi)=\operatorname{softmax}(-\gamma G(\pi))$. This reduced rule assumes equal past-observation evidence across policies. Choose the lower-$G$ action or sample from this posterior; the figure shows both scores and posterior probabilities.

| One-step policy | Preference cost | Information gain | $G$ (nats) | Policy probability |
| --- | --- | --- | --- | --- |
| Stay | 0.8054 | 0.0694 | 0.7360 | 0.4439 |
| Reset | 0.7222 | 0.0426 | 0.6797 | 0.5561 |

Reset wins at these settings despite offering less information: its preference cost is lower enough. Change the preferred-outcome probability or likelihood accuracy to inspect the trade-off. At $a=0.5$, observations reveal no state information and both actions predict the same outcomes, so their scores tie. There is no learned transition model, extra action cost, planning horizon or parameter-uncertainty exploration in this toy example. After acting in a real environment, observe again and repeat Bayesian filtering; a predicted outcome must not be substituted for an observation.

## Evidence and limits
The cited book and process theory formalize a framework. They do not establish that all biological learning or every effective controller implements one exact algorithm. Compare against Bayesian filtering and conventional control/RL baselines at matched model knowledge and information access.
