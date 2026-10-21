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

## An implementable starting point
Build a tiny partially observable task with two hidden states, noisy observations and two actions. Specify observation likelihoods, transition matrices and prior preferences numerically. Update beliefs by Bayes' rule before trying a neural approximation. This reveals what the framework assumes rather than attributing decisions to an undefined optimization principle.

## Evidence and limits
The cited book and process theory formalize a framework. They do not establish that all biological learning or every effective controller implements one exact algorithm. Compare against Bayesian filtering and conventional control/RL baselines at matched model knowledge and information access.
