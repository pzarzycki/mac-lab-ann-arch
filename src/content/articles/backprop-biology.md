## The mathematical operation
For $a_\ell=W_\ell h_{\ell-1}$ and $h_\ell=\sigma(a_\ell)$, define $\delta_\ell=\partial\mathcal L/\partial a_\ell$. Backpropagation applies the chain rule:

$$
\delta_\ell=(W_{\ell+1}^\top\delta_{\ell+1})\odot\sigma'(a_\ell),\qquad
\nabla_{W_\ell}\mathcal L=\delta_\ell h_{\ell-1}^\top.
$$

The first equation transports an error signal; the second combines it with presynaptic activity. With a batch, gradients sum or average over samples. Reverse-mode differentiation efficiently computes a scalar objective's derivatives; the 1986 papers are major milestones, not the entire historical origin of the chain rule. [1](#ref-backprop)

## What could be biologically difficult?
Several separate questions are often compressed into “the brain cannot backprop”:

- **Weight transport:** exact transposed forward weights in a separate feedback path.
- **Timing:** availability of forward activities when downstream errors arrive.
- **Signals:** representing signed errors and derivatives in neural circuits.
- **Locality:** which variables a synapse can access.
- **Objective:** where a global supervised target or reward originates.

A method can relax one issue while retaining others. Feedback alignment changes feedback weights; predictive coding changes the organization of state inference; surrogate gradients permit training spikes but can retain global backpropagation through time. [2](#ref-feedback) [3](#ref-pc) [4](#ref-spikes)

## Avoid a false binary
Numerically approximating the BP gradient is not the same as implementing textbook forward/backward phases. Conversely, using local updates is not experimental proof of biological realization. Plausibility should specify circuitry, time scales, plasticity and signal representations rather than use one yes/no label.

## A useful evaluation table
For each method, record: derivative target, finite-budget approximation, communication graph, weight symmetry, stored state, temporal coordination, measured compute and biological evidence. A lower training loss cannot settle the neuroscience questions, and a plausible circuit cannot substitute for task performance.

## Experiment
On a fixed small network, compare update direction to exact BP. Then vary depth and inference budget. Report cosine similarity, norm ratio, loss reduction and runtime. Keep this diagnostic distinct from a full training experiment that changes the weights after each batch.
