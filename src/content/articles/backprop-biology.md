## The mathematical operation
For $a_\ell=W_\ell h_{\ell-1}$ and $h_\ell=\sigma(a_\ell)$, define $\delta_\ell=\partial\mathcal L/\partial a_\ell$. Backpropagation applies the chain rule:

$$
\delta_\ell=(W_{\ell+1}^\top\delta_{\ell+1})\odot\sigma'(a_\ell),\qquad
\nabla_{W_\ell}\mathcal L=\delta_\ell h_{\ell-1}^\top.
$$

The first equation transports an error signal; the second combines it with presynaptic activity. With a batch, gradients sum or average over samples. Reverse-mode differentiation efficiently computes a scalar objective's derivatives; the 1986 papers are major milestones, not the entire historical origin of the chain rule. [1](#ref-backprop)

## A signed calculation
The figure uses one sample, column vectors $x=(1,-1)^\top\in\mathbb R^2$, $W_1=\left[\begin{smallmatrix}0.5&-0.3\\-0.2&0.4\end{smallmatrix}\right]\in\mathbb R^{2\times2}$, $h=\tanh(W_1x)$, and a scalar linear readout $\widehat y=W_2h$ with $W_2=(0.7,-0.5)\in\mathbb R^{1\times2}$. There are no biases. With $y=0.2$ and $\mathcal L=\frac12(\widehat y-y)^2$, forward computation gives

$$
a=(0.8,-0.6)^\top,\quad h\approx(0.664,-0.537)^\top,\quad \widehat y\approx0.733.
$$

Because the output is linear, $\delta_2=\widehat y-y\approx0.533$. Reverse credit is $W_2^\top\delta_2\approx(0.373,-0.267)^\top$. Multiply elementwise by $1-h^2\approx(0.559,0.712)^\top$ to obtain $\delta_1\approx(0.209,-0.190)^\top$. The outer product with the **stored input**, rather than the error alone, gives

$$
\nabla_{W_1}\mathcal L\approx
\begin{bmatrix}0.209&-0.209\\-0.190&0.190\end{bmatrix},\qquad
\Delta W_1=-\eta\nabla_{W_1}\mathcal L.
$$

Changing a downstream weight changes both the output error and its reverse projection. Changing the target reverses some gradient signs when the prediction crosses it. The figure recomputes these dependencies; its numbers are synthetic.

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

| Construction | Algorithmic evidence and constraint | What still has to be supplied | Biological evidence |
| --- | --- | --- | --- |
| Exact BP | Chain rule gives the exact gradient for this differentiable objective | Transposed weights, activation derivatives, stored forward activity and target error | Derivative correctness alone makes no circuit claim |
| Feedback alignment | Fixed random $B$ replaces exact $W^\top$; useful learning demonstrated in the cited networks [2](#ref-feedback) | Signed feedback, local derivative, presynaptic activity and downstream error; gradient equality is not guaranteed | Relaxing symmetry does not identify a demonstrated biological implementation |

For the fixed network in the next article, the displayed random feedback actually points against exact BP (cosine about $-0.879$). Alignment is a possible result of forward-weight learning over training, not a property guaranteed for an arbitrary initial network.

## Experiment
On a fixed small network, compare update direction to exact BP. Then vary depth and inference budget. Report cosine similarity, norm ratio, loss reduction and runtime. Keep this diagnostic distinct from a full training experiment that changes the weights after each batch.
