## Change the feedback path
Feedback alignment studies whether fixed random feedback weights can support useful learning instead of exact transposed forward weights. A schematic update replaces $W_{\ell+1}^\top$ by a fixed compatible matrix $B_\ell$:

$$
\widetilde\delta_\ell=(B_\ell\widetilde\delta_{\ell+1})\odot\sigma'(a_\ell).
$$

This is an approximate credit signal; it is not generally the gradient of the original loss. Forward weights may learn representations that make the feedback useful in the tested setting. The result relaxes exact weight symmetry, not every biological requirement. [1](#ref-feedback)

## Shapes, signs and a complete local update
Use the two-input, two-hidden-unit tanh network in [the signed BP trace](#/article/backprop-biology): $a_1=W_1x\in\mathbb R^2$, $h_1=\tanh(a_1)\in\mathbb R^2$, $W_1\in\mathbb R^{2\times2}$, and linear readout $\widehat y=W_2h_1\in\mathbb R$. For squared loss, initialize $\widetilde\delta_2=\widehat y-y\in\mathbb R$; this is a derivative sign convention, so descent subtracts it. Here $B_1\in\mathbb R^{2\times1}$ and

$$
\widetilde\delta_1=(B_1\widetilde\delta_2)\odot(1-h_1^2),\qquad
\Delta W_1=-\eta\widetilde\delta_1x^\top\in\mathbb R^{2\times2}.
$$

At the default values, $B_1=(-0.4,0.9)^\top$ and $\widetilde\delta_2\approx0.533$ produce $\widetilde\delta_1\approx(-0.119,0.342)^\top$. Exact BP instead gives $(0.209,-0.190)^\top$. The cosine is about $-0.879$; neither a useful gradient direction nor biological plausibility follows merely from freezing $B$. The readout can still receive its ordinary gradient $(\widehat y-y)h_1^\top$. [1](#ref-feedback)

## Give a layer its own objective
Another design attaches an auxiliary predictor to intermediate features. A layer can reduce $\mathcal L_\ell(g_\ell(h_\ell),y)$ without waiting for the final network's exact gradient. That local goal may conflict with downstream needs. “Local loss” specifies where a target acts; it does not guarantee globally optimal representations.

Mostafa, Ramesh and Cauwenberghs give a concrete local-error construction: a fixed random classifier maps each layer's activations to class scores; the true training label defines a local loss, and feedback from that classifier teaches the layer. Their paper distinguishes transpose feedback from other feedback choices. “Random” describes classifier weights, not random labels. [4](#ref-local-errors)

The figure uses a regression analogue with a fixed $C=(0.6,0.2)$, scalar target $0.2$ and $\mathcal L_1=\frac12(Ch_1-y)^2$. Its hidden signal is $C^\top(Ch_1-y)\odot(1-h_1^2)\approx(0.031,0.013)^\top$. Update the incoming $W_1$ with its outer product with $x$; hold $C$ fixed and do not update $W_2$ from this head loss. In a deeper block, detach $h_{\ell-1}$ for this local loss so it does not update earlier blocks. A trainable-head variant would also update that head; it is a different specification.

## Enforce neighboring consistency
Predictive coding instead uses hidden-state inference and neighboring prediction errors. PC-ALM adds multipliers to accumulate constraint violations. These methods address the same broad credit-assignment problem through different mathematical constructions. [2](#ref-pc) [3](#ref-pcalm)

| Family | Credit source | Primary caveat |
| --- | --- | --- |
| Feedback alignment | Approximate feedback projection | No general exact-gradient guarantee |
| Layer-local objective | Auxiliary target at each layer | Local goals may conflict |
| Predictive coding | Inferred neighboring consistency | Finite inference and equilibrium assumptions |
| PC-ALM | Primal–dual local dynamics | Proven scope narrower than nonlinear use |

## A separate equilibrium method
Equilibrium propagation uses a free equilibrium and a nearby equilibrium obtained by weakly nudging outputs toward targets in an energy-based recurrent network. The contrast between equilibria supplies a parameter-learning signal under its assumptions. This two-phase equilibrium construction is distinct from a fixed random backward matrix, an independent local classifier, and the PC-ALM multiplier update. Finite settling time and the weak-nudging limit matter to its gradient interpretation. [5](#ref-equilibrium-propagation)

## Implementation discipline
```python
# Fixed random feedback is a parameter of the learning algorithm.
feedback.requires_grad_(False)
local_error = feedback @ downstream_error
```

State precisely which tensors require gradients and which update in each phase. An implementation that silently computes the full global gradient for the “local” update defeats the intended comparison.

## What remains open
Scaling local learning to modern language tasks requires evidence about optimization, data efficiency and systems cost, not only shallow classification. A fair study reports success and instability across seeds, depth, width and nonlinearities, with a conventional BP baseline trained comparably.
