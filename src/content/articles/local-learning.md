## Change the feedback path
Feedback alignment studies whether fixed random feedback weights can support useful learning instead of exact transposed forward weights. A schematic update replaces $W_{\ell+1}^\top$ by a fixed compatible matrix $B_\ell$:

$$
\widetilde\delta_\ell=(B_\ell\widetilde\delta_{\ell+1})\odot\sigma'(a_\ell).
$$

This is an approximate credit signal; it is not generally the gradient of the original loss. Forward weights may learn representations that make the feedback useful in the tested setting. The result relaxes exact weight symmetry, not every biological requirement. [1](#ref-feedback)

## Give a layer its own objective
Another design attaches an auxiliary predictor to intermediate features. A layer can reduce $\mathcal L_\ell(g_\ell(h_\ell),y)$ without waiting for the final network's exact gradient. That local goal may conflict with downstream needs. “Local loss” specifies where a target acts; it does not guarantee globally optimal representations.

## Enforce neighboring consistency
Predictive coding instead uses hidden-state inference and neighboring prediction errors. PC-ALM adds multipliers to accumulate constraint violations. These methods address the same broad credit-assignment problem through different mathematical constructions. [2](#ref-pc) [3](#ref-pcalm)

| Family | Credit source | Primary caveat |
| --- | --- | --- |
| Feedback alignment | Approximate feedback projection | No general exact-gradient guarantee |
| Layer-local objective | Auxiliary target at each layer | Local goals may conflict |
| Predictive coding | Inferred neighboring consistency | Finite inference and equilibrium assumptions |
| PC-ALM | Primal–dual local dynamics | Proven scope narrower than nonlinear use |

## Implementation discipline
```python
# Fixed random feedback is a parameter of the learning algorithm.
feedback.requires_grad_(False)
local_error = feedback @ downstream_error
```

State precisely which tensors require gradients and which update in each phase. An implementation that silently computes the full global gradient for the “local” update defeats the intended comparison.

## What remains open
Scaling local learning to modern language tasks requires evidence about optimization, data efficiency and systems cost, not only shallow classification. A fair study reports success and instability across seeds, depth, width and nonlinearities, with a conventional BP baseline trained comparably.
