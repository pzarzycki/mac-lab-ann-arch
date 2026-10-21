## Free the hidden activities
A feed-forward network ordinarily sets $h_i=f_i(h_{i-1};W_i)$ in a single computation. A supervised predictive-coding construction instead treats hidden activities $h_i$ as variables and penalizes violations of these relationships:

$$
r_i=h_i-f_i(h_{i-1};W_i),\qquad
\mathcal F=\ell(y,f_L(h_{L-1};W_L))+\frac\rho2\sum_{i=1}^{L-1}\|r_i\|^2.
$$

$h_0=x$ is clamped input, $y$ the target, $W_i$ layer parameters, $\rho>0$ a penalty and $r_i$ the prediction error. This is one supervised PC formulation; the broader literature includes different generative directions, clamping schemes and approximations. [1](#ref-pc) [2](#ref-pcalm)

## Inference versus parameter learning
During activity inference, keep $W$ fixed and update $h_i\leftarrow h_i-\eta_h\nabla_{h_i}\mathcal F$. Each activity gradient depends on adjacent-layer consistency errors and the local function's Jacobian. After a finite inference budget, update weights using the resulting local quantities.

```python
h = forward_initialize(x, W)
for step in range(inference_budget):
    h = simultaneous_activity_step(h, W, y)
W = one_batch_aggregated_parameter_step(h, W, y)
```

“Local” here means neighboring layers in the computational graph. An implementation may use automatic differentiation to compute local derivatives; locality is not the absence of any derivative calculation.

## Why the penalty matters
At a stationary point of a finite quadratic penalty, residuals need not be zero. Therefore activity-inferred weight updates need not equal backpropagation gradients of the original feed-forward network. The relationship depends on the formulation, limits and inference assumptions. One should not turn a correspondence result into a blanket claim of exact equivalence for finite-budget nonlinear networks.

## Cost and interpretation
Inference entails repeated local computations. It can offer an interesting parallel or distributed organization without being faster on a conventional GPU. Biological plausibility also requires more than local communication: weight symmetry, timing, error coding and plasticity remain substantive questions. The Sakana page makes the iterative inference structure explicit. [3](#ref-pcalm-project)

## Experiment
On a small fixed network, compare local parameter-update vectors with exact backpropagation through cosine similarity and relative norm error as inference steps increase. Plot wall-clock cost as well. Then read [PC-ALM](#/article/pc-alm), which introduces an accumulated constraint-error state rather than only a quadratic penalty.
