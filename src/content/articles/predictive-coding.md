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

## Expand the neighboring messages
Let $h_i,r_i\in\mathbb R^{d_i}$ and $J_{i+1}=\partial f_{i+1}/\partial h_i\in\mathbb R^{d_{i+1}\times d_i}$. For an interior hidden activity in the stated energy,

$$
\nabla_{h_i}\mathcal F=\rho r_i-J_{i+1}^\top\rho r_{i+1}.
$$

The output boundary differs: $\nabla_{h_{L-1}}\mathcal F=\rho r_{L-1}+J_L^\top\nabla_{\widehat y}\ell(y,\widehat y)$. The transpose maps a neighboring error back into the activity's own dimension; $h_0=x$ remains clamped. This is locality between neighboring layers, not a proof of a biological circuit. [2](#ref-pcalm)

For the figure's scalar network, $f_1(x)=0.8x$, $f_2(h_1)=0.7h_1$, $\widehat y=1.2h_2$, $\rho=1$ and $y=0.2$. Forward initialization gives $(h_1,h_2)=(0.8,0.56)$ and $(r_1,r_2)=(0,0)$. Thus

$$
\partial_{h_1}\mathcal F=r_1-0.7r_2=0,\quad
\partial_{h_2}\mathcal F=r_2+1.2(1.2h_2-0.2)=0.5664.
$$

One **simultaneous** step of size $0.06$ gives $(h_1,h_2)=(0.8,0.526016)$ and new residual $r_2=-0.033984$. At the next step, this neighboring mismatch reaches $h_1$ through $-0.7r_2$. Target pressure therefore changes the last hidden state before earlier states respond. Only after inference do weights change: in this linear example $\nabla_{W_1}\mathcal F=-\rho r_1x$, $\nabla_{W_2}\mathcal F=-\rho r_2h_1$, and $\nabla_{W_3}\mathcal F=(\widehat y-y)h_2$.

The figure opens in **Penalty PC**. Its optional PC-ALM comparison adds $\lambda_i^\top r_i$ to the energy, uses composite error $e_i=\lambda_i+\rho r_i$, and accumulates $\lambda_i$ after nonfinal activity steps. See [the multiplier derivation](#/article/pc-alm) before interpreting that comparison. No multiplier is present in the penalty-only derivation above.

## Why the penalty matters
At a stationary point of a finite quadratic penalty, residuals need not be zero. Therefore activity-inferred weight updates need not equal backpropagation gradients of the original feed-forward network. The relationship depends on the formulation, limits and inference assumptions. One should not turn a correspondence result into a blanket claim of exact equivalence for finite-budget nonlinear networks.

## Cost and interpretation
Inference entails repeated local computations. It can offer an interesting parallel or distributed organization without being faster on a conventional GPU. Biological plausibility also requires more than local communication: weight symmetry, timing, error coding and plasticity remain substantive questions. The Sakana page makes the iterative inference structure explicit. [3](#ref-pcalm-project)

## Experiment
On a small fixed network, compare local parameter-update vectors with exact backpropagation through cosine similarity and relative norm error as inference steps increase. Plot wall-clock cost as well. Then read [PC-ALM](#/article/pc-alm), which introduces an accumulated constraint-error state rather than only a quadratic penalty.
