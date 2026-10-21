## The constrained problem
Seely and Gould study layer-local learning by introducing independent hidden activities and constraints that recover a feed-forward network. For a network with two hidden layers and a linear readout, write

$$
h_0=x,\quad r_1=h_1-f_1(x;W_1),\quad
r_2=h_2-f_2(h_1;W_2),\quad\widehat y=W_3h_2.
$$

$x\in\mathbb R^{d_0}$ is one input; $h_1\in\mathbb R^{d_1}$ and $h_2\in\mathbb R^{d_2}$ are variable hidden activities. $W_3\in\mathbb R^{d_y\times d_2}$ maps to target dimension $d_y$. This example defines all layer roles explicitly; counting “three layers” otherwise varies by convention.

## Add a multiplier to each constraint
For squared output loss, an augmented Lagrangian is

$$
\mathcal L_\rho=
\frac12\|y-W_3h_2\|^2+
\sum_{i=1}^{2}\left[\lambda_i^\top r_i+\frac\rho2\|r_i\|^2\right].
$$

$\lambda_i$ has the same shape as $r_i$; $\rho>0$ is the penalty coefficient. Multipliers accumulate constraint violations. They are sample-specific state variables, not another set of weights shared by the dataset. The sign convention here is $r_i=h_i-f_i$; reversing the constraint reverses its multiplier convention. [1](#ref-pcalm)

## Activity gradients and local messages
Define $e_i=\lambda_i+\rho r_i$. With $J_2=\partial f_2/\partial h_1$, the hidden gradients in this example are

$$
\nabla_{h_1}\mathcal L_\rho=e_1-J_2^\top e_2,
\qquad
\nabla_{h_2}\mathcal L_\rho=W_3^\top(W_3h_2-y)+e_2.
$$

$J_2\in\mathbb R^{d_2\times d_1}$. The first expression combines the local consistency signal with a neighboring signal propagated through a local Jacobian. For $f_2(h)=\sigma(W_2h)$, $J_2=\operatorname{diag}(\sigma'(W_2h))W_2$. Thus the method can use nonlinear activations; a linear-network theorem is a narrower claim than the algorithm's implementation.

## One batch, step by step
1. Forward-initialize hidden activities for all examples with the current shared weights.
2. Reset each example's hidden multipliers to zero.
3. Hold weights fixed while taking an activity descent step with step size $\eta_h>0$.
4. Update multipliers using the **new** activities: $\lambda_i\leftarrow\lambda_i+\alpha r_i$, with dual step size $\alpha>0$.
5. Repeat the primal–dual cycle $T-1$ times, then take the final primal step specified by the paper.
6. Average parameter gradients across the batch and perform one parameter update.

```python
# Algorithm sketch following the paper's primal/dual/final-primal ordering.
h = forward_hidden_states(batch_x, weights)
lam = zeros_like(h)
for t in range(T - 1):
    h = primal_step_simultaneous(h, lam, weights, batch_y)
    lam = lam + alpha * constraint_residuals(h, weights)
h = primal_step_simultaneous(h, lam, weights, batch_y)
grad_W = mean_over_examples(local_weight_gradients(h, lam, weights, batch_y))
weights = weights - eta_W * grad_W
```

The code is pseudocode, not a substitute for the reference implementation. With batching, $h_i,r_i,\lambda_i$ have a leading batch dimension. No hidden-state averaging across samples is implied. All sample-wise dynamics can be vectorized; layer dependencies still determine what can run concurrently. [1](#ref-pcalm)

## Evidence, separated by scope
The May 2026 preprint proves convergence to the BP-gradient equilibrium under specified **linear, bias-free, fixed-weight** assumptions and stability conditions. Its nonlinear results are experiments, not the same theorem. The September project page additionally reports deep residual-MLP experiments reaching 1,000 layers on MNIST. Those results concern a particular residual architecture and training setup; they are not evidence of a 1,000-layer Transformer trained with an established speed advantage. [1](#ref-pcalm) [2](#ref-pcalm-project)

The original paper's short training sweeps and the later project page's five-epoch experiment must not be merged into one protocol. Nor should near-BP accuracy be paraphrased as numerical equality in every configuration.

## Computational interpretation
With dense width $d$, batch size $B$, depth $L$ and inference budget $T$, a simple local-iteration accounting scales roughly as $O(TBLd^2)$; constants, layer structure and parallel execution matter. These are repeated activity/Jacobian computations, not merely repeated ordinary forward passes. A comparison to BP needs total work, memory, communication and measured wall-clock time. The benefit studied is the propagation and locality of credit under finite inference, not a demonstrated universal accelerator for today's GPU training stack.

## Biological interpretation and open questions
The accumulated multiplier resembles an integral feedback state. This motivates a local-control interpretation, but it does not identify a biological neuron type implementing the exact algorithm. Symmetric feedback, nonlinear convergence, stability at scale and realistic neural timing remain open questions. [3](#ref-pc)

The adjacent lab solves a **two-hidden-layer scalar linear instance** with fixed weights; it shows actual primal–dual updates and BP reference gradients. It is an educational mechanism check, not a reproduction of the published datasets or accuracy curves.
