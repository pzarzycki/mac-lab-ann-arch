## Correct attribution
Liquid Time-constant Networks are associated with **Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus and Radu Grosu**. The cited paper was first posted in 2020 and accepted at AAAI 2021. Demis Hassabis is not an author of this work. [1](#ref-liquid)

## A changing effective time scale
A schematic LTC neuron follows

$$
\frac{dx(t)}{dt}=-\left[\frac1\tau+f(x(t),I(t),t;\theta)\right]x(t)+f(x(t),I(t),t;\theta)A.
$$

$x(t)$ is a state variable, $I(t)$ input, $\tau>0$ a baseline time constant, $A$ a driving level and $f$ a learned nonlinear gate. The effective time constant is $\tau_{\mathrm{eff}}=\tau/(1+\tau f)$ under this scalar form with $f\ge0$ and $\tau>0$, as in the bounded gate below. Vector networks require explicit elementwise and coupling conventions.

## What is liquid?
The state's response speed depends on the input and state through the gate. This does **not** mean the weights $\theta$ are automatically updated at inference. Continuous dynamics, recurrent state adaptation and parameter learning are three different notions.

```python
# A single illustrative Euler step, not a recommended universal solver.
dx = -(1 / tau + gate(x, u)) * x + gate(x, u) * A
x_next = x + dt * dx
```

Numerical solver choice and step size affect stability and cost. An adaptive gate does not remove the need for numerical analysis. Published results use a specific model and solver configuration; the scalar equation above is explanatory.

## Relation to other sequence models
Like an SSM, an LTC network maintains a recurrent state, but its continuous-time nonlinear gating is a particular dynamical construction. Like a spiking model, it treats time explicitly, but its state need not communicate through discrete spikes. Neither relationship makes the architectures equivalent. [2](#ref-mamba) [3](#ref-spikes)

## Research questions
Evaluate irregularly sampled trajectories and distribution shifts in input timing. Compare interpolation, numerical solver cost, stability and task error. Specify whether learning is offline or online. Do not infer general continual-learning ability from an adaptive time constant alone.

## State the gate assumptions
For the scalar response lab, assume normalized time, $\tau=1$, $A=1$ and the bounded nonnegative gate $f(x,I)=2\,\operatorname{sigmoid}(I+ax)$. The baseline inverse time constant $1/\tau$ and $f$ both have inverse-time units; $I+ax$ is dimensionless. The gate is strictly between zero and two, and the decay coefficient $1/\tau+f$ stays positive. These are explicit synthetic assumptions; vector LTC constructions require their own coupling and bounds. Crucially, current state feeds both the direct leak term and the gate. Setting $a=2$ exposes this feedback; setting $a=0$ freezes the gate for a constant input. [1](#ref-liquid)

## Frozen coefficients reveal two effects
Hold the gate at a constant $f_0\ge0$ and let $\lambda=1/\tau+f_0$. The resulting linear ODE has

$$
\tau_{\mathrm{eff}}=\lambda^{-1},\qquad
x_\infty=\frac{f_0 A}{\lambda},\qquad
x(t)=x_\infty+[x(0)-x_\infty]e^{-\lambda t}.
$$

With $\tau=A=1,I=0,a=0$, the gate equals one: $\tau_{\mathrm{eff}}=0.5$, $x_\infty=0.5$ and, from $x(0)=0$, $x(t)=0.5(1-e^{-2t})$. Increasing a positive gate shortens the response time and shifts the equilibrium toward $A$; it changes more than speed. These formulas describe frozen coefficients. When $f$ depends on $x$, the displayed initial $\tau_{\mathrm{eff}}$ and $x_\infty$ are local interpretations, not the full nonlinear trajectory's fixed global constants.

## Compare the numerical response
The figure compares Euler samples with the exact frozen-gate solution. In the example above, an Euler step of $\Delta t=0.2$ gives $x_1=0.2$, while the exact value at that time is about $0.16484$. With frozen coefficients, Euler error around equilibrium is multiplied by $1-\Delta t\lambda$ each step; stability requires $|1-\Delta t\lambda|<1$. Large steps can overshoot or fail this condition despite a stable continuous ODE.

For state-dependent gating, the solid reference instead uses numerical fourth-order Runge–Kutta with 100 substeps per displayed Euler step; it is labeled numerical, not exact. The orange dashed curve keeps the initial gate fixed to expose the difference in assumptions. The terminal error compares Euler with the appropriate reference, and changing input, state feedback or step size recomputes the response. This is a solver illustration, not an LTC benchmark or a guarantee of accuracy for another coupled system.
