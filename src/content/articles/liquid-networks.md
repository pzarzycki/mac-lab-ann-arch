## Correct attribution
Liquid Time-constant Networks are associated with **Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus and Radu Grosu**. The cited paper was first posted in 2020 and accepted at AAAI 2021. Demis Hassabis is not an author of this work. [1](#ref-liquid)

## A changing effective time scale
A schematic LTC neuron follows

$$
\frac{dx(t)}{dt}=-\left[\frac1\tau+f(x(t),I(t),t;\theta)\right]x(t)+f(x(t),I(t),t;\theta)A.
$$

$x(t)$ is a state variable, $I(t)$ input, $\tau>0$ a baseline time constant, $A$ a driving level and $f$ a learned nonlinear gate. The effective time constant is $\tau_{\mathrm{eff}}=\tau/(1+\tau f)$ under this scalar form and appropriate gate assumptions. Vector networks require explicit elementwise and coupling conventions.

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
