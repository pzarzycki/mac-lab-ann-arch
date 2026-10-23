## A dynamical neuron
A simple discrete-time leaky integrate-and-fire model uses

$$
u_{t+1}=\beta u_t+I_t-\vartheta z_t,\qquad
z_t=\mathbf1[u_t\ge\vartheta].
$$

$u_t$ is membrane potential, $I_t$ input drive, $0\le\beta<1$ a leak factor, $\vartheta$ threshold and $z_t$ a binary spike. This is one reset convention; other models reset before or after integration or clamp to a reset potential. State and timing matter beyond an ordinary static activation.

## The gradient problem
A hard threshold has zero derivative almost everywhere and an undefined derivative at its threshold. Surrogate-gradient methods keep the forward spike but use a continuous or smooth proxy derivative during learning. Backpropagation through time can still assign credit through the resulting temporal computation. Thus “spiking” does not imply “trained without backpropagation.” [1](#ref-spikes)

```python
spike = (membrane >= threshold).float()    # hard forward event
# In a custom backward rule, substitute a bounded continuous proxy derivative.
# Temporal learning may still require an unrolled computation graph.
```

## Biological and hardware axes
An event-based representation resembles one aspect of biological communication. Weight symmetry, global error objectives and unrolled gradient storage can remain biologically implausible. Feedback alignment addresses a different axis: what feedback weights carry credit. [2](#ref-feedback)

Potential energy efficiency depends on hardware, event sparsity, temporal resolution, communication and the task. Simulating many dense timesteps on a general GPU need not save energy compared with an ordinary ANN. State the measured hardware and power boundary before claiming a benefit.

## Experiment
Use an event-timing task where the timing genuinely carries information. Compare against a rate-coded baseline and a conventional recurrent model. Report latency, number of timesteps, accuracy and energy measured under comparable input/output boundaries. Compare surrogate shape and spike regularization separately.

## Limits
A sparse spike raster does not reveal whether the learning rule is local. Conversely, a local plasticity rule does not guarantee that a network learns deep supervised tasks effectively. Keep neural dynamics, credit assignment and hardware implementation as distinct comparison axes.

## Follow threshold crossing and reset timing
The lab computes $u_0=0$ and, at each integer $t$, first tests $z_t=\mathbf1[u_t\ge\vartheta]$, then performs the stated update. Its aligned input, membrane and spike panels use the same timestep axis. The membrane curve shows the value **before** the subtractive reset in that update; a threshold crossing appears as a spike at that sample, and the reset affects the following membrane sample. There is at most one binary spike per step and no refractory period in this toy.

For example, with $\beta=0.5,I_t=1,\vartheta=1$, $u_1=1$ emits $z_1=1$. The next update gives $u_2=0.5\cdot1+1-1=0.5$, so $z_2=0$. Then $u_3=1.25$ emits another spike. Constant and pulsed inputs, leak, threshold and the inspected timestep all change the calculated trace. A coarse discrete sample can overshoot threshold; it is not an exact continuous-time event solver.

## Make the backward substitution explicit
One illustrative surrogate derivative is the triangular window

$$
\widetilde{\frac{\partial z}{\partial u}}=
\gamma\max\left(0,1-\frac{|u-\vartheta|}{w}\right),\qquad w>0,\ \gamma\ge0.
$$

$w$ controls the membrane interval around threshold that receives gradient; $\gamma$ controls its peak. This is an illustrative member of the surrogate-gradient design space, not a claim that all methods use this shape. It is **not the true derivative** of the hard spike. Changing width at fixed peak also changes total gradient area, so comparing shapes requires stating both parameters. The forward event remains binary. [1](#ref-spikes)

With upstream derivative $\partial L/\partial z=2$, $u-\vartheta=0.1$, $w=0.5$ and $\gamma=1$, the local spike branch supplies $\partial L/\partial u\approx2(1-0.1/0.5)=1.6$. The figure evaluates this contribution at the selected membrane sample. Full temporal backpropagation also includes the membrane's direct recurrent path and whichever treatment of reset gradients the implementation adopts. A useful biased backward rule is not an exact gradient of the hard-threshold forward model.
