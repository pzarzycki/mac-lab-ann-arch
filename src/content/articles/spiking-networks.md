## A dynamical neuron
A simple discrete-time leaky integrate-and-fire model uses

$$
u_{t+1}=\beta u_t+I_t-\vartheta z_t,\qquad
z_t=\mathbf1[u_t\ge\vartheta].
$$

$u_t$ is membrane potential, $I_t$ input drive, $0\le\beta<1$ a leak factor, $\vartheta$ threshold and $z_t$ a binary spike. This is one reset convention; other models reset before or after integration or clamp to a reset potential. State and timing matter beyond an ordinary static activation.

## The gradient problem
A hard threshold has zero derivative almost everywhere and an undefined derivative at its threshold. Surrogate-gradient methods keep the forward spike but use a smooth proxy derivative during learning. Backpropagation through time can still assign credit through the resulting temporal computation. Thus “spiking” does not imply “trained without backpropagation.” [1](#ref-spikes)

```python
spike = (membrane >= threshold).float()    # hard forward event
# In a custom backward rule, substitute a bounded smooth derivative.
# Temporal learning may still require an unrolled computation graph.
```

## Biological and hardware axes
An event-based representation resembles one aspect of biological communication. Weight symmetry, global error objectives and unrolled gradient storage can remain biologically implausible. Feedback alignment addresses a different axis: what feedback weights carry credit. [2](#ref-feedback)

Potential energy efficiency depends on hardware, event sparsity, temporal resolution, communication and the task. Simulating many dense timesteps on a general GPU need not save energy compared with an ordinary ANN. State the measured hardware and power boundary before claiming a benefit.

## Experiment
Use an event-timing task where the timing genuinely carries information. Compare against a rate-coded baseline and a conventional recurrent model. Report latency, number of timesteps, accuracy and energy measured under comparable input/output boundaries. Compare surrogate shape and spike regularization separately.

## Limits
A sparse spike raster does not reveal whether the learning rule is local. Conversely, a local plasticity rule does not guarantee that a network learns deep supervised tasks effectively. Keep neural dynamics, credit assignment and hardware implementation as distinct comparison axes.
