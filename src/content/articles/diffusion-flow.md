## Continuous denoising
A familiar Gaussian corruption is

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
$$

$x_0$ is data, $t$ a noise level and $\bar\alpha_t\in[0,1]$ the signal-retention schedule. A network may predict noise, clean data or another equivalent parameterization under stated assumptions. Sampling involves a chosen reverse-time solver. The denoiser architecture need not be a Transformer. [1](#ref-ddpm)

## Discrete language diffusion
LLaDA instead uses a forward masking process and a Transformer trained to predict masked tokens. Iterative generation can fill positions in an order unlike left-to-right autoregression. Changing the objective and decoding procedure does not mean the attention backbone has disappeared. [2](#ref-llada)

The interactive figure shows a **deterministic illustrative reveal schedule**, not samples from a trained diffusion model. Move the step control to compare left-to-right reveal with a mask-based order. A real diffusion sampler must specify confidence, remasking, conditioning and number of network evaluations.

## Continuous flow matching
For noise $x_0$ and data $x_1$, a linear interpolant is $x_t=(1-t)x_0+tx_1$. A simple flow-matching objective is

$$
\mathcal L_{\mathrm{FM}}=\mathbb E_{t,x_0,x_1}\left\|v_\theta(x_t,t)-(x_1-x_0)\right\|^2.
$$

$v_\theta$ predicts a velocity field. Sampling integrates $dx/dt=v_\theta(x,t)$. The conditional straight-line target does not imply that every learned marginal trajectory is exactly straight or that one Euler step is always accurate. [3](#ref-flow)

```python
x_t = (1 - t) * noise + t * data
loss = mean_squared_error(velocity(x_t, t), data - noise)
```

## Evidence and limits
Compare generation quality against **network evaluations, total FLOPs, latency and output length**, not only the number of decoding rounds. A bidirectional denoiser can expose more context per iteration, but repeated full-sequence passes can be costly. Variable lengths and KV reuse require method-specific treatment.
