## Continuous denoising
A familiar Gaussian corruption is

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
$$

$x_0$ is data, $t$ a noise level and $\bar\alpha_t\in[0,1]$ the signal-retention schedule. A network may predict noise, clean data or another equivalent parameterization under stated assumptions. Sampling involves a chosen reverse-time solver. The denoiser architecture need not be a Transformer. [1](#ref-ddpm)

One concrete training choice samples a data vector $x_0\in\mathbb R^d$, Gaussian noise $\epsilon\in\mathbb R^d$ and a discrete schedule index $t$ uniformly from $1,\ldots,T$, then minimizes

$$
\mathcal L_\epsilon=\mathbb E_{x_0,t,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2.
$$

$c$ is optional conditioning and is omitted for unconditional generation. Noise is known during training because we sampled it; at inference the network must estimate it. A simplified unweighted noise loss should not be confused with every weighting of a variational objective. [1](#ref-ddpm)

For a concrete **deterministic DDIM-style step**, choose a less noisy index $s<t$ with $\bar\alpha_s>\bar\alpha_t$:

$$
\widehat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\widehat\epsilon}{\sqrt{\bar\alpha_t}},\qquad
x_s=\sqrt{\bar\alpha_s}\widehat x_0+\sqrt{1-\bar\alpha_s}\widehat\epsilon.
$$

Here $\widehat\epsilon=\epsilon_\theta(x_t,t,c)$. This zero-variance choice differs from an ancestral DDPM step, which also specifies reverse variance and a random draw. It is a solver choice, not a new training target. [4](#ref-ddim)

The figure's scalar training pair has $x_0=2$, $\epsilon=1$ and $\bar\alpha_t=0.25$, giving $x_t=1.866$. With $\widehat\epsilon=1$, the clean estimate is exactly $2$; at $\bar\alpha_s=0.64$, $x_s=0.8(2)+0.6(1)=2.2$. “Less noisy” refers to the schedule, not a guarantee that each scalar value approaches zero or changes monotonically. Change the estimated noise to trace its effect on both loss and sampling. The true $x_0$ shown for checking this arithmetic is unavailable to a real sampler.

## Discrete language diffusion
LLaDA instead uses a forward masking process and a Transformer trained to predict masked tokens. Iterative generation can fill positions in an order unlike left-to-right autoregression. Changing the objective and decoding procedure does not mean the attention backbone has disappeared. [2](#ref-llada)

For a sequence $y=(y_1,\ldots,y_L)$, sample $t\sim\mathcal U(0,1)$ and independently replace each response token with a mask with probability $t$, producing $y_t$. A masked-token objective used by LLaDA has the form

$$
\mathcal L_{\mathrm{mask}}=-\mathbb E_{y,t,y_t}\left[\frac1t\sum_{i:(y_t)_i=\mathrm{MASK}}\log p_\theta(y_i\mid y_t,c)\right].
$$

Only masked response positions contribute; an unmasked prompt $c$ supplies conditioning in the conditional setting. The model sees the current partly observed sequence and predicts distributions for masked positions. One forward pass can predict many positions; deciding which predictions become conditioning tokens for the next pass is a separate sampling operation. [2](#ref-llada)

The new masked view prints its **synthetic confidence rule**: score each proposal, accept the top two, and optionally reopen one old token on round 2. Its provisional “box” can be replaced by “cup” before acceptance; confidence is arithmetic, not a learned probability. Remasking an accepted position changes the following network input. The retained supporting reveal view compares a fixed mask order with left-to-right order. Neither view is a sample from a trained language model.

## Continuous flow matching
Use $z\sim\mathcal N(0,I_d)$ for noise and $y\sim p_{\mathrm{data}}$ for data, sampled independently as a simple endpoint coupling, and $t\sim\mathcal U(0,1)$. Both endpoints have dimension $d$. A linear interpolant is $x_t=(1-t)z+ty$. A simple flow-matching objective is

$$
\mathcal L_{\mathrm{FM}}=\mathbb E_{t,z,y}\left\|v_\theta(x_t,t)-(y-z)\right\|^2.
$$

$v_\theta$ predicts a velocity field. Sampling starts at noise at $t=0$ and integrates $dx/dt=v_\theta(x,t)$ toward data at $t=1$. This time direction is opposite to the Gaussian corruption index above, where increasing $t$ means adding noise. Naming endpoints $z,y$ avoids calling both clean data and noise “$x_0$.” Other couplings are possible. The conditional straight-line target does not imply that every learned marginal trajectory is exactly straight or that one Euler step is always accurate. [3](#ref-flow)

```python
x_t = (1 - t) * noise + t * data
loss = mean_squared_error(velocity(x_t, t), data - noise)
```

The independent solver illustration uses the fully specified field $v(x,y,t)=(1,x)$, starting at $(0,0)$. The exact endpoint is $(1,0.5)$; explicit Euler with $N$ equal steps gives $(1,(N-1)/(2N))$. At $N=4$ it reaches $(1,0.375)$. This curved benchmark isolates integration error; it is not a fitted flow model or a claim that the displayed training pair learns this field.

| Construction | Learned prediction | One inference operation |
| --- | --- | --- |
| Gaussian example | Noise vector, same shape as sample | Estimate clean data, then move to a less noisy level |
| Masked language | Token probabilities at masked positions | Predict proposals, then accept/remask under a schedule |
| Flow example | Velocity vector, same shape as sample | Integrate $x\leftarrow x+\Delta t\,v_\theta(x,t)$ |

## Evidence and limits
Compare generation quality against **network evaluations, total FLOPs, latency and output length**, not only the number of decoding rounds. A bidirectional denoiser can expose more context per iteration, but repeated full-sequence passes can be costly. Variable lengths and KV reuse require method-specific treatment.
