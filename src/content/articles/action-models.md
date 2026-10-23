## Discretized actions
A continuous action coordinate can be quantized to a finite set of bins and predicted as a token. This connects robot supervision to a language-model output interface, as studied in RT-2. Quantization resolution, coordinate order and autoregressive dependencies affect precision and latency. [1](#ref-rt2)

For a deliberately coarse example, eight bins span $[-10,10]$ mm with width $w=2.5$ mm. Encode $x$ by $b=\operatorname{clip}(\lfloor(x+10)/w\rfloor,0,7)$ and decode $\widehat x=-10+(b+\tfrac12)w$. Thus $x=2$ mm becomes bin $4$ and reconstructs as $1.25$ mm. For in-range inputs the bin-center error is at most $w/2=1.25$ mm; clipping out-of-range commands has no such bound. These are toy units and eight bins; RT-2 uses 256 for continuous action coordinates. A serialized action vector also fixes coordinate order, so “one action” can require several autoregressive token predictions. [1](#ref-rt2)

## Continuous action distributions
Diffusion Policy models an action sequence through conditional denoising, supporting multimodal action distributions rather than collapsing incompatible actions into their mean. π₀ uses a flow-matching action architecture on top of a VLM. These share iterative generative computation but are not identical objectives or samplers. [2](#ref-diffusion-policy) [3](#ref-pi0)

For a simple conditional flow objective, let $a_0\sim\mathcal N(0,I)$ be noise with shape $H\times d_a$, $(c,a_1)$ a context/demonstrated-chunk pair sampled from the training set, and $\tau\sim\mathcal U(0,1)$ independent of the noise and training pair. The $H$ rows are future physical action steps; the $d_a$ columns are coordinates in a specified convention. Then

$$
a_\tau=(1-\tau)a_0+\tau a_1,\qquad
\mathcal L=\mathbb E\|v_\theta(a_\tau,\tau,c)-(a_1-a_0)\|^2.
$$

$\tau\in[0,1]$ is generative time, not wall-clock robot time. The action horizon is a separate axis. The formula illustrates conditional flow matching; each model specifies its own parameterization and conditioning. [4](#ref-flow)

At inference, the demonstrated endpoint $a_1$ is **unavailable**. Sample an initial noise chunk and solve

$$
\frac{da}{d\tau}=v_\theta(a,\tau,c),\qquad
a_{k+1}=a_k+\Delta\tau\,v_\theta(a_k,\tau_k,c).
$$

The second expression is explicit Euler, not an exact solver in general. Conditioning $c$ remains fixed during these sampling iterations, then changes when a new observation arrives. In this parameterization a flow network predicts velocity; a noise-parameterized diffusion network predicts the noise in a scheduled corrupted chunk and uses it in a reverse denoising update. Both can update all chunk entries in one network evaluation; neither fact makes the two objectives identical. [2](#ref-diffusion-policy) [4](#ref-flow)

The figure isolates these operations with an analytic **toy** conditional center $\mu(c)$ and a fixed seed. Its flow uses $v=\mu(c)-a$; at $N=4$, the first coordinate with seed $-6$ mm and center $2$ mm ends at $2+(-6-2)(1-1/4)^4=-0.53125$ mm. Its synthetic denoiser predicts the known center exactly, so its zero-variance reverse trace reaches that center. This is not a quality comparison: the two hand-defined functions have different finite-time behavior, and real learned predictions have error. The token branch makes quantization error visible instead. Each branch prints the resulting $4\times2$ chunk.

## Execute a prefix, then observe again
Predict a horizon $H$, execute only $h\le H$ actions and replan with a new observation. This receding-horizon policy can react more often than executing the entire plan. Smaller $h$ increases inference demand; larger $h$ increases open-loop exposure.

With the figure's $H=4$, $h=2$ and $20$ ms action period, two rows execute over $40$ ms before sensing again; the remaining two are proposals rather than committed commands. Sampling latency is excluded from that physical-time strip and must be added to a real schedule. Environment feedback refreshes **both** the next camera image and the next proprioceptive state, then synchronizes them before the next policy call.

```python
while task_active:
    chunk = sample_actions(policy, recent_observation)
    execute(chunk[:h])
    recent_observation = observe()
```

## What to measure
Report control frequency, inference latency distribution, chunk length, executed prefix, action units and controller behavior. Compare task success under disturbances, not only offline action reconstruction. Inference steps are not free; use matched latency or total-compute comparisons when contrasting token and flow policies.

## Limits
A generative model can represent several plausible action sequences yet still sample an unsafe one. Contact physics, calibration and feedback determine execution. A low-level controller and explicit constraints remain part of the system being evaluated.
