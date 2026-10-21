## Discretized actions
A continuous action coordinate can be quantized to a finite set of bins and predicted as a token. This connects robot supervision to a language-model output interface, as studied in RT-2. Quantization resolution, coordinate order and autoregressive dependencies affect precision and latency. [1](#ref-rt2)

## Continuous action distributions
Diffusion Policy models an action sequence through conditional denoising, supporting multimodal action distributions rather than collapsing incompatible actions into their mean. π₀ uses a flow-matching action architecture on top of a VLM. These share iterative generative computation but are not identical objectives or samplers. [2](#ref-diffusion-policy) [3](#ref-pi0)

For a simple conditional flow objective, let $a_0$ be noise, $a_1$ a demonstrated action chunk and $c$ observation/instruction context:

$$
a_\tau=(1-\tau)a_0+\tau a_1,\qquad
\mathcal L=\mathbb E\|v_\theta(a_\tau,\tau,c)-(a_1-a_0)\|^2.
$$

$\tau\in[0,1]$ is generative time, not wall-clock robot time. The action horizon is a separate axis. The formula illustrates conditional flow matching; each model specifies its own parameterization and conditioning. [4](#ref-flow)

## Execute a prefix, then observe again
Predict a horizon $H$, execute only $h\le H$ actions and replan with a new observation. This receding-horizon policy can react more often than executing the entire plan. Smaller $h$ increases inference demand; larger $h$ increases open-loop exposure.

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
