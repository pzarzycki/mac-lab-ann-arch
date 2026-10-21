## The unit of evaluation is a rollout
Offline prediction error does not fully characterize a policy that changes its own observations through action. Small errors can move a robot into unfamiliar states. Evaluate repeated closed-loop rollouts under a specified initial-state distribution and success criterion. The VLA and Diffusion Policy papers illustrate why task and hardware definitions matter. [1](#ref-openvla) [2](#ref-diffusion-policy)

## A structured test matrix
| Axis | Controlled variation | What to record |
| --- | --- | --- |
| Perception | Lighting, occlusion, object identity | Failures attributable to observation |
| Language | Paraphrases and ambiguous references | Grounding errors |
| Dynamics | Object mass, friction, disturbances | Recovery and stability |
| Timing | Sensor age and inference delay | Missed deadlines, stale actions |
| Embodiment | Calibration and action convention | Transfer requirements |

Do not call a new object, a new instruction and a new robot the same kind of out-of-distribution test.

## Timing belongs in the specification
A simplified observation-to-action delay is

$$
\Delta=t_{\mathrm{capture}}+t_{\mathrm{transport}}+t_{\mathrm{inference}}+t_{\mathrm{dispatch}}.
$$

This additive expression assumes sequential stages; pipelined systems need a timing diagram with overlaps. At speed $v$, a simple constant-velocity displacement over delay is $v\Delta$. This is a dimensional illustration, not a sufficient collision or stability model.

## Demonstrations and interventions
State whether trajectories include failures, recovery, human takeovers and multiple embodiments. Report how often an operator intervenes and how interventions affect the success metric. π₀'s multi-platform setting is a concrete reference for the data and adaptation questions, not a universal benchmark standard. [3](#ref-pi0)

```python
record = {
    'observation_timestamp': sensor_time,
    'action_timestamp': execution_time,
    'intervention': intervention_occurred,
    'success': independently_scored_success,
}
```

## A reproducible report
Provide seeds or initial-state identifiers, number of trials, confidence intervals, controller version, model checkpoint, calibration and termination rules. Keep development episodes separate from final evaluation. Treat task success, safety violations and latency as separate outcomes instead of hiding them in one score.

## Limits
This atlas provides architecture and experiment guidance. Its diagrams do not run a robot controller, validate a deployment or establish real-world safety. Evaluate the whole feedback loop rather than infer physical competence from language fluency.
