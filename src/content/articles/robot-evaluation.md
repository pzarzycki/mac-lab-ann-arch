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

The timing figure specifies two synthetic cycles rather than assuming pipelining always reduces delay. Capture takes $12$ ms, transport $8$ ms, inference $40$ ms and dispatch $5$ ms; capture timestamps mark exposure start. Frame 0's first action uses an observation aged $65$ ms. Two executed actions at $20$ ms each finish at $105$ ms. In the sequential schedule, frame 1 starts then and is used at $170$ ms, again aged $65$ ms. In the pipelined schedule it starts at $25$ ms, waits for the single inference server until $60$ ms, and is used at $105$ ms: age $80$ ms. Thus faster throughput can coexist with older observations. At $v=0.2$ m/s, $80$ ms corresponds to $0.016$ m, or $16$ mm, under the stated constant-velocity assumption.

## Demonstrations and interventions
State whether trajectories include failures, recovery, human takeovers and multiple embodiments. Report how often an operator intervenes and how interventions affect the success metric. π₀'s multi-platform setting is a concrete reference for the data and adaptation questions, not a universal benchmark standard. [3](#ref-pi0)

```python
record = {
    'trial_id': 'T02',
    'initial_state_id': 'S02',
    'observation_timestamp': sensor_time,
    'action_timestamp': execution_time,
    'intervention': intervention_occurred,
    'termination': 'goal_after_takeover',
    'success': independently_scored_success,
    'autonomous_success': False,
    'p95_observation_to_action_ms': 80,
}
```

## A small, fully specified aggregation
Here is an **invented protocol**, motivated by the task-dependent success rules and repeated evaluations in OpenVLA, rather than a reproduction of its scores: move a block into a marked tray, release it, and confirm that it remains wholly inside for $2$ seconds. Use a fixed camera, a versioned controller/checkpoint/calibration and three recorded initial poses S01–S03, replayed for each compared policy in randomized order. An episode ends at confirmed completion, a $30$-second timeout or an operator stop. An operator takeover ends autonomous evaluation; a continued assisted completion is recorded separately. A missed completion declaration remains a timeout. No trial is dropped and no partial credit is used. The same rules would need more trials and more initial states for a useful estimate. [1](#ref-openvla)

| Trial / initial state | Termination and intervention | Autonomous success | Completion including assistance | p95 observation-to-action delay |
| --- | --- | --- | --- | --- |
| T01 / S01 | Goal confirmed; no takeover | 1 | 1 | 62 ms |
| T02 / S02 | Goal after human takeover | 0 | 1 | 80 ms |
| T03 / S03 | Timeout; no takeover | 0 | 0 | 73 ms |

Autonomous success is $1/3=33.3\%$, assisted-inclusive completion is $2/3=66.7\%$, and takeover rate is $1/3$. The denominator remains all three attempted rollouts. A failed or early termination is not removed; partial completion does not count toward this binary criterion. Latency is a separate per-rollout outcome: averaging the three reported p95 values would **not** recover a pooled p95. Pool timestamp records or state the aggregation you actually use.

For the figure's approximate $95\%$ Wilson interval, $n$ is the attempted-trial count, $\widehat p=k/n$, and $z=1.96$:

$$
\frac{\widehat p+z^2/(2n)\;\pm\;z\sqrt{\widehat p(1-\widehat p)/n+z^2/(4n^2)}}{1+z^2/n}.
$$

For one success in three trials this is approximately $[6.1\%,79.2\%]$: the tiny cohort is visibly inconclusive. This binomial calculation assumes independent trials with a common success probability; repeated correlated starts or a heterogeneous task mixture need a reporting/uncertainty scheme that respects that design. The displayed interval is our statistical illustration, not an OpenVLA confidence interval.

## A reproducible report
Provide seeds or initial-state identifiers, number of trials, confidence intervals, controller version, model checkpoint, calibration and termination rules. Keep development episodes separate from final evaluation. Treat task success, safety violations and latency as separate outcomes instead of hiding them in one score.

## Limits
This atlas provides architecture and experiment guidance. Its diagrams do not run a robot controller, validate a deployment or establish real-world safety. Evaluate the whole feedback loop rather than infer physical competence from language fluency.
