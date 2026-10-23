## From observations to actions
A VLA policy conditions actions on visual observations, language and often robot state:

$$
\pi_\theta(a_{t:t+H-1}\mid o_{\le t},\ell,q_t).
$$

$o$ denotes observations, $\ell$ a language instruction, $q_t$ proprioceptive state and $H$ an action horizon. Not every published VLA uses every conditioning variable or predicts a chunk; this notation identifies the design dimensions rather than imposing one architecture.

## Two foundational examples
RT-2 studies co-fine-tuning vision-language models with robot trajectories and web-derived vision-language tasks, representing actions through a token interface. OpenVLA provides an open 7B model built from a language backbone and visual representations, with a recipe for robot adaptation. Their reported success rates belong to their own task and embodiment evaluations. [1](#ref-rt2) [2](#ref-openvla)

RT-2's action interface includes a termination command, three end-effector translation displacements, three rotation displacements and gripper extension. Continuous coordinates use 256 bins; valid action tokens are selected and converted back to commands. OpenVLA also bins each action dimension into 256 values, using dimension-specific training-data 1st/99th quantiles to set its interval; it applies next-token cross-entropy to action tokens. This action supervision supplies the missing connection from visual-language features to motor labels. [1](#ref-rt2) [2](#ref-openvla)

For action tokens $b_1,\ldots,b_K$ and observation/instruction context $c$, a schematic imitation loss is $-\sum_j\log p_\theta(b_j\mid c,b_{<j})$. A **toy** translation coordinate makes the representation change explicit: eight uniform bins on $[-10,10]$ mm have width $2.5$ mm. A $+2$ mm demonstration maps to $b=\lfloor(2+10)/2.5\rfloor=4$; decoding that bin's center gives $-10+(4+\tfrac12)2.5=1.25$ mm. Edge values are clipped to bins $0,\ldots,7$. Eight bins, this unit interval and bin-center reconstruction are illustration choices, not a published robot calibration.

## Representation is not control
A visual-language backbone can transfer object and instruction knowledge. It still needs action supervision and compatible actuation semantics. Joint positions, joint deltas, end-effector pose deltas, forces and gripper commands have different units, constraints and controllers. Normalizing an action vector is not sufficient to make robots interchangeable.

The vector figure follows visual features and language through a policy, an action representation, execution and a new observation. Its environmental return branches into **camera acquisition and proprioception**, both indexed $t+1$, then synchronizes them before the next policy call. The instruction remains persistent task conditioning. Toggle action tokens, noise prediction or velocity flow to expose the actual representation computation. It is a generic interface diagram, not an exact reproduction of RT-2, OpenVLA or π₀; its two-coordinate action assumes a compatible low-level controller.

## Continuous action experts
π₀ builds on a pretrained VLM with a flow-matching action architecture. This bridges a semantic representation to continuous action chunks, rather than requiring each action coordinate to be generated as ordinary text. [3](#ref-pi0)

An action chunk $A_t\in\mathbb R^{H\times d_a}$ contains $H$ future action vectors, each with $d_a$ coordinates. The figure uses $H=4,d_a=2$ with displacement coordinates in mm. It marks the executed prefix $h$ and discards the remaining proposed actions when replanning. Generative time $\tau$ labels iterations that construct the chunk; physical time labels when its rows execute. Changing sampler iterations is therefore different from changing how long the robot acts before refreshing its sensors.

| Interface | Conditioning at the abstraction used here | Output and horizon |
| --- | --- | --- |
| RT-2 token policy | Camera image and task instruction | Serialized coordinates of a robot action; not the figure's four-row chunk |
| Original OpenVLA token policy | Image and language context | Discretized dimensions of one action; conventions depend on embodiment |
| π₀ continuous expert | Visual/language context and robot state | Continuous action chunk; execution prefix is a control-system choice |

These rows identify representation boundaries rather than matching observation layouts, control rates or horizons across papers. [1](#ref-rt2) [2](#ref-openvla) [3](#ref-pi0)

```python
observation = synchronize(cameras, robot_state)
action_chunk = policy(observation, instruction)
controller.execute(action_chunk[:execution_horizon])
```

## Limits and research questions
Generalization across objects, language, scenes and embodiments are different axes. Report dataset overlap, action normalization, fine-tuning data and intervention policy. A plausible instruction-conditioned action is not automatically feasible under contact, latency or actuator limits. Read [action representations](#/article/action-models) and [closed-loop evaluation](#/article/robot-evaluation).
