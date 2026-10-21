## From observations to actions
A VLA policy conditions actions on visual observations, language and often robot state:

$$
\pi_\theta(a_{t:t+H-1}\mid o_{\le t},\ell,q_t).
$$

$o$ denotes observations, $\ell$ a language instruction, $q_t$ proprioceptive state and $H$ an action horizon. Not every published VLA uses every conditioning variable or predicts a chunk; this notation identifies the design dimensions rather than imposing one architecture.

## Two foundational examples
RT-2 studies co-fine-tuning vision-language models with robot trajectories and web-derived vision-language tasks, representing actions through a token interface. OpenVLA provides an open 7B model built from a language backbone and visual representations, with a recipe for robot adaptation. Their reported success rates belong to their own task and embodiment evaluations. [1](#ref-rt2) [2](#ref-openvla)

## Representation is not control
A visual-language backbone can transfer object and instruction knowledge. It still needs action supervision and compatible actuation semantics. Joint positions, joint deltas, end-effector pose deltas, forces and gripper commands have different units, constraints and controllers. Normalizing an action vector is not sufficient to make robots interchangeable.

The vector figure follows visual features and language through a policy, an action representation, a low-level controller and a new observation. Toggle discrete or continuous actions to locate the changing component. It is a generic interface diagram, not an exact reproduction of RT-2, OpenVLA or π₀.

## Continuous action experts
π₀ builds on a pretrained VLM with a flow-matching action architecture. This bridges a semantic representation to continuous action chunks, rather than requiring each action coordinate to be generated as ordinary text. [3](#ref-pi0)

```python
observation = synchronize(cameras, robot_state)
action_chunk = policy(observation, instruction)
controller.execute(action_chunk[:execution_horizon])
```

## Limits and research questions
Generalization across objects, language, scenes and embodiments are different axes. Report dataset overlap, action normalization, fine-tuning data and intervention policy. A plausible instruction-conditioned action is not automatically feasible under contact, latency or actuator limits. Read [action representations](#/article/action-models) and [closed-loop evaluation](#/article/robot-evaluation).
