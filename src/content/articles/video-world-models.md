## What does a world model predict?
A useful operational definition is a learned model that predicts aspects of an environment's evolution. The predicted object may be pixels, latent features or other state variables. A visually convincing video generator and a useful action-conditioned predictor therefore require different evaluations. For control, ask whether changing a proposed action produces a useful prediction of its consequences. A model can represent video well without already providing that interface.

V-JEPA learns masked video features; V-JEPA 2 develops larger-scale visual representations. V-JEPA 2-AC adds action conditioning on top of a frozen V-JEPA 2 encoder. Keep these names separate: action-free pretraining is not itself the robot controller. [1](#ref-vjepa) [2](#ref-vjepa2)

## Observation is not intervention
Action-free video can reveal regularities of movement and appearance. It does not supply an explicit response to a command the observing agent could choose. V-JEPA 2-AC is post-trained with robot interaction data containing visual observations and end-effector state signals. “Unlabeled” here excludes additional task, success and reward annotations; it does not mean that all information about robot movement is absent. [2](#ref-vjepa2)

The official repository separates video encoders from the action-conditioned backbone and provides an example for inspecting a learned energy landscape from a robot trajectory. It also exposes probe-based evaluations. The repository also lists V-JEPA 2.1, which revises representation training; the control example here specifically concerns V-JEPA 2-AC. Recognition, prediction and control should not be collapsed into one score. [3](#ref-vjepa2-code)

## Roll out candidate actions
The following notation illustrates an action-conditioned latent model; it is not the exact V-JEPA 2-AC tensor layout:

$$
z_t=f(o_{\leq t}),\quad
\widehat z_{t+k+1}=F_\psi(\widehat z_{t+k},a_{t+k},q_{t+k}),\quad
\widehat z_t=z_t.
$$

$o$ is an observation history, $z\in\mathbb R^d$ a simplified feature vector, $a\in\mathbb R^r$ an action and $q$ robot state. Real predictors can retain token arrays and longer histories. Future $q$ must be supplied by a specified transition convention or predicted; a planner cannot read future measured robot states before executing. The diagram omits this implementation detail to expose the difference between observational prediction and action-conditioned rollouts.

For an illustrative image-goal planner, encode $z_g=f(o_g)$ and solve

$$
A^*=\arg\min_{A\in\mathcal A^H}
D(\widehat z_{t+H}(A),z_g)+\lambda C(A).
$$

$A=(a_t,\ldots,a_{t+H-1})$ is a candidate action sequence, $D$ a feature discrepancy and $C$ an optional constraint/cost term. This is a generic planning objective. The illustrated candidate scores are not learned-model measurements or reproduced paper results.

The trace companion puts every candidate on **one common scalar-feature axis**, at the same predicted steps $k=0,1,2$. It sets $\lambda=0$ and $D(u,g)=|u-g|$, so the endpoint determines the score:

| Candidate | $\widehat z_t$ | $\widehat z_{t+1}$ | $\widehat z_{t+2}$ | Distance to goal $+34$ | Distance to goal $-35$ |
| --- | --- | --- | --- | --- | --- |
| A | 0 | −18 | −35 | $\lvert -35-34\rvert=69$ | $\lvert -35-(-35)\rvert=0$ |
| B | 0 | 10 | 34 | $\lvert 34-34\rvert=0$ | $\lvert 34-(-35)\rvert=69$ |
| C | 0 | 6 | 12 | $\lvert 12-34\rvert=22$ | $\lvert 12-(-35)\rvert=47$ |

These invented features are neither meters nor task-success probabilities. Candidate B wins at goal $+34$; A wins at goal $-35$. The slider also exposes intermediate goals, with alphabetical tie-breaking only as a deterministic display rule. Intermediate rollout values are visible even though this particular score uses only the endpoint. Choosing an endpoint-only loss is itself a modeling decision: a path can pass through an unacceptable state while finishing close to the goal. No such physical constraint is modeled by this scalar example.

## Close the loop
Execute a prefix of the selected sequence, observe again and replan. This allows new evidence to replace a potentially inaccurate imagined rollout. It also costs sensing and planning time. An action horizon, number of sampled candidates, executed prefix and observation delay are separate quantities; reducing one does not automatically improve all the others.

In the companion, $H=2$ proposed actions connect the three plotted feature times. The prefix control executes either $h=1$ or $h=2$ before a new measured observation replaces the imagined starting context. The feedback wire runs outside the chart so it cannot be mistaken for another candidate trajectory. Its abstract action labels do not specify an executable robot command or future state estimator; those remain necessary interfaces in a real planner. The expandable architecture figure retains the comparison with action-free video prediction.

The V-JEPA 2 work demonstrates image-goal robot planning, with limitations including camera sensitivity and long-horizon behavior. These experiments support particular environments and tasks, not arbitrary physical understanding. A feature-space goal can also omit a requirement that matters to a user. [2](#ref-vjepa2)

## Evaluate prediction and control separately
For an exploratory experiment, first compare one-step and multi-step prediction under held-out action sequences. Then evaluate closed-loop success, recovery, latency and interventions. Record camera geometry, action units, robot/controller configuration and goal-image construction. Include a simple control baseline and report candidate-search compute.

Our diagram deliberately shows an abstract candidate comparison rather than a photorealistic generated future: its purpose is to locate the decision boundary. A promising latent rollout still needs an executable action interface and feedback. Connect this topic to [JEPA](#/article/jepa), [action models](#/article/action-models), and [closed-loop evaluation](#/article/robot-evaluation).
