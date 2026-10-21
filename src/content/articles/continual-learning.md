## A nonstationary training sequence
In continual learning, a model encounters changing data or tasks rather than one fixed shuffled dataset. Updating shared parameters for new data can reduce performance on previous data. Define an evaluation matrix $A_{t,j}$: performance on task $j$ after training through stage $t$. Forgetting should be computed from this history, with task boundaries and access to old data stated explicitly.

## Consolidate important parameters
Elastic weight consolidation (EWC) penalizes movement in parameters estimated to matter for a previous task:

$$
\mathcal L(\theta)=\mathcal L_{\mathrm{new}}(\theta)
+\frac\lambda2\sum_k F_k(\theta_k-\theta_k^*)^2.
$$

$\theta^*$ is a previous solution, $F_k$ a diagonal importance estimate related to the Fisher information, and $\lambda$ a strength. This approximation is not a complete posterior over all parameters and may constrain later learning. [1](#ref-ewc)

## Rehearse previous experience
Replay mixes stored or generated past examples into later training. The buffer's selection, size and sampling policy matter. Rolnick and collaborators studied replay for continual learning without task-boundary labels in their reinforcement-learning setting. That result does not eliminate the challenges of large-scale language-model updating. [2](#ref-replay)

```python
for fresh_batch in stream:
    old_batch = replay_buffer.sample()
    train_step(concatenate(fresh_batch, old_batch))
    replay_buffer.add(fresh_batch)
```

## Isolate changes
Adapters or low-rank modules can isolate some task-specific parameters. They can reduce interference in a frozen backbone, but routing, capacity growth and task selection remain problems. LoRA itself is an adaptation parameterization, not a guarantee of continual learning. [3](#ref-lora)

## Evaluation
Measure forward transfer, backward transfer, forgetting and new-task learning under a stated memory/computation budget. State whether task IDs are provided at inference. Keep a joint-training baseline where feasible. A model that stores one complete checkpoint per task solves a different resource problem than a fixed-capacity continual learner.

## Limits
A growing context window is not persistent parameter learning. A recurrent state that resets between sessions is not accumulated knowledge. A continually updated retrieval index is a memory-system change, distinct from weight consolidation. Continue to [memory and test-time adaptation](#/article/memory-adaptation).
