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

## Calculate retention from a task matrix
For one explicit higher-is-better convention, let $T$ be the final stage and define old-task forgetting and backward transfer by

$$
f_j=\max_{t=j,\ldots,T-1}A_{t,j}-A_{T,j},\qquad
\mathrm{BWT}=\frac1{T-1}\sum_{j<T}(A_{T,j}-A_{j,j}).
$$

This “previous best” definition allows negative forgetting if the final score exceeds every earlier score; some protocols include the final stage in the maximum to clamp forgetting at zero. Report the convention. Define forward transfer as $\mathrm{FWT}=\frac1{T-1}\sum_{j=2}^T(A_{j-1,j}-b_j)$, where $b_j$ is the untrained-model performance on the same task and metric. Forward transfer requires evaluation **before** learning each task; it cannot be recovered from only a final row.

The figure calculates its matrix from three invented quadratic tasks, not benchmark accuracies. Each uses $\ell_j(\theta)=\frac12\|\theta-v_j\|^2$ with $v_A=(1,0)$, $v_B=(-1,1)$, $v_C=(0,-1)$ and score $\exp(-\ell_j)$. Sequential exact fitting gives

| Training stage | Task A | Task B | Task C |
| --- | --- | --- | --- |
| After A | 1.000 | 0.082 | 0.368 |
| After B | 0.082 | 1.000 | 0.082 |
| After C | 0.368 | 0.082 | 1.000 |

Old-task forgetting is $(0.632,0.918)$, mean $0.775$; BWT is $-0.775$. For an untrained $\theta_0=(0,0)$, $b_B=\exp(-1)\approx0.368$ and $b_C=\exp(-0.5)\approx0.607$. Thus FWT is $[(0.082-0.368)+(0.082-0.607)]/2\approx-0.405$.

The comparison solves its quadratics exactly. EWC-like consolidation uses fixed illustrative $F=(5,0.2)$ and $\theta_{t,k}=(v_{t,k}+\lambda F_k\theta_{t-1,k})/(1+\lambda F_k)$; these manual values are **not** measured Fisher information. Replay uses $\theta_t=(v_t+\beta\bar v_{<t})/(1+\beta)$, equivalent to a weighted average of prior target losses plus the fresh loss. Isolation stores one vector per task and selects it using a known task ID. Its perfect toy retention costs growing storage; it is not a fair fixed-memory victory. The controls show retention and new-task score together.

## Limits
A growing context window is not persistent parameter learning. A recurrent state that resets between sessions is not accumulated knowledge. A continually updated retrieval index is a memory-system change, distinct from weight consolidation. Continue to [memory and test-time adaptation](#/article/memory-adaptation).
