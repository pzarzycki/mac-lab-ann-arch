## A state update
A discrete state-space recurrence can be written

$$
s_t=\overline A_t s_{t-1}+\overline B_t x_t,\qquad y_t=C_t s_t.
$$

$x_t$ is current input, $s_t\in\mathbb R^{d_s}$ the recurrent state and $y_t$ the output. Shapes and channel structure vary by implementation. In a selective model, some effective parameters depend on the input; the model can choose what to retain or discard instead of applying one fixed linear filter. Mamba combines this selection with a hardware-aware scan and a specific architecture. [1](#ref-mamba)

## Different memory organizations
Attention can keep explicit keys and values for prior positions. A state-space recurrence compresses history into a state whose size is independent of sequence length for a fixed model. This can improve long-sequence scaling, but compression and explicit content-addressable retrieval have different inductive biases. A recurrent state is not a lossless substitute for arbitrary token memory.

## Hybrid designs
Combining attention layers with recurrent or state-space layers is a design space: retain some explicit retrieval while using compressed state elsewhere. This is a general architectural possibility, not a claim that one layout is optimal for all workloads. The number and placement of attention layers affect cache memory and retrieval capacity. [2](#ref-transformer)

```python
state = initial_state
for token in tokens:
    A, B, C = selective_parameters(token)
    state = A @ state + B @ token
    output = C @ state
```

The code is conceptual. Efficient implementations exploit structured parameterization and scans rather than arbitrary dense matrix products per token.

## Read the evidence carefully
The Mamba paper's task and throughput results belong to its tested models and hardware. Linear scaling in sequence length does not establish a universal latency advantage or unbounded exact recall. Compare parameter count, training data, retrieval tasks, context length and systems implementation together.

## Experiment
Use a long-range associative-recall task with distractors and variable delays. Test lengths beyond training. Measure state size alongside accuracy. Compare against attention at a matched memory or compute budget, rather than only matching model parameters.
