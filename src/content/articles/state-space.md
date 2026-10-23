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

## Choose shapes before interpreting selection
For one input channel, take $x_t,y_t\in\mathbb R$, $s_t\in\mathbb R^N$, $\overline A_t\in\mathbb R^{N\times N}$, $\overline B_t\in\mathbb R^{N\times1}$ and $C_t\in\mathbb R^{1\times N}$. The state update produces the next carried memory. The readout branches from that state and leaves the recurrence; **$y_t$ is not substituted for $s_t$**. Multi-channel implementations add channel and batch axes and often use diagonal structure rather than dense $N\times N$ matrices.

Mamba's selection makes $\Delta_t,B_t,C_t$ input-dependent while its continuous-time $A$ is a learned structured parameter. Discretization makes the effective transition input-dependent: for example $\overline A_t=\exp(\Delta_t A)$. Thus saying “selective $\overline A_t$” does not mean that a fresh unconstrained continuous matrix is predicted at each token. The learned timestep controls retention; the write and read parameters control what enters state and what is exposed. [1](#ref-mamba)

## A six-token trace
The synthetic lab reduces the shapes to $N=1$. Start at $s_0=0$. On a write token choose $(a_t,b_t)=(0,1)$; on noise or read tokens choose $(1,0)$. Set $c_t=1$ on read tokens and zero otherwise. This is a hand-written selective rule, not Mamba's learned gate.

For “write 2,” $s_1=0\cdot0+1\cdot2=2$. For the next distractor “noise 9,” $s_2=1\cdot2+0\cdot9=2$. Its readout is zero, yet the next prior remains 2. A second distractor leaves it intact; the read token exposes 2. A later “write −1” overwrites state, so the final read returns −1. The alternative fixed filter uses $a=b=0.5$ at every step: its first two states are 1 and 5, showing how a distractor can enter a nonselective summary. The displayed table computes every retained term, write term, state and readout.

The memory comparison chooses its own exact toy widths: four recurrent FP16 scalars require eight bytes, while K and V of width two each require $2T\cdot2\cdot2=8T$ bytes at length $T$. This describes only those arrays, not full Mamba or Transformer memory; real recurrent blocks may also keep convolution state.

## One concrete hybrid layout
Jamba's initial released configuration uses four blocks of eight layers, each mixing one attention layer with seven Mamba layers; every layer also has an MLP or expert feed-forward stage. Its attention layers retain token-indexed K/V, while its Mamba layers retain recurrent and convolution states. The placement of only some attention layers reduces the number of growing K/V arrays but preserves explicit attention in the network. This is a documented design point, not a universal best ratio or a claim that an SSM state can reconstruct all prior keys and values. [4](#ref-jamba)
