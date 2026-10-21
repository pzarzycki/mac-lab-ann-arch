## Soft prompts
Prompt tuning optimizes continuous vectors prepended to a frozen language model. If $P\in\mathbb R^{m\times d}$ is a learned prompt and $E(x)$ the embedded input,

$$
P^*=\arg\min_P\;\mathbb E_{(x,y)}[-\log p_{\theta\ \mathrm{fixed}}(y\mid[P;E(x)])].
$$

$m$ is prompt length and $d$ model width. These vectors need not correspond to vocabulary entries and are typically task parameters learned during training. This differs from low-rank weight adaptation, where $\Delta W=BA$ modifies a projection through trainable low-rank factors. [1](#ref-prompt) [3](#ref-lora)

## Soft token mixtures
Another possible construction embeds a probability distribution $p\in\Delta^{V-1}$ as $v=\sum_k p_kE_k$. The simplex $\Delta^{V-1}$ contains nonnegative weights summing to one. Such a convex mixture is mathematically distinct from an arbitrary learned prompt vector; not every vector lies in the convex hull of the embedding table. This formula describes a representation option, not a claim that all latent reasoning systems use it.

## Hidden-state feedback
Coconut feeds a last hidden state back as a subsequent input embedding during latent reasoning, rather than decoding each step into a word. That state is input-dependent and computed at inference; it is not simply a fixed soft prompt. [2](#ref-coconut)

```python
state = model(context).last_hidden_state
for _ in range(latent_steps):
    state = model(inputs_embeds=state).last_hidden_state
# Schematic only: real implementations preserve their context and cache layout.
```

## What continuous does not imply
A continuous vector is not automatically an interpretable probability distribution over alternative thoughts. A paper's search analogy must be separated from a verified explicit search algorithm. Latent steps consume compute even if they emit no readable tokens.

## Experiment
Compare visible chain-of-thought and latent reasoning with matched total compute and training supervision. Measure generalization outside the training task family, not only token savings. Inspect failure cases and whether the model learns a meaningful stopping rule. Continue to [recurrent depth](#/article/recurrent-depth).
