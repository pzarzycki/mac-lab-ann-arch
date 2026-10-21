## The autoregressive objective
For discrete tokens $x_1,\ldots,x_n$, a decoder models

$$
p_\theta(x_{1:n})=\prod_{t=1}^n p_\theta(x_t\mid x_{<t}),\qquad
\mathcal L_{\mathrm{NLL}}=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

$\theta$ denotes trainable parameters. A start token or other boundary convention supplies the first context. In teacher-forced training the full input is known, so causal masking computes predictions for many positions in one pass without exposing a target to its predictor. [1](#ref-transformer) [2](#ref-gpt)

```python
inputs, targets = tokens[:, :-1], tokens[:, 1:]
logits = decoder(inputs, causal=True)
loss = cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
loss.backward()
optimizer.step()
```

Ignore padded positions in a real loss. An optimizer step usually aggregates a batch; gradient accumulation may aggregate several microbatches before that step.

## Prefill versus decode
**Prefill** processes the prompt and constructs cached keys/values. **Decode** adds generated positions, usually one per target-model step. Earlier tokens are available, but the next sampled token is unknown until its distribution has been evaluated. Caching reuses causal computations; it does not remove dependence on previous generated tokens. See [KV cache](#/article/kv-cache).

## Instruction tuning and preferences
Supervised fine-tuning changes the data and loss mask, often scoring assistant completions rather than every input token. RLHF adds preference learning and policy optimization. DPO derives a preference objective relative to a reference policy; these are training procedures, not new kinds of attention. [3](#ref-instruct) [4](#ref-dpo)

For preferred/rejected responses $y_w,y_l$ to prompt $x$, define $r_\theta(y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)$. DPO uses $-\log\operatorname{sigmoid}[\beta(r_\theta(y_w)-r_\theta(y_l))]$, with scale $\beta>0$. This formula assumes a particular preference-model derivation; it is not a universal guarantee of truthfulness.

## What to measure
Distinguish training loss, held-out negative log-likelihood, instruction-following evaluation and generation latency. Perplexity is $\exp(\text{mean NLL})$ under a fixed tokenization and scoring convention. A lower number across incompatible tokenizers is not a valid comparison.
