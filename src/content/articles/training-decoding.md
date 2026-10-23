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

## Align one short sequence
Let $B$ be batch size, $T$ the number of scored input positions, and $V$ vocabulary size. Integer inputs and targets each have shape $B\times T$; logits have shape $B\times T\times V$. Softmax over the last axis supplies a probability distribution for each position's next target. Flattening in the code preserves that input/target alignment.

For the illustrative sequence `[BOS, The, model, learns, EOS]`, use inputs `[BOS, The, model, learns]` and targets `[The, model, learns, EOS]`, so $B=1,T=4$. At output position 3, the causal model reads `[BOS, The, model]` and scores `learns`; it cannot read the fourth input. A toy target probability $0.60$ contributes $-\ln(0.60)\approx0.511$ **nats**. With stipulated target probabilities $[0.70,0.50,0.60,0.80]$, the unmasked mean loss is about $0.446$ nats. Logarithms here are natural; perplexity is $\exp(\text{mean NLL})$. These numbers illustrate loss arithmetic and are not measurements of a language model.

## Prefill versus decode
**Prefill** processes the prompt and constructs cached keys/values. **Decode** adds generated positions, usually one per target-model step. Earlier tokens are available, but the next sampled token is unknown until its distribution has been evaluated. Caching reuses causal computations; it does not remove dependence on previous generated tokens. See [KV cache](#/article/kv-cache).

The inference side starts with the three-token prompt `[BOS, The, model]`. Prefill produces three cached K/V positions per layer and prompt logits of shape $1\times3\times V$. The **last** prompt-position distribution predicts the first new token. Suppose sampling returns `learns`: the emitted token is not yet in the cache. The following decode call consumes `learns`, projects its K/V at every layer and appends position 4. Its new query reads positions 1–4, and logits of shape $1\times1\times V$ predict the following token, such as `EOS`. Move the phase control to distinguish prefill, sampling and consumption; sampling itself does not append cache entries.

## Instruction tuning and preferences
Supervised fine-tuning changes the data and loss mask, often scoring assistant completions rather than every input token. RLHF adds preference learning and policy optimization. DPO derives a preference objective relative to a reference policy; these are training procedures, not new kinds of attention. [3](#ref-instruct) [4](#ref-dpo)

For preferred/rejected responses $y_w,y_l$ to prompt $x$, define $r_\theta(y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)$. DPO uses $-\log\operatorname{sigmoid}[\beta(r_\theta(y_w)-r_\theta(y_l))]$, with scale $\beta>0$. This formula assumes a particular preference-model derivation; it is not a universal guarantee of truthfulness.

## What to measure
Distinguish training loss, held-out negative log-likelihood, instruction-following evaluation and generation latency. Perplexity is $\exp(\text{mean NLL})$ under a fixed tokenization and scoring convention. A lower number across incompatible tokenizers is not a valid comparison.
