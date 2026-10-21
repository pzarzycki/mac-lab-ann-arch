## Route a token to a subset
A mixture-of-experts (MoE) feed-forward block uses a router to select experts. With token state $h$, logits $z=W_rh$ and selected set $S(h)$,

$$
y(h)=\sum_{e\in S(h)}g_e(h)E_e(h).
$$

$E_e$ is expert $e$, and $g_e$ its routing weight. The lab uses softmax **renormalized over the selected top-$k$ logits**. This is an explicit educational convention; published systems differ in scoring, normalization, shared experts and balancing. Switch Transformers studied a simplified top-one scheme. [1](#ref-switch)

## Sparse does not mean cheap in every sense
Experts increase stored parameters while activating a subset for each token. Routing can require all-to-all communication across devices, uneven loads, capacity management and extra buffers. “Active parameters” is not a complete FLOP or latency measure. Total parameter memory still matters unless experts are otherwise offloaded or partitioned.

```python
selected = topk(router_logits, k)
weights = softmax(router_logits[selected])
y = sum(weights[j] * experts[e](h) for j, e in enumerate(selected))
```

## Balance and specialization
If the router sends most tokens to one expert, the system wastes parallel capacity. Balancing losses, capacity policies and other mechanisms address this issue. DeepSeek-V3 provides another concrete design point with routed and shared experts and a distinct load-balancing approach. [2](#ref-deepseek)

## Inspect the router
Move a router logit and change $k$. Only highlighted experts execute in the illustration. The numeric coefficients are computed from your inputs; the picture does not assert that experts have clean semantic labels or stable human-interpretable specializations.

## Evaluate the right quantities
Compare training compute, communication volume, inference latency at representative batch sizes, parameter storage and downstream quality. Report dropped tokens if capacity-limited routing drops them. A single “trillion parameter” label does not establish superiority over a dense baseline at matched cost.
