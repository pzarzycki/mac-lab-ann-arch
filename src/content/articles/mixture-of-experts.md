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

## Trace data and gate signals separately
For one explicit vector convention, $h\in\mathbb R^d$, $W_r\in\mathbb R^{E\times d}$, $z\in\mathbb R^E$, and each expert returns $E_e(h)\in\mathbb R^d$. The router receives $h$ to choose **where** to compute. Each chosen expert also receives the original $h$ as its **data**; it does not process the router's probability vector. The selected gate multiplies the expert's output vector before summation.

The synthetic lab uses $d=2$, $h=[1,2]$ and $E_e(h)=[e h_1,h_2-e]$ for one-based expert index $e$. At its default logits, top two selects experts 3 and 1, with gates about $0.690$ and $0.310$. Their outputs are $[3,-1]$ and $[1,1]$, giving $y\approx[2.380,-0.380]$ when both routes are admitted. These small fixed experts make the sum inspectable; they are not learned specializations.

## Capacity is a batch constraint
Define expert capacity $C$ as the maximum number of token–expert assignments that an expert may admit in a batch. A useful general top-$k$ bookkeeping convention is $C=\lceil\phi Bk/E\rceil$, where $B$ is tokens, $E$ experts and $\phi$ a capacity factor; exact rounding and policies differ across systems. Switch's top-one setting uses the corresponding tokens-per-expert capacity factor. [1](#ref-switch)

In the six-token lab, shared default preferences and $k=2$ produce demand $[6,0,6,0,0,0]$. With $C=2$, experts 1 and 3 each accept two routes: four assignments execute and eight overflow. Admission is in token order, with overflow **dropped from this expert branch**, and surviving gates are not renormalized. The enclosing residual bypass can still carry token data; dropping a branch contribution is not deleting a token from the sequence. Select a later batch token to inspect the resulting zero expert-branch output. Cyclic preferences rotate the same logit vector by token index to illustrate a balanced demand pattern. Increasing $k$ raises total requested assignments from $B$ to $Bk$ even when weights sum to one for every token.

One concrete balancing objective in Switch is proportional to $E\sum_e f_eP_e$: $f_e$ is the fraction of tokens dispatched to expert $e$, and $P_e$ the mean router probability for that expert over the batch. The probability term supplies a gradient despite hard dispatch counts. Encouraging uniform load can reduce overload, but its coefficient trades routing preference against balance; it neither guarantees equal realized counts nor eliminates capacity overflow. The lab's cyclic pattern is a hand-built comparison, not training with that loss. [1](#ref-switch)
