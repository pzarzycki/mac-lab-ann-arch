## Quantize a representation
A schematic affine quantizer uses integer codes $q$ to approximate a weight $w$:

$$
q=\operatorname{clip}(\operatorname{round}(w/s)+z,q_{\min},q_{\max}),\qquad
\widehat w=s(q-z).
$$

$s>0$ is scale, $z$ zero point and $[q_{\min},q_{\max}]$ the code range. Scale grouping, calibration data, outliers and kernel support matter. GPTQ uses approximate second-order information to choose a low-bit weight representation; it is more specific than naive rounding. [1](#ref-gptq)

Weight quantization, activation quantization and KV-cache quantization change different memory terms. Compression can perturb output distributions. The bit width alone is not a quality or speed guarantee, and scales and packing add storage overhead.

The toy quantizer groups five synthetic weights $[-1.3,-0.35,0.2,0.9,2.1]$. At four bits, $s=0.25$, $z=8$ and unsigned codes $0\ldots15$, it stores $[3,7,9,12,15]$ and reconstructs $[-1.25,-0.25,0.25,1,1.75]$. The last value clips at the representable maximum; smaller rounding errors affect the others. The panel shows signed residual errors and mean squared error as scale and bit width change. Its stated format needs $\lceil5\times4/8\rceil=3$ packed payload bytes plus 4 bytes for one float32 scale and 1 byte for one uint8 zero point, totaling 8 bytes versus 20 bytes of float32 weights. Real grouping, alignment and metadata layouts differ; this simple quantizer is not GPTQ.

## Speculate, then verify
A cheap draft model proposes several tokens. The target model evaluates them together and an acceptance/correction procedure preserves the target distribution under the algorithm's assumptions. This differs from accepting the draft whenever two argmax tokens agree. [2](#ref-speculative)

For a proposed token $x$ from draft distribution $q$, acceptance uses $\min(1,p(x)/q(x))$, where $p$ is the target distribution conditioned on the corresponding accepted prefix. After a rejection, exact correction samples from a distribution proportional to $\max(0,p-q)$. Details such as the extra target token and sequential conditioning are essential.

Here $p$ and $q$ must be the **actual distributions after sampling transformations**, such as temperature or truncation, rather than unadjusted logits. Draft proposal $x_i$ and distributions $p_i,q_i$ use the same original context followed by proposals $x_{<i}$. Once the first rejection occurs, those later proposed contexts are abandoned. [2](#ref-speculative)

```python
proposals, q = draft_sequentially(context, proposal_count)
p = target_verify(context, proposals)  # distributions p[0] ... p[k]
for i, token in enumerate(proposals):
    if uniform() < min(1, p[i][token] / q[i][token]):
        continue                       # keep token; check next proposal
    correction = normalize(positive_part(p[i] - q[i]))
    return context + proposals[:i] + [sample(correction)]
# Every proposal was accepted: p[k] follows the complete proposal prefix.
return context + proposals + [sample(p[len(proposals)])]
```

The rejection branch retains the accepted prefix, discards the rejected proposal **and every later proposal**, then samples one correction token. Only the all-accepted branch samples a bonus token from the target after the complete proposal prefix. Stop-token handling can terminate generation earlier. [2](#ref-speculative)

For a one-step three-token vocabulary, let $p=[0.2,0.5,0.3]$ and $q=[0.5,0.3,0.2]$. Accepted draft mass is $q(x)\min(1,p(x)/q(x))=\min(p(x),q(x))$, namely $[0.2,0.3,0.2]$. Total rejection mass is $0.3$ and the normalized correction is $[0,2/3,1/3]$. Adding its unconditional contribution gives

$$
[0.2,0.3,0.2]+0.3[0,2/3,1/3]=[0.2,0.5,0.3]=p.
$$

The interactive cycle uses these same distributions at every prefix as an explicit toy assumption, with draft proposals `A,A,C`. It exposes rejection at the first or second proposal and full acceptance, including the resulting token strips. In a real language model the distributions change with the prefix; the same conditional correction argument applies at each checked step. Exactness assumes consistent distributions and probability arithmetic; it does not imply identical random samples to ordinary decoding. [2](#ref-speculative)

## Compare costs
Drafting adds work. Speed depends on acceptance rate, proposal length, verification efficiency and hardware utilization. A strong draft can be too expensive; a weak draft can have poor acceptance. Published speedups are measurements of a specific system, not an architecture-wide constant.

## Experiment design
Measure identical prompts, sampling settings, output lengths and target weights with and without speculation. Check distributions statistically rather than expecting identical random samples. For quantization, add held-out quality evaluation and report true packed memory. Both methods can coexist with a [grouped-head KV cache](#/article/kv-cache).
