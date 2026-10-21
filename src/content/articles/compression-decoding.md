## Quantize a representation
A schematic affine quantizer uses integer codes $q$ to approximate a weight $w$:

$$
q=\operatorname{clip}(\operatorname{round}(w/s)+z,q_{\min},q_{\max}),\qquad
\widehat w=s(q-z).
$$

$s>0$ is scale, $z$ zero point and $[q_{\min},q_{\max}]$ the code range. Scale grouping, calibration data, outliers and kernel support matter. GPTQ uses approximate second-order information to choose a low-bit weight representation; it is more specific than naive rounding. [1](#ref-gptq)

Weight quantization, activation quantization and KV-cache quantization change different memory terms. Compression can perturb output distributions. The bit width alone is not a quality or speed guarantee, and scales and packing add storage overhead.

## Speculate, then verify
A cheap draft model proposes several tokens. The target model evaluates them together and an acceptance/correction procedure preserves the target distribution under the algorithm's assumptions. This differs from accepting the draft whenever two argmax tokens agree. [2](#ref-speculative)

For a proposed token $x$ from draft distribution $q$, acceptance uses $\min(1,p(x)/q(x))$, where $p$ is the target distribution conditioned on the corresponding accepted prefix. After a rejection, exact correction samples from a distribution proportional to $\max(0,p-q)$. Details such as the extra target token and sequential conditioning are essential.

## Compare costs
Drafting adds work. Speed depends on acceptance rate, proposal length, verification efficiency and hardware utilization. A strong draft can be too expensive; a weak draft can have poor acceptance. Published speedups are measurements of a specific system, not an architecture-wide constant.

## Experiment design
Measure identical prompts, sampling settings, output lengths and target weights with and without speculation. Check distributions statistically rather than expecting identical random samples. For quantization, add held-out quality evaluation and report true packed memory. Both methods can coexist with a [grouped-head KV cache](#/article/kv-cache).
