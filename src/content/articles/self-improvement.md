## Improve an answer or improve a model?
Self-Refine repeatedly generates feedback and revisions with fixed model weights. STaR instead generates reasoning traces, selects or rationalizes correct solutions using answer supervision, and fine-tunes on resulting data. One changes an output at inference; the other changes parameters through a training loop. [1](#ref-star) [2](#ref-selfrefine)

## A generic generate–verify–learn loop
Let $G_{\theta_k}$ generate candidates from prompts $x$, and let $V(x,y)$ be a selection score. One schematic update is

$$
\mathcal D_k=\{(x,y):y\sim G_{\theta_k}(\cdot\mid x),\;V(x,y)\ge\tau\},\qquad
\theta_{k+1}=\operatorname{Train}(\theta_k,\mathcal D_k).
$$

$\tau$ is an acceptance threshold. This formalizes a family of procedures, not the exact STaR algorithm. Selection can use known answers, unit tests, execution, a learned reward model or human judgment. Each verifier has blind spots.

```python
candidates = generate(training_prompts)
accepted = [c for c in candidates if independent_verifier(c)]
next_model = train(current_model, accepted)
evaluate(next_model, untouched_test_set)
```

## The hard part is evidence
A model judging its own output may prefer its own characteristic mistakes. Training on accepted synthetic text can amplify a biased selection signal. Repeatedly inspecting a test set can turn it into training feedback even without gradient updates. Preference-based post-training also depends on what labels reward. [3](#ref-instruct)

## Evaluation protocol
Freeze a genuinely held-out task suite before the loop. Track compute, external information and supervision supplied by the verifier. Include comparisons with more samples, more inference steps and ordinary supervised training at matched cost. Test whether improvement transfers beyond the verifier's original task distribution.

## Open questions
A narrow system with reliable executable feedback can improve a specific skill. That is not a demonstration of open-ended autonomous scientific progress or a general recursive-intelligence law. The interesting engineering question is where a trustworthy, informative signal can be obtained cheaply enough to drive a useful loop.
