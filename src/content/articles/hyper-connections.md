## Widen the residual pathway
Conventional residual learning carries one width-$d$ state per token. Hyper-Connections introduce multiple residual lanes and learned maps that combine lanes into a sublayer input and redistribute its output. The lane axis is **within each token**. It is separate from the sequence axis and from attention heads. [1](#ref-resnet) [2](#ref-hc)

Using a schematic column convention, let $X_i\in\mathbb R^{m\times d}$ collect $m$ lanes at token $i$. A generalized update is

$$
X_i'=R_iX_i+b_iF_i(\{a_j^\top X_j\}_{j\le i}).
$$

$R_i\in\mathbb R^{m\times m}$ mixes lanes, $a_j,b_i\in\mathbb R^m$, and $F_i$ returns a width-$d$ sublayer output. The cross-token dependence belongs to attention inside $F$; an MLP version is token-local. This notation summarizes connectivity rather than reproducing every published parameterization.

## Constrain the mixing
mHC constrains the residual mixing matrix to nonnegative, doubly stochastic matrices:

$$
R\mathbf1=\mathbf1,\qquad\mathbf1^\top R=\mathbf1^\top,\qquad R_{ab}\ge0.
$$

The resulting Birkhoff polytope is a convex polytope, despite the paper's broader “manifold” terminology. Its matrices mix lanes through convex combinations and are closed under composition. The residual mixer is non-expansive in spectral norm; this property is **not** a proof that the full nonlinear network preserves every norm or cannot diverge. [3](#ref-mhc)

## A numerical construction
Sinkhorn-style row and column normalization can approximate these constraints from a positive matrix. A finite number of iterations leaves numerical error. Inspect the row and column sums in the lab as the iteration budget changes.

```python
R = exp(logits - logits.max())
for _ in range(iterations):
    R = R / R.sum(axis=0, keepdims=True)
    R = R / R.sum(axis=1, keepdims=True)
```

This example illustrates the constraint and uses a fixed small matrix; it is not the full mHC infrastructure implementation.

## Evidence and limits
The mHC paper combines connectivity constraints with system optimizations. Its empirical gains must be read with the training configuration and overhead. Multiple lanes cost memory traffic; the mathematical constraint alone does not eliminate that cost. Compare against both standard residual and unconstrained HC baselines.
