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

## Follow one feature through all three maps
The lab specializes the displayed update to one token with $m=3,d=1$: $X=[1,2,-1]^\top$. Every output lane is calculated as $(RX)_i=\sum_jR_{ij}X_j$. After sufficient normalization, these are convex combinations of the three inputs. If both marginal constraints hold exactly, their sum is also preserved: $\mathbf1^\top RX=\mathbf1^\top X=2$. Finite Sinkhorn error makes the latter approximate; raw positive weights at zero iterations have neither guarantee.

The lower pathway completes an illustrative HC update rather than stopping at normalization. It sets $a=[0.5,0.25,0.25]^\top$, so $a^\top X=0.75$; a toy token-local sublayer gives $F=\tanh(0.75)\approx0.635$. Redistribution with $b=[1,0.5,0]^\top$ adds approximately $[0.635,0.318,0]^\top$ to $RX$. Thus conservation of the **mixer** does not imply conservation of the whole residual update. Changing lane 3 updates aggregation, mixing and the final sum together. In a real attention sublayer, aggregation occurs for every token before cross-token attention inside $F$.

The positive seed is $[ [\exp(\mathrm{bias}),0.6,1.2],[0.4,1.8,0.8],[1.5,0.3,1] ]$. At zero iterations it is explicitly labeled raw. Logarithmically bounded wires preserve readability even at bias 5, while numeric cells display the unbounded values.

## A scoped empirical comparison
The original HC paper motivates learned expanded connectivity; mHC subsequently adds constraints and an optimized implementation. In mHC's own 27B comparison, trained on 262B tokens with four residual lanes and 20 Sinkhorn iterations, Table 4 reports 3-shot BBH exact match of 43.8 for the residual baseline, 48.9 for HC and 51.0 for mHC. This compares the paper's configurations, not every HC design. The paper separately reports 6.7% additional training time with expansion four after its infrastructure optimizations; a stochastic matrix alone does not establish that overhead. [3](#ref-mhc)
