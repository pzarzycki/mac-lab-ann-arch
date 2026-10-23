## Learn a correction
ResNet popularized the parameterization $y=x+F_\theta(x)$ for very deep networks. When input and output widths agree, the skip can be the identity. With a projection skip $P$, the block is $y=Px+F_\theta(x)$ instead; its derivative is $P+J_F$, not $I+J_F$. Earlier skip connections and gated highway designs are part of the broader history, so ResNet should not be credited with every use of a bypass. [1](#ref-resnet)

$$
\frac{\partial\mathcal L}{\partial x}
=\left(I+J_{F_\theta}(x)\right)^\top\frac{\partial\mathcal L}{\partial y}.
$$

$J_F$ is the Jacobian of the learned residual function. The identity term supplies a direct derivative route. It does not make gradients immune to accumulation, poor scaling or unstable normalization.

The displayed derivative is for the **additive junction before any post-add activation**. Original ResNet blocks also applied ReLU after the sum; differentiating the whole block multiplies the junction derivative by the ReLU Jacobian. A closed gate can remove the local direct path. [1](#ref-resnet)

## Why the original degradation result mattered
The original paper's CIFAR-10 comparison found a 56-layer plain network harder to optimize than a 20-layer plain network: it had higher **training** error, as well as higher test error. That observation cannot be explained only by overfitting. A deeper model could in principle reproduce a shallower solution using identity-like added layers, yet the optimizer did not simply find that solution. Residual parameterization makes such an added mapping expressible as $F(x)=0$, rather than asking a stack of nonlinear layers to learn identity directly. This is the scoped empirical motivation, not a theorem that all residual networks train successfully. [1](#ref-resnet)

## A derivative that can cancel
The figure sets $x=1$, upstream gradient $g_y=1$ and a synthetic scalar branch $F(x)=jx$. A plain block gives $g_x=j$, an identity residual block gives $g_x=1+j$, and a projection skip $P=0.5$ gives $g_x=0.5+j$. At $j=0.25$, these are $0.25$, $1.25$ and $0.75$. At $j=-1$, the identity and branch terms cancel exactly. Large positive $j$ can amplify a gradient; repeated blocks can accumulate that effect. Toggle post-add ReLU to see the further gate, with derivative zero chosen at the kink. These are local derivative calculations, not measured training curves.

## From convolutional blocks to token states
The Transformer explicitly wraps attention and feed-forward sublayers in residual connections. Replacing convolutions with attention does not erase this inheritance. Each token still carries its own vector through elementwise addition, even though attention reads from other tokens. [2](#ref-transformer)

## The modern extension
Hyper-Connections generalize routing across residual pathways. mHC constrains that mixing to improve stability. This is an extension of a residual-connectivity axis, not the same operation as adding more attention heads. [3](#ref-hc) [4](#ref-mhc)

```python
# Identity skip requires equal tensor shapes.
assert branch(x).shape == x.shape
y = x + branch(x)
```

## Evidence and limits
The original residual paper studied image recognition. Its direct architectural importance to Transformers is supported by the Transformer's own residual construction; the original ImageNet benchmark is not evidence about language-model performance. A sound historical account separates the original experiment from the later mechanism transfer.

## A useful experiment
Compare a deep plain MLP with an equal-depth residual MLP. Record loss, layerwise gradient norms and activation scale. Match widths and training budget. If you add normalization only to one model, the result no longer isolates the identity path. Read [mHC](#/article/hyper-connections) after understanding this baseline.
