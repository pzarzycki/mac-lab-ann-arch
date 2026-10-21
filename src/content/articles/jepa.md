## Predict a representation
A Joint-Embedding Predictive Architecture (JEPA) learns to predict a target's representation from a context representation. The family describes a learning arrangement, not a replacement for attention: its encoders and predictor can themselves be Transformers. I-JEPA instantiates this idea for images, predicting target-block features from an informative context block. Its masking choices encourage prediction beyond immediately adjacent texture. [1](#ref-ijepa)

The distinction from reconstruction is the prediction target. A pixel reconstruction loss asks for an image array; a JEPA loss compares learned features. This permits the representation to discard some visual detail, but does not establish that every retained feature is useful or that all discarded detail is irrelevant. The diagram switches the target space while keeping this distinction visible.

## Context, target and predictor
Let an image or clip contain $N$ patch positions. A context mask selects $C$ and target positions form $T$, with $C\cap T=\varnothing$. A useful schematic is

$$
Z_C=f_\theta(x_C),\quad
Y_T=[f_{\bar\theta}(x)]_T,\quad
\widehat Y_T=g_\phi(Z_C,p_T).
$$

$Z_C\in\mathbb R^{|C|\times d}$ contains visible-context features; $Y_T,\widehat Y_T\in\mathbb R^{|T|\times d}$ are target and predicted features. $p_T$ identifies the requested positions. In the loss below, rows retain their original target-position labels even when stored as a compact array. The target encoder can see the full input; target positions are selected from its output. The context branch must not silently receive the hidden target pixels. Multiple target masks and normalization details belong to each implementation. [1](#ref-ijepa) [2](#ref-vjepa)

These are separate token-indexed feature arrays. A diagram that reduces each branch to one box is summarizing its computation, not claiming that the encoder has only one shared residual vector. Inside a Transformer encoder, every visible patch retains its own residual state.

## Why two encoder updates differ
For illustration, a feature-regression objective is

$$
\mathcal L=\frac1{|T|d}\sum_{i\in T}\sum_{k=1}^d
\rho(\widehat Y_{ik}-\operatorname{sg}(Y_{ik})),\qquad
\bar\theta\leftarrow m\bar\theta+(1-m)\theta.
$$

$\operatorname{sg}$ stops gradients through the target; $m\in[0,1)$ controls an exponential moving average. Gradient updates train the context encoder and predictor, while the target follows the moving average. The penalty $\rho$ is method-specific: V-JEPA uses absolute error. This is a family-level notation, not a claim that all JEPA variants use identical losses. Naive jointly learned feature matching admits a constant-output collapse; the asymmetric predictor, moving-average target and stop-gradient form a practical prevention strategy, not a universal theorem of successful representation learning. [2](#ref-vjepa)

## From images to video
I-JEPA uses spatial image context and targets. V-JEPA extends masked feature prediction to video, making positions spatiotemporal and evaluating motion-sensitive representations. A hidden region need not be strictly in the future: masked video modeling and causal forecasting are different prediction tasks. V-JEPA 2 scales video representation learning, while its action-conditioned extension adds a further stage for planning. Follow [video world models](#/article/video-world-models) for that distinction. [2](#ref-vjepa) [3](#ref-vjepa2)

## Inspect what was learned
An informative comparison holds backbone size, data and compute approximately fixed and varies the prediction target or mask. Inspect feature variability, frozen-feature probes and full fine-tuning separately. A trained probe introduces supervision even when pretraining used none. Our suggested diagnostic is to vary context size and ask whether improvements transfer across appearance and motion tasks.

The official I-JEPA demonstrations visualize features using a separately trained generative decoder. Such pictures help inspect representations; they do not turn the underlying feature predictor into a pixel generator. Similarly, low feature error alone does not establish causal understanding, physical accuracy or useful planning. [4](#ref-ijepa-code)
