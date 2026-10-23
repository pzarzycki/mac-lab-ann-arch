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

## Trace matched rows and an update
The numerical companion retains original target-position IDs, rather than comparing arbitrary rows. Its **invented** width-two feature rule is $Y_i=(i/10,1-i/20)$ and $\widehat Y_i=Y_i+(0.2,-0.1)$. For two of the block's target positions:

| Original position | Prediction $\widehat Y_i$ | Stopped target $Y_i$ | Entrywise absolute difference |
| --- | --- | --- | --- |
| 7 | $(0.90,0.55)$ | $(0.70,0.65)$ | $(0.20,0.10)$ |
| 8 | $(1.00,0.50)$ | $(0.80,0.60)$ | $(0.20,0.10)$ |

Mean absolute error over these four scalar entries is $(0.2+0.1+0.2+0.1)/4=0.15$. The figure includes positions $12,13$ as well, with the same error per entry, so its eight-entry mean remains $0.15$. Changing the mask changes both displayed position IDs and their target values; the prediction-error control changes every residual and the reduced loss. The backward arrows run from loss through the predictor **and context encoder**, with no loss-gradient route through the target encoder.

A separate two-parameter example makes EMA arithmetic visible. With old target parameters $\bar\theta=(0.4,0.8)$, updated online parameters $\theta=(1.0,0.2)$ and $m=0.9$, the next target parameters are $(0.46,0.74)$. These are parameter coordinates, not the feature values in the table. The target is moved by that rule after the online update, not trained by differentiating the target side of the matching loss. [1](#ref-ijepa) [2](#ref-vjepa)

If both freely learned branches output $(0,0)$ for every image and position, their matching loss is zero while every feature-coordinate variance across examples is zero. They have erased input distinctions. The companion prints the mean feature-coordinate variance across its four target rows; this is a small diagnostic of variation, not a guarantee of semantic quality. Real evaluations should inspect variability across held-out inputs as well as positions. A nonzero variance alone can coexist with useless features, and a low regression loss alone cannot rule out collapse.

## From images to video
I-JEPA uses spatial image context and targets. V-JEPA extends masked feature prediction to video, making positions spatiotemporal and evaluating motion-sensitive representations. A hidden region need not be strictly in the future: masked video modeling and causal forecasting are different prediction tasks. V-JEPA 2 scales video representation learning, while its action-conditioned extension adds a further stage for planning. Follow [video world models](#/article/video-world-models) for that distinction. [2](#ref-vjepa) [3](#ref-vjepa2)

## Inspect what was learned
An informative comparison holds backbone size, data and compute approximately fixed and varies the prediction target or mask. Inspect feature variability, frozen-feature probes and full fine-tuning separately. A trained probe introduces supervision even when pretraining used none. Our suggested diagnostic is to vary context size and ask whether improvements transfer across appearance and motion tasks.

The official I-JEPA demonstrations visualize features using a separately trained generative decoder. Such pictures help inspect representations; they do not turn the underlying feature predictor into a pixel generator. Similarly, low feature error alone does not establish causal understanding, physical accuracy or useful planning. [4](#ref-ijepa-code)
