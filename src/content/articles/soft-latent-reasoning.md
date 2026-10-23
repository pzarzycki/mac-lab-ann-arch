## Soft prompts
Prompt tuning optimizes continuous vectors prepended to a frozen language model. If $P\in\mathbb R^{m\times d}$ is a learned prompt and $E(x)$ the embedded input,

$$
P^*=\arg\min_P\;\mathbb E_{(x,y)}[-\log p_{\theta\ \mathrm{fixed}}(y\mid[P;E(x)])].
$$

$m$ is prompt length and $d$ model width. These vectors need not correspond to vocabulary entries and are typically task parameters learned during training. This differs from low-rank weight adaptation, where $\Delta W=BA$ modifies a projection through trainable low-rank factors. [1](#ref-prompt) [3](#ref-lora)

## Soft token mixtures
Another possible construction embeds a probability distribution $p\in\Delta^{V-1}$ as $v=\sum_k p_kE_k$. The simplex $\Delta^{V-1}$ contains nonnegative weights summing to one. Such a convex mixture is mathematically distinct from an arbitrary learned prompt vector; not every vector lies in the convex hull of the embedding table. This formula describes a representation option, not a claim that all latent reasoning systems use it.

The figure makes both constructions numerical. Its scalar prompt example freezes $\widehat y=P+0.5$ and uses $\mathcal L=\tfrac12(\widehat y-1)^2$, so $\partial\mathcal L/\partial P=P-0.5$. At $P=0$, learning rate $0.25$ moves $P$ to $0.125$; the backbone has no parameter update. In the mixture example, $E_1=(1,0)$, $E_2=(0,2)$ and $E_3=(-1,0)$. At $p=(0.5,0.3,0.2)$, $v=(0.3,0.6)$, inside their triangle. Adjusting the first weight redistributes the remainder in a fixed $3:2$ ratio. These toy values illustrate computation, not trained prompts or language semantics.

## Hidden-state feedback
Coconut feeds a last hidden state back as a subsequent input embedding during latent reasoning, rather than decoding each step into a word. That state is input-dependent and computed at inference; it is not simply a fixed soft prompt. [2](#ref-coconut)

Let $U_j\in\mathbb R^{j\times d}$ contain **all** input vectors through position $j$, and let $h_j=[F_\theta(U_j)]_j\in\mathbb R^d$ select only the final-position normalized hidden state. During a latent step,

$$
U_{j+1}=[U_j;h_j],\qquad h_{j+1}=[F_\theta(U_{j+1})]_{j+1}.
$$

In ordinary decoding, the appended vector instead comes from a vocabulary choice: $w_{j+1}\sim\operatorname{softmax}(Wh_j)$, followed by $E(w_{j+1})$. The latent path bypasses both operations; it does not reinterpret $h_j$ as vocabulary probabilities.

Coconut brackets latent positions with **`<bot>` and `<eot>`**. After the entry marker, the final-position state supplies the next embedding; at exit, token embeddings and vocabulary decoding resume. Retaining $U_j$ is essential. An equivalent cached implementation keeps prior key/value states, advances positions and supplies only the new embedding; it does not repeatedly treat one isolated state as the whole context. The figure compares these paths with an explicitly invented recurrence, not a trained model. [2](#ref-coconut)

Training starts with ordinary language reasoning chains, then stage $k$ replaces the first $k$ reasoning steps by $kc$ continuous thoughts, where $c$ is a chosen count per removed step. Cross-entropy supervises remaining language/answer tokens, with question and latent-position losses masked; gradients still pass through the latent computations. The published experiments generally fix the inference latent count to the final training stage; a learned stop classifier is another discussed option. Applying feedback to an arbitrary frozen model does not reproduce this training. [2](#ref-coconut)

## What continuous does not imply
A continuous vector is not automatically an interpretable probability distribution over alternative thoughts. A paper's search analogy must be separated from a verified explicit search algorithm. Latent steps consume compute even if they emit no readable tokens.

## Experiment
Compare visible chain-of-thought and latent reasoning with matched total compute and training supervision. Measure generalization outside the training task family, not only token savings. Inspect failure cases and whether the model learns a meaningful stopping rule. Continue to [recurrent depth](#/article/recurrent-depth).
