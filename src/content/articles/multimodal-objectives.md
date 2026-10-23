## Contrastive alignment
CLIP learns image and text encoders whose normalized outputs are compared through similarity. For a batch of $B$ paired examples, one direction of a contrastive loss is

$$
\mathcal L_{I\to T}=-\frac1B\sum_i\log\frac{\exp(s(v_i,t_i)/\tau)}{\sum_j\exp(s(v_i,t_j)/\tau)}.
$$

$v_i,t_i$ are normalized image/text embeddings, $s$ their dot product and $\tau>0$ temperature. The symmetric objective also scores text-to-image retrieval. In-batch negatives create the classification problem; false negatives and biased captions can affect learning. [1](#ref-clip)

## A three-pair batch that can be calculated
The figure uses synthetic normalized image and text vectors $[1,0]$, $[0,1]$, $[0.8,0.6]$ with invented labels red square, blue circle and red object. Their dot-product matrix is

$$
S=\begin{bmatrix}1&0&0.8\\0&1&0.6\\0.8&0.6&1\end{bmatrix}.
$$

At $\tau=0.3$, image 1's row gives probabilities $[0.646,0.023,0.331]$ over text candidates, so the paired-target loss is $-\log(0.646)=0.438$ nats. Text-to-image instead normalizes a column. This matrix happens to be symmetric because the toy encoders share the displayed vectors; trained dual encoders need not produce a symmetric cross-batch matrix. A smaller temperature sharpens the distribution and changes the loss; it does not change which pair is labeled positive. Red object might also describe image 1, exposing the possible false-negative problem of diagonal pair labels. The symmetric training loss averages the row and column batch means. [1](#ref-clip)

## Conditional language generation
A generative vision-language objective instead scores text given visual input $v$:

$$
\mathcal L_{\mathrm{gen}}=-\sum_t\log p_\theta(y_t\mid y_{<t},v).
$$

This asks for a token distribution, not merely a paired-embedding similarity. A model can use contrastive and generative terms in different phases. BLIP-2 is a useful example of a staged bridge between pretrained vision and language components. [2](#ref-blip)

## Visual instruction tuning
Instruction data specify how to answer questions about images. LLaVA studied generating multimodal instruction data and fine-tuning a vision-language assistant. Instruction following adds behavior and task formatting; it does not prove that the model's visual representation captures every answer-relevant detail. [3](#ref-llava)

```python
loss = image_text_contrastive_loss(image_repr, text_repr)
# A separate generative stage may optimize answer tokens only.
loss = conditional_answer_loss(images, instructions, answers, answer_mask)
```

Visual instruction tuning here is a **data and training regime using conditional likelihood**, not a third mathematical loss separate from generation. What changes is the instruction/response data and the supervised span. In the figure, the conditioning sequence is `[image] User: What color? Assistant:`; the answer is `red [stop]`. The target for `red` is predicted from the preceding positions, and the target for `[stop]` includes `red` in its context. An answer mask $[0,0,0,0,0,1,1]$ scores only those last two target positions; it is a loss mask, not an attention mask. With assumed probabilities 0.8 and 0.5, the summed loss is $-\log(0.8)-\log(0.5)=0.916$, or 0.458 per scored token. Prompt positions still affect the answer through conditioning. [3](#ref-llava)

| Regime | Supervision / output | Loss and scored span | Typical trainable modules in the cited example |
| --- | --- | --- | --- |
| CLIP alignment | Paired image/text; pair identity | Symmetric batch classification | Image and text encoders |
| Conditional caption generation | Image/caption; caption tokens | Conditional likelihood of target text | Architecture/stage dependent; state which modules are frozen |
| LLaVA visual instruction tuning | Image, instruction, response; assistant answer | Same conditional token likelihood, assistant response spans | Projector and language model in instruction tuning; vision encoder frozen |

BLIP-2 combines objectives in two stages: with its image encoder frozen, it trains Q-Former with image–text contrastive, image-grounded generation and image–text matching losses; then it trains Q-Former and its width projection to condition a frozen LLM with a generative objective. The pretrained image encoder stays frozen. The image/text interface, objective and frozen modules are thus separate choices even within one system. [2](#ref-blip)

## Keep the axes separate
An early-fusion model can be trained generatively. A dual encoder can be trained contrastively. A query bottleneck can support several losses. The interface, attention pattern, data mixture and training objective should be separate fields in a model comparison.

## Research questions
Test compositional retrieval, fine-grained attribute swaps and image-question counterfactuals. Include resolution and visual-token budgets. Compare a language-only baseline to quantify how much the task rewards visual evidence rather than dataset priors. Report which encoders were frozen during each phase.
