## Contrastive alignment
CLIP learns image and text encoders whose normalized outputs are compared through similarity. For a batch of $B$ paired examples, one direction of a contrastive loss is

$$
\mathcal L_{I\to T}=-\frac1B\sum_i\log\frac{\exp(s(v_i,t_i)/\tau)}{\sum_j\exp(s(v_i,t_j)/\tau)}.
$$

$v_i,t_i$ are normalized image/text embeddings, $s$ their dot product and $\tau>0$ temperature. The symmetric objective also scores text-to-image retrieval. In-batch negatives create the classification problem; false negatives and biased captions can affect learning. [1](#ref-clip)

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

## Keep the axes separate
An early-fusion model can be trained generatively. A dual encoder can be trained contrastively. A query bottleneck can support several losses. The interface, attention pattern, data mixture and training objective should be separate fields in a model comparison.

## Research questions
Test compositional retrieval, fine-grained attribute swaps and image-question counterfactuals. Include resolution and visual-token budgets. Compare a language-only baseline to quantify how much the task rewards visual evidence rather than dataset priors. Report which encoders were frozen during each phase.
