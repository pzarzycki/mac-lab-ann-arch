## A common width does not imply a common meaning
Text IDs become learned lookup vectors. An image can be divided into patches, projected into a sequence and processed by a vision encoder. Audio may become time–frequency features or learned tokens. To enter a language backbone, these representations need a compatible width and a specified ordering/masking convention. [1](#ref-vit)

For image features $Z\in\mathbb R^{p\times d_v}$ and a learned projector $P\in\mathbb R^{d_v\times d}$, define $V=ZP$. $p$ is visual-token count, $d_v$ encoder width and $d$ language width. A width match does not guarantee semantic alignment; training establishes how the backbone uses these vectors.

## Three fusion patterns
| Pattern | Interface | Main design issue |
| --- | --- | --- |
| Visual prefix | Concatenate projected visual tokens with text | Sequence cost and masking |
| Learned query bottleneck | Query a larger feature set to produce fewer tokens | Information lost by compression |
| Cross-attention | Language states query visual memory in selected layers | Extra projections and insertion points |

LLaVA provides an influential encoder–projector–language-model route. BLIP-2 uses a Q-Former between frozen pretrained components. Flamingo uses a visual resampler and gated cross-attention to connect pretrained vision and language systems. These are specific designs, not interchangeable names for “multimodal attention.” [2](#ref-flamingo) [3](#ref-blip) [4](#ref-llava)

## Inspect the interface
Switch the fusion mode in the vector figure. Count which tensors join the language sequence and which remain external memory. The displayed token counts are illustrative; actual models differ in image crops, resolution, resampling and temporal handling.

```python
visual = vision_encoder(images)
visual_tokens = projector(visual)
sequence = concatenate([visual_tokens, text_embeddings], axis='positions')
answer = decoder(sequence, mask=multimodal_mask)
```

## Evidence and limits
A captioning improvement does not establish precise spatial grounding. Frozen backbones reduce trainable parameters but do not remove the cost of their forward passes. Evaluation should vary image content while preserving the question, and include cases where language priors suggest the wrong answer. Continue to [training objectives](#/article/multimodal-objectives).
