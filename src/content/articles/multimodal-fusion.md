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

## Sequence membership and an explicit mask
Take $p=8$ encoded visual tokens and $n=4$ text tokens. A projected prefix has order $[v_1,\ldots,v_8,t_1,\ldots,t_4]$, length 12, all at language width $d$. One illustrative prefix mask allows each visual position to read all visual positions and no text, and each text position to read all visual positions plus text positions up to itself. This is a specified example, not the universal mask of all named systems.

| Interface in the figure | Language self-attention length | Visual read / extra work per head |
| --- | --- | --- |
| Projected prefix | $p+n=12$ | Dense score matrix has $12^2=144$ entries |
| $m=2$ query outputs projected as prefix | $m+n=6$ | $mp=16$ query-to-vision scores, then $6^2=36$ self scores |
| External visual memory | $n=4$ | $4^2=16$ language self scores plus $np=32$ cross scores |
| Resampled external memory | $n=4$ | $mp=16$ resampler scores plus $4^2=16$ self and $nm=8$ cross scores |

These are dense score-entry counts before masking, not full FLOPs or wall-clock speed predictions; projections, encoder computation, depth and compression loss also matter. For external cross-attention, $Q=H W_Q\in\mathbb R^{n\times d_k}$ and $K=V W_K\in\mathbb R^{p\times d_k}$ come from different sequences; the value projection has $p$ rows too. The $n\times p$ read produces $n$ outputs returned to the language stream. Visual memory never becomes an additional language position in that mode.

Query compression is an upstream choice that can feed prefix fusion **or** cross-attention. BLIP-2 projects Q-Former outputs to the LLM input width and prepends them as soft visual prompts. Flamingo instead combines a learned-query resampler with external visual cross-attention whose queries come from language and whose keys/values come from vision. [2](#ref-flamingo) [3](#ref-blip)

## Inspect the interface
Switch the fusion mode in the vector figure. Count which tensors join the language sequence and which remain external memory. The displayed token counts are illustrative; actual models differ in image crops, resolution, resampling and temporal handling.

```python
visual = vision_encoder(images)
visual_tokens = projector(visual)
sequence = concatenate([visual_tokens, text_embeddings], axis='positions')
# Prefix example: V reads V only; text reads V and earlier/current text.
answer = decoder(sequence, mask=prefix_mask(visual_count=len(visual_tokens)))
```

## Evidence and limits
A captioning improvement does not establish precise spatial grounding. Frozen backbones reduce trainable parameters but do not remove the cost of their forward passes. Evaluation should vary image content while preserving the question, and include cases where language priors suggest the wrong answer. Continue to [training objectives](#/article/multimodal-objectives).
