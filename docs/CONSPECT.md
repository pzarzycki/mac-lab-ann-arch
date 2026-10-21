# Neural Atlas — conspect

An interactive research guide to artificial neural architectures, learning rules and embodied systems. Readers enter through a topic, inspect its essential distinctions, open a detailed account and follow connections in any direction. Topic order does not imply prerequisites, importance or a required reading sequence.

Source snapshot: **21 October 2026**. This edition contains **37 deep dives in 10 connected groups**, with **69 primary references**. All 33 original articles remain available at their original article URLs. JEPA, video world models, NEAT and HyperNEAT add four articles.

## The exploration model

1. **Aggregate:** a topic group names a question and the mechanisms it brings together.
2. **Preview:** each topic exposes a short abstract, three key distinctions, searchable technique tags, primary-source count and an optional quick view with the live visual.
3. **Deep dive:** a research-paper-style article develops the mechanism, mathematics, implementation considerations, evidence and limitations around an inspectable visual.
4. **Connect:** related-topic cards explain why a different article is relevant. Relationships cross the hierarchy: objective, architecture, execution method and application remain different axes.

The direct deep-dive link is always available. Quick views are optional. Question collections are unordered selections. There are no lessons, completion requirements, grades, prerequisite locks or global next/previous sequence. Bookmarks form a personal collection.

## Topic groups and implemented deep dives

| Group | Organizing question | Deep dives and visual focus |
| --- | --- | --- |
| **Transformer mechanics** | What happens inside a Transformer block? | **Residual stream:** separate token states, identity paths, pre/post-LN. **Attention:** queries, keys, values, masks and source reads. **Normalization, position & gated MLPs:** feature statistics, rotary geometry and gated projections. **Training, prefill & decoding:** parallel scoring, cached inference and sequential generation. |
| **Sequence architectures** | Where can a sequence model spend its computation? | **Mixture of experts:** router logits, top-k activation and expert weights. **Hyper-connections & mHC:** residual lanes within each token and constrained mixing. **Recurrent depth & looped Transformers:** repeated computation and a contraction example. **Selective state spaces & hybrids:** input-dependent recurrent state versus explicit token reads. |
| **Efficient execution** | What makes a model cheaper to run? | **KV-cache memory & grouped heads:** batch, layers, context, heads, precision and bytes. **Exact kernels, sparse patterns:** IO-efficient exact attention versus changed connectivity. **Compression & speculative decoding:** numeric precision, proposal generation and target-model verification. |
| **Multimodal & predictive representations** | What should a representation preserve or predict? | **Modality fusion:** prefix tokens, query resampling and cross-attention. **Alignment, generation & instruction:** objective differences. **Audio, video & temporal grounding:** sampling, synchronization and temporal evaluation. **JEPA:** context/target encoders, masks, feature targets, stop-gradient and EMA. **Video world models & latent planning:** action-free representations, action conditioning, candidate rollouts and replanning. |
| **Generation & latent computation** | What changes when generation leaves the token-by-token loop? | **Diffusion & flow-based generation:** generative construction versus illustrative reveal order. **Soft prompts & continuous thought:** learned input vectors, latent computation and soft-token mixtures as distinct mechanisms. |
| **Learning dynamics & credit assignment** | How does a learning system assign credit? | **Predictive coding:** activities, prediction errors and local inference. **Augmented Lagrangian predictive coding:** constraints, multipliers, penalty and a linear comparison with BP. **Active inference:** beliefs, generative models and policy selection. **Spiking neural networks:** membrane dynamics, thresholds and surrogate gradients. **Liquid time-constant networks:** continuous hidden dynamics and state-dependent response. **Backpropagation & biological plausibility:** reverse-mode differentiation and separate biological constraints. **Feedback alignment & local objectives:** alternative feedback and update locations. |
| **Neuroevolution & architecture search** | Can we evolve the network and its connectivity? | **NEAT:** connection genes, structural mutation, historical identifiers and speciation. **HyperNEAT:** an evolved CPPN maps coordinate pairs to a separate substrate network; inspect geometric symmetry and generated weights. |
| **Memory & adaptation** | What changes between interactions, and what persists? | **Remembering while learning:** replay, consolidation, isolation and forgetting. **Memory, context & test-time adaptation:** distinguish cache, retrieval, context and parameter updates. **Self-improvement needs a verifier:** data generation, selection, held-out feedback and update loops. |
| **Embodied intelligence** | How do representations become actions in a changing world? | **Vision–language–action models:** observations, instruction, robot state, action head and controller. **Tokens, diffusion & flow for action:** distributions, action chunks and receding-horizon execution. **Evaluate the closed loop:** latency, freshness, success and distribution shifts. |
| **Intellectual lineage** | Which earlier ideas made these mechanisms possible? | **Distributed representations:** identifiers, embeddings, relational prediction and language models. **Sequence models discover attention:** recurrent summaries, alignment and Transformer construction. **ResNet:** identity paths and learned transformations, with earlier skip/highway work acknowledged. **Transformer to language-model scaling:** generative transfer, empirical scaling fits and recipes combining independent inventions. |

These groups are browsing containers. JEPA is a predictive representation-learning arrangement and may use Transformers; diffusion is a generative construction; NEAT is population-based search; KV caching is an execution technique. Shared words such as “prediction,” “memory” and “sparsity” do not establish algorithmic equivalence.

## Questions across the hierarchy

The live directory offers four optional, unordered collections:

| Question | Connections |
| --- | --- |
| **Where does the computation go?** | Attention, MoE, recurrent depth, KV cache, efficient attention and state spaces: compare the attention graph, activated parameters, repeated depth, retained state and execution cost. |
| **What is the model predicting?** | Autoregressive tokens, JEPA targets, video features, diffusion/flow outputs, predictive-coding errors and robot actions: state the predicted variable and target space. |
| **What can change in a learning system?** | Backpropagation, PC-ALM activities/multipliers, NEAT genomes, HyperNEAT generators, continual-learning state and test-time parameters: separate updates and timescales. |
| **How does prediction become action?** | Multimodal perception, video world models, active inference, VLA policies, action distributions and closed-loop evaluation: predictive accuracy alone does not establish control success. |

## Connections to preserve

- **ResNet ↔ residual streams ↔ mHC:** identity paths, token states and multiple within-token lanes; never confuse lane count with head count.
- **Attention ↔ FlashAttention ↔ sparse attention:** operator, implementation and changed graph are different axes.
- **KV cache ↔ state-space state ↔ external memory ↔ test-time training:** identify what is stored and whether weights change.
- **JEPA ↔ multimodal objectives ↔ predictive coding:** compare target spaces and learning dynamics without implying one common algorithm.
- **JEPA ↔ video world models ↔ VLA:** separate masked representation learning, action-conditioned prediction, planning and direct policy generation.
- **NEAT ↔ HyperNEAT:** direct genes encode a network; an indirect CPPN generates task-network weights from geometry.
- **Neuroevolution ↔ backpropagation ↔ continual learning:** population search, derivative-based updates and lifetime adaptation can combine, but should not be conflated.
- **Diffusion/flow ↔ action generation:** related constructions can apply to different variables and constraints.
- **Looped Transformers ↔ latent reasoning ↔ state spaces:** recurrence across depth, hidden computation and recurrence across sequence time are different choices.
- **PC-ALM ↔ predictive coding ↔ backpropagation:** compare local equations and the exact scope of convergence/equivalence claims.
- **Self-improvement ↔ continual learning ↔ robotics evaluation:** independent criteria, resource budgets and retained capabilities belong in the evidence.

Every article records at least three explicit relationships with a short reason. These are editorial connections rather than similarity inferred only from keywords.

## Coverage register and expansion priorities

The current articles contain multiple techniques; a tagged mention is not automatically a complete survey. This register retains the larger ambition and identifies where dedicated treatment is warranted. Future topics must be researched and illustrated before they become navigable deep dives; no empty article substitutes for coverage.

| Area | Present treatment | Further deep dives to research |
| --- | --- | --- |
| Representations and language interfaces | Embeddings, token states, position, probabilistic language models | BPE, WordPiece, SentencePiece, tokenizer-free and byte-level models; vocabulary/output heads; tied embeddings; pooling and sentence representations. |
| Transformer families | Decoder mechanics; encoder–decoder ancestry; cross-attention | BERT/masked language modeling; T5/span corruption; encoder-only versus decoder-only versus encoder–decoder; bidirectional inference and sequence-to-sequence objectives. |
| Training and adaptation | Next-token loss, SFT, RLHF, DPO, scaling; adaptation overview | Data curation/deduplication; AdamW/optimizer dynamics; stability, precision and initialization; LoRA/QLoRA and adapters; distillation; preference optimization and post-training evaluation. |
| Efficient architectures and systems | MQA/GQA, MLA caveat, FlashAttention, sparse/windows, MoE, mHC, quantization, speculative decoding | MLA derivation; context extrapolation; distributed experts; parallelism and ZeRO; prefix sharing, cache eviction/quantization, prefill/decode disaggregation; pruning and adaptive compute. |
| Generative constructions | Autoregression, diffusion, flow matching; continuous computation | Score-based SDEs, rectified flows, discrete diffusion, energy-based models, normalizing flows, VAEs and consistency models; compare objectives and sampling semantics. |
| Reasoning and inference-time computation | Soft prompts, latent thought, loops and verifier caveats | Chain-of-thought, search, tool use/ReAct, process versus outcome verification and compute allocation; distinguish behavior from claims about internal reasoning. |
| Multimodal and predictive representations | Contrastive/generative/instruction objectives; fusion; audio/video; I-JEPA, V-JEPA, V-JEPA 2/2-AC | CLIP-style geometry; masked autoencoding versus JEPA; discrete/continuous modality tokenizers; speech/video objectives; grounding/retrieval; world-model uncertainty and intervention evaluation. |
| Recurrent and associative computation | Looped Transformers, selective SSMs and hybrids | Linear attention, retention/recurrent hybrids, RWKV-like designs, modern Hopfield networks and associative memory; explicit state-capacity tradeoffs. |
| Learning and biological constraints | BP, feedback alignment, local losses, equilibrium-propagation context, PC/PC-ALM, active inference, spikes, liquid dynamics | Detailed equilibrium propagation; contrastive Hebbian learning; target propagation; forward-forward; dendritic credit, three-factor plasticity, STDP; BPTT versus online rules; independent plausibility criteria. |
| Evolution and architecture discovery | NEAT, HyperNEAT, CPPNs; encoding and speciation | Evolution strategies, quality diversity/MAP-Elites, novelty search, gradient/RL architecture search; evolved plasticity, Baldwinian/Lamarckian variants and ES-HyperNEAT. |
| Memory and improvement | Replay/consolidation/isolation, retrieval/context/cache distinction, test-time updates, verifier loops | Task-free/online streams; episodic versus semantic memory; retrieval training; meta-learning; reset policies; model editing and self-modification under independent evaluation. |
| Embodied models and control | VLA, discrete/diffusion/flow action heads, action chunks, closed-loop evaluation and latent planning | World-model-based control, inverse dynamics, offline/online RL, imitation learning, morphology transfer, hierarchical policies, tactile/proprioceptive inputs, geometric representations, simulation-to-real. |
| Shared evaluation and interpretability | Per-article evidence, assumptions, latency and resource accounting | Contamination, calibration, mechanistic interpretability and causal intervention; learned representations versus faithful explanation; robust/causal generalization and reproducibility. |

The register is broader than the current implementation. It avoids claiming that 37 articles cover every important technique and prevents topics from disappearing during regrouping.

## Deep-dive editorial contract

Every implemented deep dive has an abstract, three concrete key points, technique tags and a visible diagram or computational visual. It develops a mechanism, defines mathematical symbols/dimensions where needed, links primary evidence, states limitations and offers implementation or evaluation considerations. Related-topic links include a reason.

Research ideas need not fit an identical subsection template. A paper-focused article can concentrate on derivation and assumptions; a family overview can compare constructions and route readers to specific accounts. Numerical examples distinguish analytical calculation, synthetic simulation and measured results.

## Visual and interaction contract

Use semantic native SVG with a viewBox, accessible title/description, theme variables and vector export. Detailed figures preserve distinct axes: sequence position, hidden width, heads, residual lanes, experts and time. No Mermaid or rasterized architecture diagrams.

Interaction answers a question: select a token, change a mask, inspect a component, compare targets, vary a formula, split a NEAT connection or query a geometric weight field. A figure must not imply a trained model is running.

Every deep dive has a visual. Shared renderers contain distinct graphs and captions. The cache calculator and source-linked historical timeline are computational/interactive visuals; the remaining articles expose vector diagrams. Vector figures support zoom, expansion and SVG export.

## Sources and maintenance

Article Markdown is the manuscript source. `articles.json` defines previews, tags, references, visuals and explained connections. `categories.json` and `collections.json` define browsing views. `sources.json` records primary URLs, years, review date and provenance. The generated manuscript reproduces article bodies, key points, relationships and references.

Claims retain their scope: linear versus nonlinear results, complexity versus runtime, action-free data versus interaction signals, theoretical versus empirical findings. Attribute liquid time-constant networks to Hasani and collaborators. Label the original Transformer separately from a modern pre-norm decoder.

Validate citation numbering, TeX, local links, visual assignments and the relationship graph. Check meaningful numerical invariants, render all visuals, inspect bright/dark and narrow layouts, and preserve shareable article URLs. See `PLAN.md` for implementation and `VALIDATION.md` for checks.
