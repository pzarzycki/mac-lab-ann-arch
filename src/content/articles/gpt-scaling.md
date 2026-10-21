## Generative pretraining
GPT's 2018 report used a Transformer-based language model for pretraining followed by supervised adaptation. The contribution is not simply “inventing next-token prediction”: the report combined a scalable generative objective with transfer to language understanding tasks. [1](#ref-gpt)

## Scaling as an empirical relationship
A convenient schematic fit is

$$
\mathcal L(N,D)\approx \mathcal L_\infty+aN^{-\alpha}+bD^{-\beta}.
$$

$N$ is parameter count, $D$ training tokens, $a,b,\alpha,\beta>0$ fitted quantities and $\mathcal L_\infty$ an asymptotic loss term. This expresses a modeling choice, not a universal law with fixed constants. Training compute is often approximated as proportional to $ND$ for a specified dense model family. [2](#ref-scaling)

Hoffmann and collaborators revisited compute allocation and found a different balance between model size and tokens under their experiments. A rule learned under fixed training compute need not minimize total serving cost, and neither paper proves that more parameters alone guarantee reasoning ability. [3](#ref-chinchilla)

## An architecture becomes a recipe
LLaMA is an important open research milestone combining a decoder with design choices such as RMSNorm, rotary positions and gated MLPs. It inherited many separate ideas. Instruction tuning and preference optimization then change how a pretrained model behaves, while leaving its high-level attention stack recognizable. [4](#ref-llama) [5](#ref-instruct)

## An evidence-aware timeline
The atlas timeline groups by first publication year; conference years can differ. Select a milestone to inspect the primary reference and its specific contribution. It is a curated set of anchors, not an exhaustive history of language modeling.

## Research questions
Evaluate scaling claims against data mixture, deduplication, contamination control, optimizer, tokenization and inference cost. A model can improve on a benchmark through better data or post-training without an architectural invention. Keep architecture and procedure as separate entries in an experiment table.
