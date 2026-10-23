## Generative pretraining
GPT's 2018 report used a Transformer-based language model for pretraining followed by supervised adaptation. The contribution is not simply “inventing next-token prediction”: the report combined a scalable generative objective with transfer to language understanding tasks. [1](#ref-gpt)

## Scaling as an empirical relationship
A convenient schematic fit is

$$
\mathcal L(N,D)\approx \mathcal L_\infty+aN^{-\alpha}+bD^{-\beta}.
$$

$\mathcal L$ denotes held-out language-model cross-entropy, evaluated for a specified tokenizer, data distribution and training setup; losses across incompatible tokenizations or evaluation mixtures are not directly comparable. $N$ is parameter count, $D$ training tokens, $a,b,\alpha,\beta>0$ fitted quantities and $\mathcal L_\infty$ an asymptotic loss term. This expresses a modeling choice, not a universal law with fixed constants. Training compute is often approximated as proportional to $ND$ for a specified dense model family. [2](#ref-scaling)

Kaplan and collaborators' compute-efficient prescription emphasized larger models with relatively fewer training tokens and stopping before convergence. Hoffmann and collaborators instead found that model size and training tokens should scale approximately equally as compute grows within their experiments: roughly doubling model size should be accompanied by doubling training tokens. Their 70B Chinchilla model used four times Gopher's training data at the same training-compute budget. These conclusions concern particular experiments and fits, not invariant coefficients for every future model. [2](#ref-scaling) [3](#ref-chinchilla) A rule learned under fixed training compute need not minimize total serving cost, and neither paper proves that more parameters alone guarantee reasoning ability. [3](#ref-chinchilla)

## A fixed-compute calculation
Write the compute constraint as $ND=C$ after absorbing a model-dependent proportionality constant into $C$. Substitution gives

$$
\mathcal L(N,C/N)=\mathcal L_\infty+aN^{-\alpha}+bC^{-\beta}N^\beta.
$$

Increasing $N$ reduces the capacity term while reducing available tokens and increasing the data term. Differentiating with respect to $\log N$ gives the interior balance

$$
\alpha aN^{-\alpha}=\beta bD^{-\beta},\qquad
N_* =\left(\frac{\alpha a}{\beta b}C^\beta\right)^{1/(\alpha+\beta)}.
$$

The interactive figure uses **synthetic normalized units**: $a=b=1$, $\alpha=\beta=0.5$, $\mathcal L_\infty=1$. At $C=64$, $(N,D)=(8,8)$ gives $L=1.707$; allocations $(2,32)$ or $(32,2)$ both give $L=1.884$. The equal optimum follows from these chosen symmetric terms; it is not a reproduction of either paper's fitted frontier. Changing the budget shifts the optimum and the entire curve. This illustration explains the allocation mechanism while keeping published conclusions separate.

## An architecture becomes a recipe
LLaMA is an important open research milestone combining a decoder with design choices such as RMSNorm, rotary positions and gated MLPs. It inherited many separate ideas. Instruction tuning and preference optimization then change how a pretrained model behaves, while leaving its high-level attention stack recognizable. [4](#ref-llama) [5](#ref-instruct)

## An evidence-aware timeline
The broader atlas timeline groups by first publication year; conference years can differ. The relevant anchors here are GPT (2018), Kaplan scaling (2020), Hoffmann compute allocation and instruction tuning (2022), and LLaMA (2023). Each changed a different part of the recipe. This article’s figure focuses on compute allocation; the broader timeline remains a curated set of anchors, not an exhaustive history of language modeling.

## Research questions
Evaluate scaling claims against data mixture, deduplication, contamination control, optimizer, tokenization and inference cost. A model can improve on a benchmark through better data or post-training without an architectural invention. Keep architecture and procedure as separate entries in an experiment table.
