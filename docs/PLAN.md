# Neural Atlas — implementation and editorial plan

The authoritative topic map is [CONSPECT.md](CONSPECT.md). Neural Atlas is an exploration guide for advanced readers: topic groups, optional previews, deep dives and explained connections. The data model does not impose lessons, completion tracking or a reading sequence.

## Implemented structure

- Ten browsing groups aggregate 37 articles; all original article URLs remain stable.
- Topic cards expose an abstract, three key points, tags, quick view and direct deep-dive link.
- Quick views contain the article's live visual. Articles develop mechanisms, mathematics, evidence and limits, then offer related topics with explicit relationship labels.
- The exploration directory supports topic/technique search, group filtering and four unordered question collections. Legacy `#/paths` URLs resolve to this directory.
- Bookmarks, full-text search, glossary, source library and BibTeX export support individual research.
- The conspect distinguishes implemented treatment from topics awaiting deeper research.

## Representation and research decisions

Use original papers, author material, official code and publisher records. A paper label does not imply peer review. First arXiv and conference years can differ.

ResNet is a direct architectural influence; earlier skip connections and Highway Networks are acknowledged. A conventional Transformer has one residual stream per token. Attention heads, mHC lanes and sequence positions are separate axes. Define hidden vectors in a space such as R^d. LN means Layer Normalization across each token's features; distinguish RMSNorm. Original post-LN encoder–decoder and modern pre-norm decoder variants are labeled separately.

Distinguish JEPA feature prediction, pixel reconstruction, predictive coding and action-conditioned world models. Distinguish NEAT's genotype, HyperNEAT's CPPN and the generated task substrate. Evolution and lifetime plasticity operate on different timescales. PC-ALM's linear results and nonlinear empirical evidence retain their separate scope. Soft prompts, latent reasoning and soft-token mixtures remain distinct.

## Technical design

React + Vite emits portable static `dist/` output. Hash routes support static hosts and subdirectories. No model API or application backend is required.

Markdown uses GFM, math plugins, KaTeX HTML/MathML and highlighted fenced code. Fonts/assets are bundled locally. Each deep dive has a semantic interactive SVG or computational visual. SVG export preserves computed theme styles. Inspectable graphs share a renderer with separate mechanism data.

Bright/dark/system themes and bookmarks persist locally. Responsive navigation, paper TOC, search/preview dialogs, reduced-motion behavior and print styles support research use.

## Content workflow

1. Research primary sources and record provenance.
2. Author the article and concise preview together.
3. Draw the mechanism and expose a meaningful interaction.
4. Add searchable tags and at least three explained related-topic links.
5. Validate citations, strict TeX, links, groups, visuals and relationships.
6. Export the synchronized manuscript and bibliography.
7. Check numerical invariants, render every visual, build production assets and inspect browser behavior/layout.
8. Update the supplied GitHub repository and existing private Site.

`VALIDATION.md` records evidence and limits. The conspect remains the coverage map; `MANUSCRIPT.md` is generated from the exact article sources.
