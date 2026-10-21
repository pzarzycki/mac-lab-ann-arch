# Neural Atlas

An interactive research guide for individual exploration of neural architectures, learning algorithms and embodied intelligence.

**37 deep dives · 10 connected topic groups · 69 primary references.** Source snapshot: 21 October 2026.

Browse a group, inspect a concise topic preview and its live visual, open the detailed article, then follow explained connections to related ideas. Question collections offer alternative views of the same content. The interface has no lessons, required order, grades or completion tracking.

New coverage includes **JEPA/I-JEPA/V-JEPA, video world models and latent planning, NEAT and HyperNEAT**. Every deep dive has a visual; most use detailed, interactive native SVG. The cache calculator and historical timeline are computational/reference visuals. All 33 original article URLs remain valid.

## Run locally

Use Node.js 20.19+ or a compatible newer release:

```bash
npm ci
npm run dev
```

Open the URL Vite prints (normally port 4173). All interactions execute locally in the browser. No application backend, model API key, account system or remote font service is needed.

```bash
npm run check       # citations, links, previews, visuals, relationships, strict TeX
npm run check:visuals # render every actual visual; validate concept graph geometry
npm test            # numerical invariants and linear PC-ALM/BP comparison
npm run manuscript  # export the synchronized manuscript and bibliography
npm run build       # validated static production output in dist/
npm run preview     # serve production output locally
```

## Content and conspect

- `docs/CONSPECT.md`: topic map, all implemented deep dives, crosslinks and future coverage register.
- `docs/PLAN.md`: editorial decisions, implementation and maintenance.
- `docs/MANUSCRIPT.md`: generated article text, key points, related topics and primary references.
- `docs/VALIDATION.md`: verification evidence and known limits.
- `src/content/articles/*.md`: article bodies with TeX and highlighted fenced code.
- `src/content/articles.json`: abstracts, three key points, technique tags, citations, visuals and explained connections.
- `src/content/categories.json`: topic groups and their connections.
- `src/content/collections.json`: unordered question-based exploration collections.
- `src/content/sources.json`: primary bibliography and review provenance.
- `src/components/Explore.jsx`: topic cards, quick views, directory and related-topic navigation.
- `src/components/*Figures.jsx`: semantic SVG diagrams and interactive calculations.
- `src/content/visuals.json`: registry connecting article visual IDs to renderer families.
- `src/math.js`: numerical calculations; see `tests/math.test.mjs`.

Add Markdown and metadata together. Declare references in their numbered order; inline `[1](#ref-source-id)` citations are validated. Article links use `#/article/slug`. Each article must have three preview points, tags, a resolved visual and at least three explained related-topic links. Run the checks and manuscript export before publication.

The conspect distinguishes existing treatment from topics awaiting further research. A mention or tag does not claim a complete survey, and future coverage is not represented by empty articles.

## Rendering and navigation

React/Vite; Markdown through react-markdown, GFM and math plugins; KaTeX HTML/MathML; rehype-highlight for code. SVG figures support inspection, zoom, expansion and vector export. Diagrams preserve separate token streams and distinguish heads, residual lanes, experts and time. Bright/dark/system themes and bookmarks use browser-local storage.

Search covers titles, technique tags and article text. The directory can filter by topic group and cross-field question. Related-topic links explain their relationship instead of imposing a global previous/next order. Old article URLs remain stable; old category URLs and `#/paths` have compatible aliases.

Illustrative masks, mutations, coordinate fields and latent rollouts are labeled as such. These interactions are not trained models, reproduced benchmark runs or robot controllers. Numerical calculations and paper evidence retain their separate assumptions.

## Static deployment

`npm run build` emits `dist/`. Publish that directory to any static host. Vite uses relative asset paths, including under a subdirectory. Hash routes do not require a server-side fallback. No runtime secrets are required. `.openai/hosting.json` associates the existing private Site; it is deployment metadata, not an application backend.

## GitHub

Repository: https://github.com/pzarzycki/mac-lab-artficial-neural-archictectures

```bash
git clone git@github.com:pzarzycki/mac-lab-artficial-neural-archictectures.git
cd mac-lab-artficial-neural-archictectures
npm ci
npm run dev
```

The repository contains editable source, the conspect, generated manuscript and bibliography. Build output and dependencies are excluded from Git.
