import articleIndex from './articles.json';
import categories from './categories.json';
import sources from './sources.json';
const files = import.meta.glob('./articles/*.md', {query:'?raw',import:'default',eager:true});
export const articles = articleIndex.map(a=>({...a,body:files[`./articles/${a.slug}.md`]}));
export {categories,sources};
export const sourceMap=Object.fromEntries(sources.map(s=>[s.id,s]));
export const articleMap=Object.fromEntries(articles.map(a=>[a.slug,a]));
export function headingId(text){return text.toLowerCase().replace(/[^a-z0-9\s-]/g,'').trim().replace(/\s+/g,'-');}
export function headings(body){return [...body.matchAll(/^## (.+)$/gm)].map(m=>({label:m[1],id:headingId(m[1])}));}
export {default as collections} from './collections.json';
export const categoryAliases={advanced:'systems',paradigms:'generation',dynamics:'learning'};
export const glossary=[
 ['Token embedding','A learned continuous vector selected by a discrete vocabulary ID; distinct from a contextual hidden state.','token-embeddings'],
 ['Unembedding','The output projection from hidden width to vocabulary logits, usually after final normalization. It is not generally an inverse of input lookup.','token-embeddings'],
 ['Logit lens','Applying the model’s final normalization and output head to an intermediate state, without a fitted layer translator.','logit-tuned-lenses'],
 ['Tuned lens','A per-layer learned affine translator followed by the frozen final vocabulary decoder.','logit-tuned-lenses'],
 ['J-Lens / Jacobian lens','A vocabulary readout using a layer-specific transport constructed from averaged downstream Jacobians.','jacobian-lens'],
 ['Activation patching','Replacing an internal activation with a controlled cached alternative and rerunning the downstream model.','causal-features-circuits'],
 ['Transcoder','A sparse feature model approximating a component’s input-to-output mapping; distinct from reconstructing one activation with an SAE.','causal-features-circuits'],
 ['Residual stream','The token-indexed hidden state carried along residual paths. A standard Transformer retains one width-d vector per token.','residual-streams'],
 ['Hidden state','A vector h in a specified space such as Rᵈ. H often stacks many such token states into a matrix.','residual-streams'],
 ['Layer Normalization (LN)','Normalization across the features of each token independently, with learned feature-wise scale and offset.','normalization-position'],
 ['RMSNorm','Feature scaling by root mean square without LayerNorm’s mean subtraction.','normalization-position'],
 ['Query / key / value','Learned projections used to score and retrieve information in attention. A query belongs to a destination; keys and values to source positions.','attention'],
 ['Causal mask','A connectivity rule that prevents a next-token predictor from reading later input positions.','attention'],
 ['Prefill','Processing the prompt and constructing layer-wise keys and values before incremental generation.','training-decoding'],
 ['KV cache','Stored inference keys and values, usually per layer and past position; it is state rather than model parameters.','kv-cache'],
 ['GQA','Grouped-query attention: multiple query heads share each key/value head.','kv-cache'],
 ['FlashAttention','An IO-aware implementation of exact attention that avoids materializing the entire score matrix in HBM.','efficient-attention'],
 ['MoE','Mixture of experts: a router selects a subset of parameterized experts for each input.','mixture-of-experts'],
 ['mHC','Manifold-constrained hyper-connections: constrained mixing among residual lanes inside each token.','hyper-connections'],
 ['Doubly stochastic matrix','A nonnegative matrix whose rows and columns each sum to one.','hyper-connections'],
 ['Soft prompt','Learned continuous input vectors, often optimized while freezing a language backbone.','soft-latent-reasoning'],
 ['Latent reasoning','Extra computation on hidden representations rather than necessarily emitting human-readable reasoning tokens.','soft-latent-reasoning'],
 ['Flow matching','Learning a conditional velocity field for transport along a chosen probability path.','diffusion-flow'],
 ['Primal / dual','Primal activities are optimized state variables; dual multipliers accumulate constraint violations under a specified sign convention.','pc-alm'],
 ['PC-ALM','Augmented Lagrangian predictive coding, combining local activity descent with multiplier ascent.','pc-alm'],
 ['Variational free energy','A probabilistic functional that bounds negative log evidence; its units are not automatically physical energy.','active-inference'],
 ['Surrogate gradient','A proxy derivative used to train a computation such as a hard spiking threshold.','spiking-networks'],
 ['Continual learning','Learning across a changing stream while retaining previous useful behavior under stated resource limits.','continual-learning'],
 ['Test-time training','Parameter updates from a test-time self-supervised objective, with a specified reset policy.','memory-adaptation'],
 ['VLA','Vision–language–action model: a policy connecting visual/language conditioning to robot action representations.','vla-foundations'],
 ['JEPA','Joint-Embedding Predictive Architecture: predicts target representations from context rather than requiring reconstruction of observations.','jepa'],
 ['World model','A model of state or observation evolution. Planning additionally needs an action interface and an objective.','video-world-models'],
 ['NEAT','NeuroEvolution of Augmenting Topologies: evolution of direct network encodings using historical alignment, structural mutation and speciation.','neat'],
 ['HyperNEAT','An indirect encoding in which an evolved CPPN maps geometric coordinate pairs to task-network connections.','hyperneat'],
 ['CPPN','Compositional pattern-producing network: a function whose chosen activation functions can express spatial regularities.','hyperneat'],
 ['Action chunk','A predicted sequence of future actions; execution may use only a prefix before observing again.','action-models']
];
