import React, {useId, useState} from 'react';
import {Figure, ArrowDefs} from './Figures';

// Each graph depicts the mechanism described in its article. Selection exposes
// assumptions and local meaning; it does not claim to execute a trained model.
const C = {input:'var(--blue)',state:'var(--green)',update:'var(--orange)',base:'var(--fig-line)'};
const n=(id,x,y,label,sub,detail,kind='base',w=196,h=56)=>({id,x,y,label,sub,detail,kind,w,h});
const e=(from,to,fromSide='right',toSide='left',extra={})=>({from,to,fromSide,toSide,...extra});
const p=(x,y,w,h,label)=>({x,y,w,h,label});

export const conceptDiagrams = {
 'normalization-position': {
  title:'Three operations, three different roles',height:514,
  caption:'Independent constructions within a modern block: feature normalization, query/key position geometry, and a gated position-wise MLP. The columns are not consecutive layers.',
  panels:[p(16,20,260,464,'FEATURE SCALE · PER TOKEN'),p(290,20,260,464,'POSITION · QUERY / KEY'),p(564,20,260,464,'GATING · PER TOKEN')],
  nodes:[
   n('nx',146,65,'xᵢ ∈ ℝᵈ','one token’s features','The normalization statistics are computed across this token’s features, independently of other tokens.','input',218),
   n('norm',146,171,'LN or RMSNorm','centering differs','LayerNorm subtracts the feature mean; RMSNorm rescales by root mean square without that mean subtraction. Both have learned feature scales.','state',218),
   n('normalized',146,277,'Normalized features','same token, same width','Normalization changes feature scale. It does not mix information between sequence positions.','state',218),
   n('normrole',146,383,'Scale control','placement is a separate choice','Pre-norm versus post-norm describes where normalization appears relative to a residual addition.','base',218),
   n('qk',420,65,'qᵢ = Wq xᵢ · kⱼ = Wk xⱼ','project token states','The positions i and j index sequence positions; q and k contain feature coordinates.','input',218),
   n('rotation',420,171,'q′ᵢ = Rᵢqᵢ · k′ⱼ = Rⱼkⱼ','rotate coordinate pairs','RoPE rotates pairs of query and key coordinates using their sequence positions. It generally does not add a positional vector to the residual stream.','state',218),
   n('score',420,277,'qᵢᵀ Rⱼ₋ᵢ kⱼ','relative displacement','For a compatible rotation family, the attention dot product depends on relative position through R(j−i).','state',218),
   n('posrole',420,383,'Ordering information','no length guarantee','An algebraic dependence on relative position alone does not establish arbitrary-length extrapolation.','base',218),
   n('mx',694,65,'xᵢ → Wg xᵢ and Wu xᵢ','parallel feature projections','The gate and value projections expand one token’s features. Parameters are shared across positions.','input',218),
   n('gate',694,171,'SiLU(Wg xᵢ) ⊙ Wu xᵢ','elementwise gated product','SwiGLU multiplies a nonlinear gate by a second feature projection. This is not an attention operation.','state',218),
   n('down',694,277,'Wo [gated features]','project back to width d','The output projection restores the residual width so an equal-shaped residual addition is possible.','state',218),
   n('mlprole',694,383,'Feature transformation','no sequence mixing','Every token is transformed separately by the same MLP weights; cross-token communication is supplied elsewhere.','base',218)
  ],
  edges:[e('nx','norm','bottom','top'),e('norm','normalized','bottom','top'),e('normalized','normrole','bottom','top'),e('qk','rotation','bottom','top'),e('rotation','score','bottom','top'),e('score','posrole','bottom','top'),e('mx','gate','bottom','top'),e('gate','down','bottom','top'),e('down','mlprole','bottom','top')]
 },
 'training-decoding': {
  title:'Known targets versus unknown next tokens',height:470,
  caption:'Training can score shifted targets in parallel under a causal mask. Prompt prefill builds state; generation then depends on newly sampled tokens. Weights θ are fixed during these inference phases.',
  panels:[p(18,20,390,421,'TEACHER-FORCED TRAINING'),p(432,20,390,421,'INFERENCE · FIXED WEIGHTS')],
  nodes:[
   n('input',213,64,'Known tokens x₁ … xₙ₋₁','shifted inputs','The full training example is available. The predictor at position t must not read its target x(t+1).','input',306),
   n('causal',213,164,'Causally masked decoder','many positions in one pass','A causal mask permits earlier input positions while blocking later ones. Parallel computation does not expose future labels.','state',306),
   n('loss',213,264,'Targets x₂ … xₙ','sum masked token losses','Logits from one position are compared with the following token. Padding and non-scored instruction spans require a loss mask.','update',306),
   n('grad',213,364,'Backward → optimizer step','updates θ using a batch','An optimizer step usually aggregates many token losses, and may accumulate several microbatches.','update',306),
   n('prompt',627,64,'Prompt x₁ … xₘ','prefill together','Prefill evaluates known prompt positions and constructs the layer-wise key/value cache.','input',306),
   n('cache',627,164,'Cached K/V + last logits','sample xₘ₊₁','Cache entries are inference state, not trainable model parameters. The next token is unknown until a distribution is evaluated.','state',306),
   n('decode',627,264,'Evaluate new position','append K/V; sample next token','Incremental decoding evaluates the new position against past keys and values. Sampling creates the input to the following step.','state',306),
   n('stop',627,364,'Repeat or stop','generation is conditional','Caching reuses earlier computations; it does not remove dependence on previously generated tokens.','base',306)
  ],
  edges:[e('input','causal','bottom','top'),e('causal','loss','bottom','top'),e('loss','grad','bottom','top'),e('prompt','cache','bottom','top'),e('cache','decode','bottom','top'),e('decode','stop','bottom','top'),e('stop','decode','right','right',{d:'M780 392 H805 V292 H780',dashed:true})]
 },
 'distributed-representations': {
  title:'From discrete identity to shared learned features',height:445,
  caption:'A modern conceptual view of relational learning and neural language modeling, rather than a reconstruction of the original 1986 circuitry. Representation, learning rule, and probability factorization remain distinct.',
  nodes:[
   n('identity',142,62,'Entity identifier i','eᵢ ∈ {0,1}ⱽ','An identifier is categorical. A one-hot vector selects a column of an embedding table.','input',210),
   n('embed',420,62,'Embedding table E','vᵢ = E eᵢ ∈ ℝᵈ','Learning E makes continuous features available to multiple examples; the underlying entity remains discrete.','state',210),
   n('hidden',698,62,'Shared hidden computation','reuse features across facts','Shared hidden features can capture regularities across entities and relations. Their geometry is not automatically symbolic logic.','state',210),
   n('relation',142,222,'Person + relation','a relation-completion query','A relational task asks which entity fills a missing role given a known entity and a relation.','input',210),
   n('relout',420,222,'p(person₂ | person₁, relation)','a distribution over entities','The formula abstracts the task. Holding out combinations is necessary to distinguish generalization from fact memorization.','state',236),
   n('factloss',698,222,'Relational prediction loss','learn from known facts','A loss shapes representations through the selected credit-assignment algorithm; the loss and architecture are separate choices.','update',210),
   n('words',142,352,'Context word identifiers','lookup → concatenate','Neural probabilistic language models reused learned word features in sequence prediction.','input',210),
   n('language',420,352,'Shared nonlinear model','p(next word | context)','The 2003 neural probabilistic language model connected word representation learning and sequence probabilities. It was not a Transformer.','state',236),
   n('wordloss',698,352,'Next-word likelihood','learn word features jointly','Language-model training couples the embedding table and predictor through the next-word objective.','update',210)
  ],
  edges:[e('identity','embed'),e('embed','hidden'),e('relation','relout'),e('relout','factloss'),e('words','language'),e('language','wordloss'),e('embed','relout','bottom','top',{dashed:true}),e('hidden','factloss','bottom','top',{dashed:true})]
 },
 'sequence-attention-history': {
  title:'The read operation that changed sequence modeling',height:485,
  caption:'Mechanism comparison: a fixed summary bottleneck, learned encoder–decoder alignment, and Transformer attention. Arrows describe computation, not a complete priority or citation graph.',
  panels:[p(16,22,808,126,'FIXED-SUMMARY ENCODER–DECODER'),p(16,170,808,126,'ADDITIVE ENCODER–DECODER ATTENTION'),p(16,318,808,142,'TRANSFORMER CONSTRUCTION')],
  nodes:[
   n('encoder',146,72,'Input recurrence','ordered encoder states','A recurrent encoder processes source positions in sequence. A fixed-summary architecture relies on a compressed state.','input'),
   n('summary',420,72,'Final summary state','fixed-width bottleneck','A single summary must retain information relevant to every subsequent output step. This creates a different memory organization from explicit reads over states.','state'),
   n('decoder',694,72,'Recurrent decoder','generate from summary','The output recurrence conditions on its previous state and the available source summary.','base'),
   n('states',146,220,'All encoder states hⱼ','retain source positions','Additive attention gives the decoder access to a collection of source-position states rather than only the final summary.','input'),
   n('align',420,220,'eₜⱼ → softmaxⱼ → αₜⱼ','query comes from decoder state','The alignment score compares the previous decoder state with each encoder state through learned projections and a nonlinearity.','state',226),
   n('context',694,220,'cₜ = Σⱼ αₜⱼ hⱼ','input-dependent source read','The weighted context is a differentiable read. The alignment is not a hard symbolic correspondence.','state'),
   n('self',146,368,'Multi-head self-attention','position-aware token states','The Transformer makes attention central to the encoder and decoder while supplying position information separately.','input'),
   n('cross',420,368,'Original decoder: self + cross','masked self; encoder read','The original translation decoder included causal self-attention and cross-attention to the encoder. Decoder-only GPT omits that translation encoder.','state',240),
   n('block',694,368,'Residual + norm + MLP','parallel teacher forcing','Attention alone is not the full Transformer: residual paths, normalization and position-wise networks are essential components.','base')
  ],
  edges:[e('encoder','summary'),e('summary','decoder'),e('states','align'),e('align','context'),e('self','cross'),e('cross','block')]
 },
 'resnet-lineage': {
  title:'A forward bypass and a direct derivative route',height:432,
  caption:'Equal-width identity residual block. A projection skip replaces I by P in the derivative. The identity term helps route gradients but does not guarantee stability or eliminate accumulated scaling.',
  nodes:[
   n('x',125,155,'Input x','equal input/output width','An identity skip requires compatible tensor shapes. Otherwise the bypass must be a projection or another explicit map.','input',164),
   n('f',420,66,'Residual branch Fθ(x)','learn a correction','The learned branch transforms x while the identity route preserves x itself.','state',234),
   n('sum',705,155,'y = x + Fθ(x)','elementwise addition','The two routes meet at an equal-shaped sum. In a Transformer the residual state remains separate for each token.','state',218),
   n('dy',705,322,'Upstream gradient gᵧ','∂L / ∂y','Reverse-mode differentiation starts with a derivative supplied by downstream computation.','update',218),
   n('jf',420,256,'Jꜰ(x)ᵀ gᵧ','branch derivative','The residual function contributes its Jacobian-transpose action to the input gradient.','update',234),
   n('dx',125,322,'gₓ = gᵧ + Jꜰᵀgᵧ','identity term + branch term','A direct derivative term remains even when the branch Jacobian is small. It is one term in a sum, not a proof that gradients cannot explode or cancel.','update',218)
  ],
  edges:[e('x','f','top','left',{d:'M125 155 V94 H303'}),e('f','sum','right','top',{d:'M537 94 H705 V155'}),e('x','sum','right','left',{label:'IDENTITY · I',lx:420,ly:171}),e('sum','dy','bottom','top',{dashed:true}),e('dy','jf','left','right',{d:'M596 350 H566 V284 H537'}),e('jf','dx','left','top',{d:'M303 284 H125 V322'}),e('dy','dx','left','right',{label:'DIRECT DERIVATIVE',lx:420,ly:373,dashed:true})]
 },
 'compression-decoding': {
  title:'Compress numbers or speculate on tokens',height:464,
  caption:'These techniques change different parts of inference. Quantization approximates numerical values; properly corrected speculative sampling preserves the target distribution under the stated algorithm’s assumptions.',
  panels:[p(16,20,808,158,'QUANTIZATION · NUMERICAL REPRESENTATION'),p(16,200,808,236,'SPECULATIVE SAMPLING · COMPUTATION SCHEDULE')],
  nodes:[
   n('weight',142,74,'Weight w + calibration','choose grouping and scales','Calibration, outliers and group size affect quantization error. The nominal number of bits alone does not establish quality.','input',218),
   n('code',420,74,'q = clip(round(w/s) + z)','integer code + metadata','Codes, scales, zero points and packing all contribute to storage. GPTQ chooses codes using additional second-order approximations.','state',238),
   n('approx',698,74,'ŵ = s(q − z)','approximate value in computation','Weight approximation may alter output distributions. Hardware kernels determine whether fewer bits translate into lower latency.','state',218),
   n('draft',142,250,'Draft q proposes tokens','several candidate positions','The draft adds computation. Its proposal length and agreement with the target influence the potential speed gain.','input',218),
   n('target',420,250,'Target p verifies together','condition on accepted prefix','The target evaluates proposed positions in parallel, but acceptance still follows the correctly conditioned prefix.','state',238),
   n('accept',698,250,'Accept with min(1, p(x)/q(x))','sequential acceptance checks','An agreement of argmax outputs alone is not the exact distribution-preserving sampling algorithm.','state',238),
   n('correct',698,356,'After rejection: [p − q]₊','normalize correction distribution','After the first rejection, sample the correction from the positive part of p−q, normalized over tokens, following the complete algorithm.','update',238),
   n('continue',420,356,'Append accepted/corrected output','start the next draft cycle','The complete procedure also handles the extra target token after a fully accepted proposal. This diagram summarizes the acceptance/correction logic.','base',254)
  ],
  edges:[e('weight','code'),e('code','approx'),e('draft','target'),e('target','accept'),e('accept','correct','bottom','top'),e('correct','continue','left','right'),e('continue','draft','left','bottom',{d:'M293 384 H142 V306',dashed:true})]
 },
 'multimodal-objectives': {
  title:'The objective defines the required evidence',height:462,
  caption:'Three compatible but distinct training constructions. A fusion interface does not determine its loss, and an instruction-following answer does not by itself prove visual grounding.',
  panels:[p(16,20,260,414,'CONTRASTIVE ALIGNMENT'),p(290,20,260,414,'CONDITIONAL GENERATION'),p(564,20,260,414,'VISUAL INSTRUCTION TUNING')],
  nodes:[
   n('pair',146,65,'Paired images and text','batch of B corresponding pairs','Each image has a corresponding text example; other batch entries provide competing matches in the contrastive classification problem.','input',220),
   n('dual',146,171,'Separate encoders → vᵢ, tⱼ','normalized embedding outputs','Image and text encoders map inputs into a space where a similarity can compare candidate pairs.','state',220),
   n('sim',146,277,'Similarity matrix s(vᵢ,tⱼ)/τ','diagonal pairs are positives','Temperature rescales comparisons. A symmetric objective also classifies images from text; false negatives can affect training.','state',220),
   n('closs',146,373,'Symmetric contrastive loss','retrieval / alignment signal','This objective asks for paired-embedding discrimination, rather than a next-word distribution.','update',220,42),
   n('visual',420,65,'Visual features + prefix text','condition on observed evidence','Conditional generation supplies visual information and a previously observed text prefix to a language predictor.','input',220),
   n('gen',420,171,'Fusion / query bridge','interface chosen separately','A prefix, learned-query bridge or cross-attention can be paired with different losses. Frozen and trainable components must be specified.','state',220),
   n('token',420,277,'p(yₜ | y<ₜ, visual input)','predict answer tokens','The generative model outputs a categorical distribution over the next token.','state',220),
   n('gloss',420,373,'Conditional token loss','score selected text spans','The scored spans and training stage matter; not every observed input token must contribute to the loss.','update',220,42),
   n('inst',694,65,'Image + instruction + answer','curated or synthetic examples','Instruction data define a task and a desired answer format in addition to providing a visual input.','input',220),
   n('assistant',694,171,'Vision–language assistant','retain access to visual features','Language priors can still answer some questions without using the image, so visual access alone does not establish grounding.','state',220),
   n('answer',694,277,'Score assistant completions','instruction-specific behavior','Answer masking shapes the desired assistant behavior. This can coexist with earlier contrastive or generative training stages.','state',220),
   n('ground',694,373,'Counterfactual evaluation','test image dependence','Changing answer-relevant visual evidence or using a language-only baseline helps reveal reliance on dataset priors.','base',220,42)
  ],
  edges:[e('pair','dual','bottom','top'),e('dual','sim','bottom','top'),e('sim','closs','bottom','top'),e('visual','gen','bottom','top'),e('gen','token','bottom','top'),e('token','gloss','bottom','top'),e('inst','assistant','bottom','top'),e('assistant','answer','bottom','top'),e('answer','ground','bottom','top')]
 },
 'temporal-modalities': {
  title:'Time sampling determines what can be known',height:449,
  caption:'Illustrative synchronized video/audio interface. Sampling, temporal position and clocks are separate design choices. No later decoder can reconstruct an event that the frontend never observed.',
  nodes:[
   n('video',140,65,'Video over duration T','continuous scene → sampled frames','Frames per second determine which motion and state changes can be observed. Spatial detail is controlled by resolution and patching.','input',218),
   n('sample',420,65,'Sample at timestamps t₁ … tₘ','approximately T × f frames','T seconds at f frames per second produces approximately T×f frames, subject to boundary and sampling conventions.','state',240),
   n('patch',700,65,'p spatial tokens per frame','nᵥ ≈ T × f × p','A ten-second clip with two frames per second and 196 tokens per frame yields 3,920 visual tokens before compression.','state',218),
   n('audio',140,214,'Audio waveform','time-indexed signal','Audio frontends discretize and summarize windows. Their stride and receptive field also impose temporal information limits.','input',218),
   n('windows',420,214,'Feature windows + timestamps','clock calibration / missing spans','Timestamps identify intended alignment; they do not repair an inaccurate clock or recover dropped observations.','state',240),
   n('merge',700,214,'Temporal representation + masks','align modalities and access','Spatial and temporal positions serve different roles. Attention masks should preserve the intended temporal and cross-modal information access.','state',218),
   n('tests',420,355,'Controlled order / offset tests','shuffle frames; shift audio','Event order, duration and audiovisual correspondence require tests beyond a plausible clip caption.','base',240),
   n('output',700,355,'Grounded output or policy','report sensor age and latency','For action, the physical world continues to evolve while the observation travels through the inference pipeline.','base',218)
  ],
  edges:[e('video','sample'),e('sample','patch'),e('audio','windows'),e('windows','merge'),e('patch','merge','bottom','top'),e('merge','output','bottom','top'),e('tests','output')]
 },
 'soft-latent-reasoning': {
  title:'Three continuous-vector constructions',height:459,
  caption:'The same vector width does not make these mechanisms equivalent. Fixed learned prompts, convex embedding mixtures, and input-dependent hidden-state feedback have different constraints and persistence.',
  panels:[p(16,20,260,410,'SOFT PROMPT · TASK PARAMETERS'),p(290,20,260,410,'SOFT TOKEN · SIMPLEX MIXTURE'),p(564,20,260,410,'LATENT STEP · INFERENCE STATE')],
  nodes:[
   n('prompt',146,68,'Learn P ∈ ℝᵐˣᵈ','optimize prompt vectors','Prompt tuning learns continuous task parameters while keeping the language backbone fixed. The vectors need not be vocabulary embeddings.','update',220),
   n('prepend',146,181,'[P ; E(input tokens)]','prepend to embedded input','The prompt is concatenated with token embeddings along the sequence dimension. It remains a learned task parameter across inputs.','input',220),
   n('fixed',146,294,'Frozen language backbone','prediction loss updates P','Gradient flow can reach prompt parameters without updating backbone weights. Saved prompts persist, unlike temporary activations.','state',220),
   n('dist',420,68,'p ∈ Δⱽ⁻¹','pₖ ≥ 0; Σₖ pₖ = 1','A simplex distribution is constrained to nonnegative coefficients summing to one. This assumption is stronger than being an arbitrary continuous vector.','input',220),
   n('mix',420,181,'v = Σₖ pₖ Eₖ','convex mixture of embeddings','The mixture lies in the convex hull of the vocabulary embeddings; an unrestricted soft prompt vector need not.','state',220),
   n('mixinput',420,294,'Use v as an input embedding','a representation option','This is a possible mathematical representation, not a claim that every continuous-reasoning model uses vocabulary probabilities.','base',220),
   n('context',694,68,'Context → last hidden state','input-dependent h','A hidden state is produced by the current input and model computation, rather than optimized as one fixed task prompt.','input',220),
   n('feedback',694,181,'Feed h back as next embedding','preserve context / cache layout','Coconut uses a previous hidden state as a subsequent input embedding during latent reasoning. The full context and cache handling still matter.','state',220),
   n('latent',694,294,'Additional latent computation','then resume token output','Latent steps consume computation even when they emit no readable tokens; their vectors are not automatically interpretable distributions over thoughts.','state',220)
  ],
  edges:[e('prompt','prepend','bottom','top'),e('prepend','fixed','bottom','top'),e('dist','mix','bottom','top'),e('mix','mixinput','bottom','top'),e('context','feedback','bottom','top'),e('feedback','latent','bottom','top'),e('latent','feedback','right','right',{d:'M804 322 H816 V209 H804',dashed:true})]
 },
 'state-space': {
  title:'A bounded recurrent state instead of explicit history',height:454,
  caption:'A conceptual selective state-space update, not an arbitrary dense implementation of Mamba. For a fixed model, state size is independent of sequence length; this compression is not lossless token retrieval.',
  panels:[p(16,22,808,258,'SELECTIVE RECURRENCE · TOKEN t'),p(16,302,808,122,'MEMORY ORGANIZATION · COMPARE THE RETAINED STATE')],
  nodes:[
   n('xt',142,69,'Current token xₜ','one new sequence position','The current input influences both the state drive and, in a selective model, some effective recurrence parameters.','input',216),
   n('params',420,69,'Input-dependent parameters','Āₜ, B̄ₜ, Cₜ','Selection changes what is retained, written and read. Concrete architectures use structured parameterizations and efficient scan algorithms.','state',236),
   n('prior',142,190,'Previous state sₜ₋₁','compressed history','The recurrent state summarizes history in a fixed number of state variables rather than keeping a separate record for every past token.','input',216),
   n('update',420,190,'sₜ = Āₜ sₜ₋₁ + B̄ₜ xₜ','retain + input-dependent write','The updated state combines transformed past state with the current input drive. Shapes and channel structure depend on the implementation.','state',236),
   n('read',698,190,'yₜ = Cₜ sₜ','read updated state','The output projects the updated state. A bounded state is not a general guarantee of exact recall at unlimited delays.','state',216),
   n('ssm',190,351,'Recurrent state','fixed model → fixed state size','State-space memory does not grow by storing one key and value for every earlier position.','base',290),
   n('kv',650,351,'Attention K/V memory','explicit records per cached position','Full-context attention retains token-indexed keys and values. Hybrid layers can combine both organizations with their associated costs.','base',290)
  ],
  edges:[e('xt','params'),e('xt','update','bottom','left',{d:'M142 125 V157 H276 V218 H302'}),e('params','update','bottom','top'),e('prior','update'),e('update','read'),e('read','prior','bottom','bottom',{d:'M698 246 V268 H142 V246',dashed:true,label:'NEXT TOKEN: sₜ BECOMES PRIOR STATE',lx:420,ly:264})]
 },
 'active-inference': {
  title:'Belief updating and policy-dependent futures',height:460,
  caption:'An abstract active-inference loop. Variational free energy F for current observations differs from expected free energy G for candidate policies. Specify the probabilistic model, preferences and approximation before choosing an implementation.',
  nodes:[
   n('world',142,62,'Environment hidden state','not directly observed','A partially observable environment produces observations and changes in response to action. Hidden state is a modelled variable, not necessarily a neural activation.','input',218),
   n('obs',420,62,'Observation o','likelihood p(o | s)','A likelihood model links hidden causes to sensory data. Inaccurate observation models change the inferred beliefs.','input',218),
   n('belief',698,62,'Infer q(s)','minimize current F[q]','Variational free energy bounds negative log evidence. Its minimization improves an approximate posterior for the current observations under the model.','state',218),
   n('model',698,218,'Predict under policies π','state transitions + observations','Candidate policies are evaluated through a generative model of possible future states and observations.','state',218),
   n('prefs',420,218,'Prior preferences + uncertainty','define desired / informative futures','Preferences and beliefs enter the evaluation of possible futures. The exact expected-free-energy decomposition depends on model and convention.','input',240),
   n('policy',698,360,'Evaluate G(π) → select policy','not the same functional as F[q]','Expected free energy evaluates policies and futures; substituting current-observation F for it erases the action-selection assumptions.','state',240),
   n('action',142,360,'Execute an action','new observation closes the loop','Action changes the environment and therefore the observations available at the next update. The diagram is not a complete robot controller.','update',218),
   n('check',420,360,'Compare control baselines','match information and model access','A Bayesian filter or conventional controller is a useful comparator when it receives the same knowledge and observations.','base',240)
  ],
  edges:[e('world','obs'),e('obs','belief'),e('belief','model','bottom','top'),e('prefs','model'),e('model','policy','bottom','top'),e('prefs','policy','bottom','left',{d:'M420 274 V328 H550 V388 H578'}),e('policy','action','bottom','bottom',{d:'M698 416 V442 H142 V416'}),e('action','world','top','bottom',{d:'M142 360 V118'})]
 },
 'spiking-networks': {
  title:'Forward events and a separate backward approximation',height:453,
  caption:'One discrete-time leaky integrate-and-fire convention. The forward path uses a hard spike; a surrogate derivative changes the backward rule. Temporal credit can still use backpropagation through time.',
  panels:[p(16,20,808,258,'FORWARD DYNAMICS · STATE AND EVENT'),p(16,301,808,122,'BACKWARD CREDIT · DIFFERENT MATHEMATICAL RULE')],
  nodes:[
   n('drive',142,68,'Input drive Iₜ','weighted incoming activity','Input currents drive the membrane state. A sparse event representation does not itself specify a learning algorithm.','input',218),
   n('membrane',420,68,'Current membrane uₜ','state before the next integration','The current membrane potential determines z(t). The update below then advances the state to t+1.','state',252),
   n('spike',698,68,'zₜ = 1[uₜ ≥ ϑ]','binary forward event','The hard threshold emits a discrete spike according to the current membrane potential. Its derivative is zero almost everywhere.','state',218),
   n('state',420,194,'uₜ₊₁ = βuₜ + Iₜ − ϑzₜ','leak + input drive − reset','The current spike supplies a subtractive reset in this chosen convention. Other models use a different reset and integration order.','state',252),
   n('bptt',142,350,'Unroll temporal computation','store or recompute needed states','Backpropagation through time may still transport a global error through many simulated steps. Event activity does not guarantee local credit.','update',218),
   n('surrogate',420,350,'Smooth proxy for ∂z/∂u','used only during backward credit','A surrogate-gradient method preserves the hard forward spike while substituting a chosen smooth derivative in the backward computation.','update',252),
   n('learn',698,350,'Compute parameter updates','compare proxy and task behavior','Training behavior depends on the surrogate, temporal dynamics and optimization. Hardware energy savings require separate measurement.','update',218)
  ],
  edges:[e('membrane','spike'),e('drive','state','bottom','left',{d:'M142 124 V222 H294'}),e('membrane','state','bottom','top'),e('spike','state','bottom','right',{d:'M698 124 V222 H546'}),e('state','membrane','bottom','top',{d:'M420 250 V267 H815 V52 H420 V68',dashed:true}),e('bptt','surrogate'),e('surrogate','learn')]
 },
 'liquid-networks': {
  title:'Input-dependent response speed with fixed inference weights',height:464,
  caption:'A scalar explanatory LTC equation. State and input modulate a gate and effective time constant; this does not automatically update weights θ online. Solver and step size remain part of the computation.',
  nodes:[
   n('input',142,61,'Input I(t)','possibly irregular observations','A continuous-time formulation makes the observation timing and interpolation assumptions explicit.','input',218),
   n('gate',420,61,'Gate f(x,I,t;θ)','nonlinear input/state dependence','The learned gate changes the dynamics as state and input change. Its parameters θ can remain fixed during inference.','state',236),
   n('speed',698,61,'τeff = τ / (1 + τf)','under the scalar assumptions','The effective response time depends on the gate. This expression needs appropriate gate assumptions and is not a generic vector-network theorem.','state',228),
   n('prior',142,221,'Current state x(t)','dynamic state, not a new weight','Changing recurrent state is distinct from parameter learning. The model can adapt its response while keeping learned weights fixed.','input',218),
   n('ode',420,221,'dx/dt = −(1/τ + f)x + fA','leak-like term + driven term','This scalar LTC form combines a gate-dependent decay rate and drive toward level A. Real vector networks require explicit coupling conventions.','state',252),
   n('solver',698,221,'Numerical state evolution','integrate over a chosen Δt','An ODE solver approximates the trajectory; an adaptive gate does not remove numerical stability and error concerns.','state',228),
   n('learning',142,369,'Parameter training','a separate optimization process','Offline training and test-time state evolution are separate phases. Report explicitly if weights are also updated online.','update',218),
   n('next',420,369,'x(t + Δt)','feed state into the next interval','An illustrative Euler step is x(next)=x+Δt·dx/dt. It is a conceptual update, not a universal recommended solver.','state',252),
   n('eval',698,369,'Timing and solver sensitivity','task error, stability and cost','Irregular sampling, integration tolerances and solver cost should be measured independently of the learned model’s task accuracy.','base',228)
  ],
  edges:[e('input','gate'),e('gate','speed'),e('gate','ode','bottom','top'),e('prior','ode'),e('ode','solver'),e('solver','next','bottom','right',{d:'M698 277 V324 H560 V397 H546'}),e('next','prior','left','bottom',{d:'M420 425 V447 H275 V306 H142 V277',dashed:true}),e('learning','gate','top','left',{d:'M142 369 H25 V40 H285 V89 H302',dashed:true})]
 },
 'backprop-biology': {
  title:'Where an exact backward pass needs information',height:459,
  caption:'A two-layer fragment of reverse-mode credit assignment. Colored return paths identify mathematical dependencies. They do not establish which circuit a biological system implements.',
  panels:[p(16,22,808,158,'FORWARD ACTIVITIES'),p(16,202,808,228,'REVERSE CREDIT AND SYNAPTIC UPDATE')],
  nodes:[
   n('hprev',142,74,'hℓ₋₁','presynaptic activity','A synaptic gradient combines an upstream activation with a postsynaptic error signal. The activation must still be available when the error arrives.','input',218),
   n('h',420,74,'aℓ = Wℓhℓ₋₁; hℓ = σ(aℓ)','store derivative-relevant state','The local nonlinear derivative depends on forward activity. Storing or reconstructing it introduces timing and memory requirements.','state',252),
   n('later',698,74,'Later layers → loss L','global training objective','The chosen objective supplies the initial downstream derivative. The existence and origin of a target are distinct from derivative transport.','state',218),
   n('delta',698,252,'Downstream error δℓ₊₁','signed error / derivative signal','An exact reverse pass carries a derivative, which may be signed. Neural signalling constraints require a proposed representation of this quantity.','update',218),
   n('transport',420,252,'δℓ = (Wℓ₊₁ᵀ δℓ₊₁) ⊙ σ′(aℓ)','transpose + local derivative','Exact transport uses the transpose of the forward weights. Feedback alignment changes this particular dependency, without solving every plausibility question.','update',264),
   n('gradient',142,252,'∇Wℓ L = δℓ hℓ₋₁ᵀ','local product of available signals','The gradient is a local outer product only after the correct error and presynaptic activity have become available. Batches sum or average these products.','update',236),
   n('axes',420,363,'Assess separate constraints','symmetry · timing · signals · locality · objective','Biological plausibility is a collection of concrete constraints, not one yes/no property. Numerical gradient agreement does not establish a biological implementation.','base',510,42)
  ],
  edges:[e('hprev','h'),e('h','later'),e('later','delta','bottom','top'),e('delta','transport','left','right'),e('transport','gradient','left','right'),e('h','transport','bottom','top',{dashed:true}),e('hprev','gradient','bottom','top',{dashed:true})]
 },
 'local-learning': {
  title:'Three different sources of a layer’s update',height:477,
  caption:'Schematic comparisons of feedback alignment, layer-local objectives and neighboring consistency. Local access, approximate gradients and objective quality must be evaluated separately.',
  panels:[p(16,20,260,430,'FEEDBACK ALIGNMENT'),p(290,20,260,430,'LAYER-LOCAL OBJECTIVE'),p(564,20,260,430,'NEIGHBORING CONSISTENCY')],
  nodes:[
   n('downstream',146,68,'Downstream error δ̃ℓ₊₁','from a later layer','A downstream error still has to arrive; feedback alignment changes its projection rather than eliminating the need for an error signal.','input',220),
   n('random',146,181,'Fixed feedback Bℓ','no exact Wℓ₊₁ᵀ requirement','A fixed compatible random matrix replaces the exact transposed forward weights. B is part of the learning algorithm and is not trained by the shown update.','state',220),
   n('approx',146,294,'δ̃ℓ = Bℓδ̃ℓ₊₁ ⊙ σ′','an approximate credit signal','The transported signal is generally not the exact gradient of the original end-to-end objective.','update',220),
   n('alignment',146,392,'Measure alignment and learning','no general equivalence guarantee','Forward weights can learn to make feedback useful in tested settings. The claim depends on task, depth and optimization.','base',220,42),
   n('features',420,68,'Intermediate features hℓ','add an auxiliary predictor gℓ','A local head is attached to a layer rather than waiting for the final network’s exact loss derivative.','input',220),
   n('aux',420,181,'Local target y','Lℓ(gℓ(hℓ), y)','A target and objective must be available at the layer. Local availability of a loss does not mean the target itself is biologically plausible.','state',220),
   n('localupdate',420,294,'Update against Lℓ','local representation objective','The local objective can shape useful features, but it can also conflict with what later layers need.','update',220),
   n('conflict',420,392,'Check downstream task quality','local goals may conflict','Evaluate the actual end-to-end task in addition to decreases in each auxiliary loss.','base',220,42),
   n('activities',694,68,'Infer neighboring activities','explicit state variables','Predictive coding treats some activities as variables adjusted to satisfy neighboring predictions or constraints.','input',220),
   n('residuals',694,181,'Prediction / constraint errors','compare adjacent layers','The credit signal comes from consistency errors and the current inferred activities.','state',220),
   n('primaldual',694,294,'PC: state inference','PC-ALM adds multipliers','Augmented-Lagrangian multipliers accumulate violations in PC-ALM. This differs mathematically from a fixed random feedback projection.','update',220),
   n('budget',694,392,'Report finite inference budget','equilibrium and scope matter','An equilibrium derivation does not establish finite-budget exactness for every nonlinear network.','base',220,42)
  ],
  edges:[e('downstream','random','bottom','top'),e('random','approx','bottom','top'),e('approx','alignment','bottom','top'),e('features','aux','bottom','top'),e('aux','localupdate','bottom','top'),e('localupdate','conflict','bottom','top'),e('activities','residuals','bottom','top'),e('residuals','primaldual','bottom','top'),e('primaldual','budget','bottom','top')]
 },
 'continual-learning': {
  title:'Three mechanisms for retaining previous behavior',height:481,
  caption:'A changing stream creates interference in shared parameters. Replay, importance-weighted consolidation and parameter isolation protect old behavior in different ways, with different resource and routing costs.',
  nodes:[
   n('stream',420,50,'New data / task stream','changing training distribution','Task boundaries, access to past data and task IDs at inference all affect what continual-learning problem is being solved.','input',306),
   n('replay',142,175,'Replay buffer','mix selected old and new examples','Stored or generated previous experience keeps some old behavior represented in later updates. Buffer selection and capacity are part of the method.','state',218),
   n('ewc',420,175,'Importance penalty','λ/2 · Σₖ Fₖ(θₖ − θ*ₖ)²','EWC penalizes movement from a previous solution according to diagonal importance estimates. This is an approximation and may restrict new learning.','state',240),
   n('isolate',698,175,'Selected parameter modules','adapters / routing / capacity','Isolating task-specific parameters can reduce shared interference, while leaving routing and capacity-growth questions. LoRA alone is not a continual-learning guarantee.','state',218),
   n('update',420,302,'Update shared or selected weights','state exact memory / compute budget','A fair comparison accounts for stored data, extra modules, retained checkpoints and update cost, rather than only counting the backbone.','update',306),
   n('matrix',420,402,'Re-evaluate all seen tasks: Aₜ,ⱼ','retention, transfer and new-task learning','An evaluation matrix records task j after stage t. Forgetting, forward transfer and backward transfer are different outcomes.','base',500,48)
  ],
  edges:[e('stream','replay','bottom','top',{d:'M420 106 V135 H142 V175'}),e('stream','ewc','bottom','top'),e('stream','isolate','bottom','top',{d:'M420 106 V135 H698 V175'}),e('replay','update','bottom','left',{d:'M142 231 V330 H267'}),e('ewc','update','bottom','top'),e('isolate','update','bottom','right',{d:'M698 231 V330 H573'}),e('update','matrix','bottom','top')]
 },
 'memory-adaptation': {
  title:'Locate the state that changes and persists',height:482,
  caption:'Four independent persistence mechanisms. A system may combine them, but more context, an updated index, a saved adapter and test-time weight updates should not be reported as one undifferentiated capability.',
  panels:[p(16,20,394,208,'CONTEXT / CACHE'),p(430,20,394,208,'EXTERNAL RETRIEVAL MEMORY'),p(16,250,394,208,'SAVED ADAPTER PARAMETERS'),p(430,250,394,208,'TEST-TIME PARAMETER UPDATES')],
  nodes:[
   n('context',213,68,'Inputs → activations and K/V','temporary per-context state','In-context conditioning changes hidden activations and cached keys/values without changing the learned model weights.','input',336),
   n('reset',213,164,'Reset removes internal context','unless context is separately saved','If a session or cache is discarded, its content is not automatically transferred into model parameters.','state',336,44),
   n('index',627,68,'Write records / update an index','external persistent store','An external memory changes when documents or records are written. The generator’s language-model weights can remain fixed.','input',336),
   n('retrieve',627,164,'Retrieve records → condition output','persistence: stored records','Retrieved material can supply evidence, but correct retrieval does not guarantee faithful use by the generator.','state',336,44),
   n('adapter',213,298,'Train a restricted ΔW','for example low-rank factors BA','Adapter training changes a specified subset or parameterization of weights. It requires a learning objective and update procedure.','update',336),
   n('saved',213,394,'Load saved adapter at inference','persistence: saved learned parameters','Saved adapter parameters survive a context reset. Task selection and interference remain separate concerns.','state',336,44),
   n('ttt',627,298,'Update on label-free test objective','copy/reset or stream continuously','Test-time training uses a self-supervised objective on available test inputs. Success depends on that objective’s relation to the target task and shift.','update',336),
   n('policy',627,394,'Specify reset policy + retained behavior','persistence depends on the protocol','Resetting after each input and continuously retaining updates produce different evaluation problems. Measure performance after adaptation and after reset.','state',336,44)
  ],
  edges:[e('context','reset','bottom','top'),e('index','retrieve','bottom','top'),e('adapter','saved','bottom','top'),e('ttt','policy','bottom','top')]
 },
 'self-improvement': {
  title:'Separate revising an output from training the model',height:486,
  caption:'A schematic generate–verify–learn loop, not the exact STaR algorithm. The verifier supplies supervision and can have blind spots; held-out evaluation must remain outside the selection loop.',
  panels:[p(16,20,808,132,'FIXED-WEIGHT OUTPUT REFINEMENT'),p(16,174,808,284,'PARAMETER-CHANGING DATA / TRAINING LOOP')],
  nodes:[
   n('draft',142,70,'Generate answer','weights remain fixed','An inference-time refinement process can start by drafting an answer with the existing model.','input',218),
   n('critique',420,70,'Feedback / critique','same or external feedback source','A critique can improve an answer without changing model parameters. Self-generated feedback may also repeat characteristic errors.','state',240),
   n('revise',698,70,'Revise answer','output changes; θ does not','Repeated revision is an inference computation, not evidence that the model has persistently learned a new capability.','state',218),
   n('generate',142,223,'Generate candidate training data','Gθₖ(y | x)','The current model proposes candidates from training prompts. The data generation budget belongs in the comparison.','input',218),
   n('verify',420,223,'Independent selection V(x,y)','answers, tests, execution or labels','The verifier provides information. Its reliability, external supervision and blind spots determine what the selection signal rewards.','state',240),
   n('train',698,223,'Train θₖ₊₁ on selected data','parameter update','Unlike output refinement, the training step changes saved model weights. Selection can amplify a biased or narrow verifier.','update',218),
   n('test',698,369,'Untouched held-out task suite','measure transfer and retained skills','Repeatedly inspecting the test set can turn it into feedback. Freeze final evaluation separately and compare matched training/inference budgets.','base',218),
   n('boundary',420,369,'Selection / evaluation boundary','do not tune to the final test suite','Information from held-out evaluation must not flow back into the training selection loop if it is to remain a final unbiased check.','base',240)
  ],
  edges:[e('draft','critique'),e('critique','revise'),e('generate','verify'),e('verify','train'),e('train','test','bottom','top'),e('train','generate','bottom','bottom',{d:'M698 279 V315 H142 V279',dashed:true,label:'NEXT ROUND USES UPDATED θ',lx:420,ly:309})]
 },
 'robot-evaluation': {
  title:'A rollout closes both the control and evidence loops',height:489,
  caption:'Closed-loop evaluation with explicit observation age, action semantics and interventions. The delay sum assumes sequential stages; pipelined systems need a timing diagram that represents overlap.',
  nodes:[
   n('scene',142,66,'Robot + physical scene','initial-state distribution','The environment evolves between observation capture and action execution. A rollout, not an isolated offline prediction, is the evaluation unit.','input',218),
   n('observe',420,66,'Capture → transport','observation timestamp','A sensor timestamp is needed to measure how old the observation is when the policy uses it.','state',236),
   n('policy',698,66,'Policy inference','checkpoint + language + calibration','Latency, embodiment conventions and the specific model/controller version are part of the experimental specification.','state',218),
   n('act',698,224,'Dispatch / execute action','action timestamp and semantics','Actions change the next observation distribution. Action units, reference frame, chunk execution and controller rate must be stated.','update',218),
   n('delay',420,224,'Δ = capture + transport','+ inference + dispatch','This additive delay assumes sequential processing. A displacement estimate vΔ is dimensional arithmetic, not a complete stability or collision model.','base',252),
   n('operator',142,224,'Intervention / termination','record takeovers and failures','Human interventions and termination rules affect success measurements; report both their count and their treatment in the metric.','input',218),
   n('record',400,371,'Log complete rollout outcomes','success · violations · delay · intervention','Separate task success from timing and safety-related outcomes. Initial-state identifiers and controller versions make failures traceable.','state',320),
   n('matrix',698,371,'Controlled test variations','perception · dynamics · timing','Changing objects, instructions and embodiment produces different generalization tests. Report trial counts, uncertainty and matched conditions.','base',218)
  ],
  edges:[e('scene','observe'),e('observe','policy'),e('policy','act','bottom','top'),e('act','scene','bottom','bottom',{d:'M698 280 V318 H25 V146 H142 V122',dashed:true}),e('operator','record','bottom','left',{d:'M142 280 V399 H240'}),e('delay','record','bottom','top',{dashed:true}),e('act','record','bottom','right',{d:'M698 280 V339 H575 V399 H560'}),e('record','matrix')]
 }
};

export const conceptFigureTypes = Object.fromEntries(Object.keys(conceptDiagrams).map(slug=>[slug,slug]));

function anchor(node,side){
 if(side==='left')return [node.x-node.w/2,node.y+node.h/2];
 if(side==='right')return [node.x+node.w/2,node.y+node.h/2];
 if(side==='top')return [node.x,node.y];
 return [node.x,node.y+node.h];
}
function edgePath(edge,lookup){
 if(edge.d)return edge.d;
 const [x1,y1]=anchor(lookup[edge.from],edge.fromSide),[x2,y2]=anchor(lookup[edge.to],edge.toSide);
 if(x1===x2||y1===y2)return `M${x1} ${y1}L${x2} ${y2}`;
 if(['left','right'].includes(edge.fromSide)){
  const middle=(x1+x2)/2;return `M${x1} ${y1}H${middle}V${y2}H${x2}`;
 }
 const middle=(y1+y2)/2;return `M${x1} ${y1}V${middle}H${x2}V${y2}`;
}

export function ConceptFigure({type}){
 const diagram=conceptDiagrams[type];
 const [selected,setSelected]=useState('');
 const marker=useId().replace(/:/g,'')+'-concept';
 if(!diagram)return null;
 const lookup=Object.fromEntries(diagram.nodes.map(node=>[node.id,node]));
 const current=lookup[selected];
 const active=new Set(current?[selected,...diagram.edges.flatMap(edge=>edge.from===selected?[edge.to]:edge.to===selected?[edge.from]:[])]:diagram.nodes.map(node=>node.id));
 const controls=<><label className="control-inline" style={{gap:10}}><span>Inspect a component</span><select aria-label="Inspect diagram component" value={current?selected:''} onChange={event=>setSelected(event.target.value)} style={{background:'var(--surface)',color:'var(--ink)',border:'1px solid var(--border)',borderRadius:5,padding:'6px 9px',maxWidth:330,font:'inherit'}}><option value="">Full diagram</option>{diagram.nodes.map(node=><option key={node.id} value={node.id}>{node.label}</option>)}</select></label><span className="small-note">Select a node to inspect its assumptions.</span></>;
 return <Figure title={diagram.title} controls={controls} caption={<span aria-live="polite">{current?<><strong>{current.label}.</strong> {current.detail} </>:null}{!current&&diagram.caption}</span>}>
  <svg viewBox={`0 0 840 ${diagram.height}`} role="img" aria-labelledby={`${marker}-title ${marker}-desc`}>
   <title id={`${marker}-title`}>{diagram.title}</title><desc id={`${marker}-desc`}>{diagram.caption}</desc><ArrowDefs id={marker}/>
   {diagram.panels?.map((panel,index)=><g key={index} aria-hidden="true"><rect x={panel.x} y={panel.y} width={panel.w} height={panel.h} fill="var(--fig-panel)" stroke="var(--border)" rx="8"/><text x={panel.x+panel.w/2} y={panel.y+22} className="svg-overline" style={{fontSize:8.8}}>{panel.label}</text></g>)}
   {diagram.edges.map((edge,index)=>{
    const lit=!current||edge.from===selected||edge.to===selected;
    return <g key={index} aria-hidden="true" opacity={lit?1:.18}><path d={edgePath(edge,lookup)} fill="none" stroke={edge.dashed?'var(--orange)':'var(--fig-line)'} strokeWidth={current&&lit?2.1:1.4} strokeDasharray={edge.dashed?'5 4':undefined} markerEnd={`url(#${marker})`}/>{edge.label&&<text x={edge.lx} y={edge.ly} className="svg-overline" style={{fontSize:8}}>{edge.label}</text>}</g>;
   })}
   {diagram.nodes.map(node=><g key={node.id} role="button" tabIndex={0} aria-label={`Inspect ${node.label}`} aria-pressed={selected===node.id} onClick={()=>setSelected(selected===node.id?'':node.id)} onKeyDown={event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();setSelected(selected===node.id?'':node.id);}}} style={{cursor:'pointer'}} opacity={active.has(node.id)?1:.28}>
    <rect x={node.x-node.w/2} y={node.y} width={node.w} height={node.h} rx="6" fill={selected===node.id?'var(--accent-soft)':'var(--fig-bg)'} stroke={selected===node.id?'var(--accent)':C[node.kind]} strokeWidth={selected===node.id?2.3:1.35}/>
    <text x={node.x} y={node.y+(node.sub?(node.h<50?17:21):node.h/2+4)} className="svg-label" style={{fontSize:11.5}}>{node.label}</text>
    {node.sub&&<text x={node.x} y={node.y+node.h-12} className="svg-small" style={{fontSize:9.15}}>{node.sub}</text>}
   </g>)}
  </svg>
 </Figure>;
}
