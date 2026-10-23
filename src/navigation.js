export const countLabel=(count,one,many=one+'s')=>`${count} ${count===1?one:many}`;
export const legacyDomains={architecture:['transformers'],sequence:['structural-extensions'],systems:['efficient-execution'],multimodal:['multimodal-world-models'],generation:['generation-computation'],learning:['learning-adaptation','inference-control'],evolution:['neuroevolution'],continual:['persistent-adaptation'],robotics:['embodied-intelligence'],history:['lineage']};
const aliases={'Layer Normalization (LN)':['LayerNorm','layer norm','LN'],'RMSNorm':['root mean square normalization'],'Query / key / value':['QKV','query key value'],'GQA':['grouped query attention'],'MoE':['mixture of experts'],'mHC':['manifold constrained hyper connections'],'JEPA':['joint embedding predictive architecture'],'CPPN':['compositional pattern producing network'],'PC-ALM':['augmented Lagrangian predictive coding'],'J-Lens / Jacobian lens':['J lens','J space','Jacobian'],'VLA':['vision language action'],'KV cache':['key value cache'],'Flow matching':['rectified flow'],'Test-time training':['TTT']};
const compact=s=>s.toLowerCase().replace(/[^a-z0-9]/g,'');
export function glossaryMatches([term,definition],query){
 const words=query.toLowerCase().trim().split(/\s+/).filter(Boolean);
 const labels=[term,...(aliases[term]||[])],hay=[...labels,definition].join(' ').toLowerCase();
 return words.every(w=>hay.includes(w)||labels.some(label=>compact(label).includes(compact(w))&&compact(w)));
}
