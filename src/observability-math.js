// Deliberately small, synthetic calculations for the atlas figures.
// These are explanatory matrices, not fitted lens parameters or model outputs.
export const embeddingRows=[[1,0],[0,1],[-1,0],[0,-1]];
export function toyEmbedding(index,alpha=null){
 if(alpha!==null){if(!Number.isFinite(alpha)||alpha<0||alpha>1)throw new RangeError('Mixture weight must be in [0,1]');return [alpha,1-alpha];}
 if(!Number.isInteger(index)||index<0||index>=embeddingRows.length)throw new RangeError('Unknown toy token');
 return [...embeddingRows[index]];
}
export function toyLens(mode,x=0,y=1){
 const h=mode==='translated'?[y,x]:[x,y],logits=[h[0],h[1],-h[0]-h[1]],m=Math.max(...logits),exp=logits.map(v=>Math.exp(v-m)),sum=exp.reduce((a,b)=>a+b,0);
 return {h,logits,probabilities:exp.map(v=>v/sum)};
}
export function toyJacobian(q){if(!Number.isFinite(q)||q<0||q>1)throw new RangeError('Context weight must be in [0,1]');return [[1,0],[0,2*q-1]];}
export function patchRecovery(clean,alternative,patched){const den=clean-alternative;return Math.abs(den)<1e-12?null:(patched-alternative)/den;}
