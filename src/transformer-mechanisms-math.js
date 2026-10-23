// Small, explicitly synthetic examples used by TransformerMechanisms.
export const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
export const add = (a,b) => a.map((x,i)=>x+b[i]);
export const matvec = (m,x) => m.map(row=>dot(row,x));
export function normStats(x,epsilon=1e-5){
 const mean=x.reduce((a,b)=>a+b,0)/x.length;
 const variance=x.reduce((s,v)=>s+(v-mean)**2,0)/x.length;
 const rms=Math.sqrt(x.reduce((s,v)=>s+v*v,0)/x.length+epsilon);
 return {mean,variance,rms,ln:x.map(v=>(v-mean)/Math.sqrt(variance+epsilon)),rmsnorm:x.map(v=>v/rms)};
}
export function rotate2(v,angle){const c=Math.cos(angle),s=Math.sin(angle);return [c*v[0]-s*v[1],s*v[0]+c*v[1]];}
export const silu = x => x/(1+Math.exp(-x));
export function swigluToy(x){
 const gate=matvec([[1,0],[0,1],[1,-1]],x),up=matvec([[1,1],[1,-1],[0,1]],x);
 const activated=gate.map(silu),product=activated.map((v,i)=>v*up[i]);
 return {gate,up,activated,product,output:matvec([[1,0,1],[0,1,-1]],product)};
}
export function softmax(logits){
 const max=Math.max(...logits);
 if(max===-Infinity)return logits.map(()=>0); // explicit all-masked sentinel
 const e=logits.map(v=>Math.exp(v-max)),z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/z);
}
export function attentionTrace(query,keys,values,allowed=keys.map(()=>true)){
 const logits=keys.map(k=>dot(query,k)/Math.sqrt(query.length));
 const masked=logits.map((v,i)=>allowed[i]?v:-Infinity),weights=softmax(masked);
 const contributions=values.map((v,i)=>v.map(x=>x*weights[i]));
 const output=values[0].map((_,j)=>contributions.reduce((s,v)=>s+v[j],0));
 return {logits,masked,weights,contributions,output};
}
export function onlineAttention(logits,values,tileSize=2){
 if(!Number.isInteger(tileSize)||tileSize<1)throw new RangeError('tileSize must be positive');
 let m=-Infinity,ell=0,numerator=values[0].map(()=>0);const states=[];
 for(let start=0;start<logits.length;start+=tileSize){
  const end=Math.min(start+tileSize,logits.length),local=logits.slice(start,end),tileMax=Math.max(...local),nextMax=Math.max(m,tileMax);
  const rescale=m===-Infinity?0:Math.exp(m-nextMax);
  const exp=local.map(v=>v===-Infinity?0:Math.exp(v-nextMax));
  ell=rescale*ell+exp.reduce((a,b)=>a+b,0);
  numerator=numerator.map((v,j)=>rescale*v+exp.reduce((sum,w,k)=>sum+w*values[start+k][j],0));m=nextMax;
  states.push({start,end,tileMax,m,ell,rescale,numerator:[...numerator],output:numerator.map(v=>ell?v/ell:0)});
 }
 return states;
}
export function quantizeAffine(weights,bits,scale){
 if(!(scale>0)||!Number.isInteger(bits)||bits<2||bits>8)throw new RangeError('positive scale and 2–8 bits required');
 const min=0,max=2**bits-1,zero=2**(bits-1);
 const rows=weights.map(w=>{const raw=Math.round(w/scale)+zero,code=Math.max(min,Math.min(max,raw)),reconstructed=scale*(code-zero);return {w,raw,code,reconstructed,error:reconstructed-w,clipped:raw<min||raw>max};});
 return {min,max,zero,rows,packedBytes:Math.ceil(weights.length*bits/8),metadataBytes:5,mse:rows.reduce((s,r)=>s+r.error**2,0)/weights.length};
}
export function speculativeMass(p,q){
 const accepted=p.map((v,i)=>Math.min(v,q[i])),rejection=1-accepted.reduce((a,b)=>a+b,0),positive=p.map((v,i)=>Math.max(0,v-q[i])),z=positive.reduce((a,b)=>a+b,0);
 const correction=positive.map(v=>z?v/z:0);
 return {accepted,rejection,correction,recovered:accepted.map((v,i)=>v+rejection*correction[i])};
}
export function sampleCategorical(p,u){let sum=0;for(let i=0;i<p.length;i++){sum+=p[i];if(u<sum)return i;}return p.length-1;}
export function speculativeCycle(p,q,proposals,uniforms,extraUniform=.7){
 const checks=[];let rejected=-1;
 for(let i=0;i<proposals.length;i++){
  const token=proposals[i],ratio=Math.min(1,p[token]/q[token]),accepted=uniforms[i]<ratio;
  checks.push({token,ratio,u:uniforms[i],accepted});if(!accepted){rejected=i;break;}
 }
 const mass=speculativeMass(p,q),extra=sampleCategorical(rejected<0?p:mass.correction,extraUniform),prefix=proposals.slice(0,rejected<0?proposals.length:rejected);
 return {checks,rejected,extra,prefix,result:[...prefix,extra],discarded:rejected<0?[]:proposals.slice(rejected)};
}
export function kvMemory({batch,layers,context,kvHeads,headDim,bytes}){return 2*batch*layers*context*kvHeads*headDim*bytes;}
export function headGroup(queryHead,queryHeads,kvHeads){
 if(queryHeads%kvHeads)throw new RangeError('equal groups require divisibility');
 return Math.floor(queryHead/(queryHeads/kvHeads));
}
