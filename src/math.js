// All laboratory values are calculated locally; none are benchmark measurements.
export function kvBytes({batch,layers,context,kvHeads,headDim,bytes}) {
 const v=[batch,layers,context,kvHeads,headDim,bytes];
 if(v.some(x=>!Number.isFinite(x)||x<=0)) throw new RangeError('Cache dimensions and scalar width must be positive.');
 return 2*batch*layers*context*kvHeads*headDim*bytes;
}
export function allowedAttention(i,j,mode='causal',window=4){return mode==='full'||(j<=i&&(mode!=='window'||i-j<window));}
export function edgeCount(n,mode,window){let e=0;for(let i=0;i<n;i++)for(let j=0;j<n;j++)if(allowedAttention(i,j,mode,window))e++;return e;}
export function topKWeights(logits,k){
 if(!Number.isInteger(k)||k<1||k>logits.length||logits.some(v=>!Number.isFinite(v)))throw new RangeError('Invalid router values.');
 const selected=logits.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v||a.i-b.i).slice(0,k);
 const max=selected[0].v,denom=selected.reduce((s,x)=>s+Math.exp(x.v-max),0);
 return logits.map((_,i)=>selected.some(x=>x.i===i)?Math.exp(logits[i]-max)/denom:0);
}
export function sinkhorn(matrix,iterations){
 if(matrix.length===0||matrix.some(row=>row.length!==matrix.length||row.some(x=>!Number.isFinite(x)||x<=0)))throw new RangeError('Expected a positive square matrix.');
 let m=matrix.map(r=>[...r]);
 for(let t=0;t<iterations;t++){
   const sums=m[0].map((_,j)=>m.reduce((s,r)=>s+r[j],0));
   m=m.map(r=>r.map((v,j)=>v/sums[j]));
   m=m.map(r=>{const sum=r.reduce((a,b)=>a+b,0);return r.map(v=>v/sum);});
 }
 return m;
}
export function recurrentValues(steps,input=1){const values=[-1];for(let i=0;i<steps;i++)values.push(Math.tanh(.6*values.at(-1)+.4*input));return values;}
export function creditSimulation(steps,mode='alm',target=.2){
 const x=1,w=[.8,.7,1.2],rho=1,eta=.06,alpha=mode==='alm'?.15:0;
 let h=[w[0]*x,w[1]*w[0]*x],lam=[0,0];
 const err=w[2]*w[1]*w[0]*x-target;
 const bp=[err*w[2]*w[1]*x,err*w[2]*w[0]*x,err*w[1]*w[0]*x];
 const history=[];
 const snapshot=()=>{
  const r=[h[0]-w[0]*x,h[1]-w[1]*h[0]],e=r.map((r,i)=>lam[i]+rho*r);
  const grads=[-e[0]*x,-e[1]*h[0],(w[2]*h[1]-target)*h[1]];
  const relativeError=Math.hypot(...grads.map((g,i)=>g-bp[i]))/Math.hypot(...bp);
  return {h:[...h],lam:[...lam],r,grads,bp,relativeError};
 };
 history.push(snapshot());
 for(let t=0;t<steps;t++){
  const r=[h[0]-w[0]*x,h[1]-w[1]*h[0]],e=r.map((r,i)=>lam[i]+rho*r);
  const grad=[e[0]-w[1]*e[1],w[2]*(w[2]*h[1]-target)+e[1]];
  h=h.map((v,i)=>v-eta*grad[i]);
  if(t<steps-1){const nr=[h[0]-w[0]*x,h[1]-w[1]*h[0]];lam=lam.map((v,i)=>v+alpha*nr[i]);}
  history.push(snapshot());
 }
 return {...snapshot(),history,steps,mode,target};
}
