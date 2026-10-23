// Exact, deliberately small synthetic calculations; no model checkpoint is loaded.
export const embeddingRows=[[1,0],[0,1],[-1,0],[0,-1]];
export function toyEmbedding(index,alpha=null){
 if(alpha!==null){if(!Number.isFinite(alpha)||alpha<0||alpha>1)throw new RangeError('Mixture weight must be in [0,1]');return [alpha,1-alpha];}
 if(!Number.isInteger(index)||index<0||index>=embeddingRows.length)throw new RangeError('Unknown toy token');return [...embeddingRows[index]];
}
export function softmax(logits){const m=Math.max(...logits),e=logits.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
export function readout(h){const logits=[h[0],h[1],-h[0]-h[1]];return {h,logits,probabilities:softmax(logits)};}
export function toyLens(mode,x=0,y=1){return readout(mode==='translated'?[y,x]:[x,y]);}
export function kl(target,p){return target.reduce((s,q,i)=>s+(q===0?0:q*Math.log(q/p[i])),0);}
export function toyJacobian(q){if(!Number.isFinite(q)||q<0||q>1)throw new RangeError('Context weight must be in [0,1]');return [[1,0],[0,2*q-1]];}
export function patchRecovery(clean,alternative,patched){const den=clean-alternative;return Math.abs(den)<1e-12?null:(patched-alternative)/den;}
export const finalGrid=[[.5,.51],[.51,.5],[-.6,.2]];
export function layerState(layer,final){if(layer===0)return [final[1],final[0]];if(layer===1)return [2*final[0]+.2,.5*final[1]-.1];return [...final];}
const fitFinals=[[-1,-.5],[-.4,.9],[.7,-.6],[1,.8]];
export function applyTranslator(h,fit){return fit.A.map((row,i)=>row[0]*h[0]+row[1]*h[1]+fit.c[i]);}
export function fitToyTranslator(layer,steps=0){
 if(!Number.isInteger(steps)||steps<0||steps>1000)throw new RangeError('Invalid fitting steps');
 const fit={A:[[1,0],[0,1]],c:[0,0]};
 const loss=()=>fitFinals.reduce((s,z)=>s+kl(readout(z).probabilities,readout(applyTranslator(layerState(layer,z),fit)).probabilities),0)/fitFinals.length;
 const initialLoss=loss();
 for(let step=0;step<steps;step++){
  const gA=[[0,0],[0,0]],gc=[0,0];
  for(const z of fitFinals){const h=layerState(layer,z),p=readout(applyTranslator(h,fit)).probabilities,t=readout(z).probabilities,e=p.map((v,i)=>v-t[i]),g=[e[0]-e[2],e[1]-e[2]];for(let i=0;i<2;i++){gc[i]+=g[i]/4;for(let j=0;j<2;j++)gA[i][j]+=g[i]*h[j]/4;}}
  for(let i=0;i<2;i++){fit.c[i]-=.25*gc[i];for(let j=0;j<2;j++)fit.A[i][j]-=.25*gA[i][j];}
 }
 return {...fit,initialLoss,loss:loss()};
}
export function toyEvidence(layer,position){
 const h=[.2+.3*position+.15*layer,.6-.2*position+.1*layer],clean=[h[0],h[1]+.4],margin=v=>Math.tanh(v[0])+v[1]**2;
 return {h,cacheShape:[1,3,2],lens:readout(h),probe:1/(1+Math.exp(-(h[0]-h[1]))),gradient:[1-Math.tanh(h[0])**2,2*h[1]],alternative:margin(h),clean:margin(clean),patched:margin([h[0],clean[1]])};
}
export const sparseFrame=[[1,0],[0,1],[1,1],[0,-1]];
export function sparseDecomposition(v,k=2,frame=sparseFrame){
 if(![1,2].includes(k))throw new RangeError('Toy sparsity must be 1 or 2');
 const candidates=[];
 const add=(indices,c)=>{if(c.some(x=>x< -1e-10))return;const coeff=[0,0,0,0];indices.forEach((j,i)=>coeff[j]=Math.max(0,c[i]));const reconstruction=[0,1].map(i=>coeff.reduce((s,a,j)=>s+a*frame[j][i],0)),error=Math.hypot(...v.map((a,i)=>a-reconstruction[i]));candidates.push({coeff,reconstruction,error,active:coeff.filter(a=>a>1e-10).length});};
 add([],[]);
 frame.forEach((a,i)=>{const norm=a[0]**2+a[1]**2;if(norm>1e-12)add([i],[(v[0]*a[0]+v[1]*a[1])/norm]);});
 if(k===2)for(let i=0;i<4;i++)for(let j=i+1;j<4;j++){const a=frame[i],b=frame[j],det=a[0]*b[1]-a[1]*b[0];if(Math.abs(det)>1e-10)add([i,j],[(v[0]*b[1]-v[1]*b[0])/det,(a[0]*v[1]-a[1]*v[0])/det]);}
 candidates.sort((a,b)=>Math.abs(a.error-b.error)>1e-10?a.error-b.error:a.active-b.active);return candidates[0];
}
export function toyForward(x,patch=null){
 const h0=[x,1-x];if(patch?.site===0)h0[0]=patch.value;
 const h1=[h0[0]+h0[1],h0[0]-h0[1]];if(patch?.site===1)h1[1]=patch.value;
 const h2=[Math.tanh(h1[0]),h1[1]+.5*h1[1]**2];if(patch?.site===2)h2[0]=patch.value;
 return {h0,h1,h2,margin:2*h2[0]+h2[1]};
}
export function toyPatch(site,cleanX=1,alternativeX=0){
 const clean=toyForward(cleanX),alternative=toyForward(alternativeX),value=[clean.h0[0],clean.h1[1],clean.h2[0]][site];
 if(value===undefined)throw new RangeError('Unknown patch site');const patched=toyForward(alternativeX,{site,value});return {clean,alternative,patched,value,recovery:patchRecovery(clean.margin,alternative.margin,patched.margin)};
}
export function toyFeatureIntervention(coefficient=.8,retainError=true){
 const h=[1.2,.7],f=[.8,.3,.2],decoder=[[1,0],[0,1],[1,1]],decode=c=>[0,1].map(i=>c.reduce((s,a,j)=>s+a*decoder[j][i],0)),reconstruction=decode(f),error=h.map((a,i)=>a-reconstruction[i]),changed=decode([coefficient,.3,.2]).map((a,i)=>a+(retainError?error[i]:0)),metric=v=>2*Math.tanh(v[0])+v[1]+.5*v[1]**2,delta=changed[0]-h[0];
 return {h,f,decoder,reconstruction,error,changed,baseline:metric(h),margin:metric(changed),linearEstimate:2*(1-Math.tanh(h[0])**2)*delta};
}
