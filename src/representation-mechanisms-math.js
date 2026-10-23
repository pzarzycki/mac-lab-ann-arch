// All values in these demonstrations are synthetic, not fitted model measurements.
export function softmax(values, temperature=1) {
 if (!values.length || !(temperature>0) || !Number.isFinite(temperature) || values.some(v=>!Number.isFinite(v))) throw new RangeError('Finite logits and positive temperature required');
 const max=Math.max(...values), weights=values.map(v=>Math.exp((v-max)/temperature)), total=weights.reduce((a,b)=>a+b,0);
 return weights.map(v=>v/total);
}
export const people=[{name:'Ada',code:[1,1,0]},{name:'Ben',code:[1,.8,.2]},{name:'Cora',code:[0,1,0]},{name:'Dan',code:[0,0,1]}];
export function relationTrace(person=0,eta=0) {
 if(!people[person] || !Number.isFinite(eta) || eta<0) throw new RangeError('Invalid relation query');
 const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0), initial=[[0,.2,0],[0,0,.2]], train=people[0].code;
 const before=softmax(initial.map(w=>dot(w,train)));
 const weights=initial.map((w,k)=>w.map((v,j)=>v+eta*((k===0?1:0)-before[k])*train[j]));
 const scores=weights.map(w=>dot(w,people[person].code));
 return {weights,scores,probabilities:softmax(scores),initial:softmax(initial.map(w=>dot(w,people[person].code))),isolated:person===0?softmax([.2+eta*(1-before[0]),-eta*before[1]]):softmax(initial.map(w=>dot(w,people[person].code)))};
}
export function additiveRead(query) {
 if(!Number.isFinite(query))throw new RangeError('Finite query required');
 const states=[-1,0,1],scores=states.map(h=>Math.tanh(h-query+1)-Math.tanh(h-query-1)),weights=softmax(scores), contributions=weights.map((a,j)=>a*states[j]),context=contributions.reduce((a,b)=>a+b,0),nextState=Math.tanh(.5*query+context+.2);
 return {states,scores,weights,contributions,context,nextState,output:softmax([nextState,-nextState])};
}
export function scalarBlock(branch,skip=1,x=1,upstream=1,postRelu=false) {
 if([branch,skip,x,upstream].some(v=>!Number.isFinite(v)))throw new RangeError('Finite scalar parameters required');
 const sum=(skip+branch)*x,gate=postRelu?(sum>0?1:0):1;
 return {sum,output:postRelu?Math.max(0,sum):sum,skipGradient:gate*skip*upstream,branchGradient:gate*branch*upstream,gradient:gate===0?0:gate*(skip+branch)*upstream,gate};
}
export function scalingLoss(N,compute=64,{a=1,b=1,alpha=.5,beta=.5,floor=1}={}) {
 if(!(N>0) || !(compute>0) || [N,compute,a,b,alpha,beta,floor].some(v=>!Number.isFinite(v)) || a<=0 || b<=0 || alpha<=0 || beta<=0)throw new RangeError('Positive finite scaling inputs required');
 const D=compute/N, model=a*N**(-alpha),data=b*D**(-beta);
 return {N,D,model,data,total:floor+model+data,optimumN:(alpha*a/(beta*b)*compute**beta)**(1/(alpha+beta))};
}
export function fusionCounts(mode,p=8,m=2,n=4,resample=false) {
 if(!['prefix','compressed','cross'].includes(mode) || [p,m,n].some(v=>!Number.isInteger(v)||v<=0)||m>p)throw new RangeError('Invalid fusion counts');
 const memory=mode==='cross'?(resample?m:p):0,visual=mode==='prefix'?p:mode==='compressed'?m:0,length=n+visual;
 return {memory,visual,length,selfPairs:length**2,crossPairs:n*memory,resamplerPairs:(mode==='compressed'||(mode==='cross'&&resample))?m*p:0};
}
export const batchVectors=[[1,0],[0,1],[.8,.6]];
export const similarityMatrix=batchVectors.map(v=>batchVectors.map(t=>v.reduce((s,a,k)=>s+a*t[k],0)));
export function contrastiveTrace(direction='image',index=0,temperature=.3) {
 if(!['image','text'].includes(direction)||!Number.isInteger(index)||index<0||index>=3)throw new RangeError('Invalid retrieval query');
 const logits=direction==='image'?similarityMatrix[index]:similarityMatrix.map(row=>row[index]);
 const probabilities=softmax(logits,temperature);
 return {logits,probabilities,loss:-Math.log(probabilities[index]),batchLoss:similarityMatrix.reduce((s,row,i)=>s-Math.log(softmax(direction==='image'?row:similarityMatrix.map(r=>r[i]),temperature)[i]),0)/3};
}
export function maskedTokenLoss(probabilities,mask) {
 if(probabilities.length!==mask.length || probabilities.some(p=>!(p>0&&p<=1)) || mask.some(m=>m!==0&&m!==1))throw new RangeError('Probability/mask mismatch');
 const scored=mask.reduce((a,b)=>a+b,0);return {scored,sum:probabilities.reduce((s,p,i)=>s-mask[i]*Math.log(p),0),mean:scored?probabilities.reduce((s,p,i)=>s-mask[i]*Math.log(p),0)/scored:0};
}
export function temporalTrace(fps=2,phase=0,offset=0,duration=4,patches=196) {
 if(!(fps>0)||!(duration>0)||!(patches>0)||!Number.isInteger(patches)||phase<0||phase>=1||[fps,phase,offset,duration].some(v=>!Number.isFinite(v)))throw new RangeError('Invalid sampling inputs');
 const frames=[];for(let k=0;;k++){const t=(k+phase)/fps;if(t>=duration-1e-10)break;frames.push(t);if(k>10000)throw new RangeError('Too many samples');}
 const event=[1.15,1.35],audio=Array.from({length:Math.floor(duration*2)},(_,i)=>{const physical=.25+.5*i,reported=physical+offset;return {physical,reported,start:reported-.1,end:reported+.1,isEvent:Math.abs(physical-1.25)<1e-9,allowed:Math.abs(reported-1.25)<=.26};});
 return {frames,event,hits:frames.filter(t=>t>=event[0]&&t<event[1]),audio,tokens:frames.length*patches,eventAudioAllowed:audio.some(w=>w.isEvent&&w.allowed)};
}
