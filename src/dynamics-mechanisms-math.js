// Synthetic mechanism calculations; no trained-model or hardware measurements.
import {topKWeights,sinkhorn} from './math.js';
export function moeBatch(logits,k,capacity,balanced=false){
 if(!Number.isInteger(capacity)||capacity<0)throw new RangeError('Capacity must be a nonnegative integer.');
 const counts=logits.map(()=>0),accepted=logits.map(()=>0);
 const tokens=Array.from({length:6},(_,t)=>{
  // Identical preferences expose collapse; cyclic preferences expose a balanced batch.
  const scores=logits.map((_,e)=>logits[(e+(balanced?t:0))%logits.length]);
  const weights=topKWeights(scores,k);
  const routes=weights.map((g,e)=>{if(!g)return {g,accepted:false};counts[e]++;const ok=accepted[e]<capacity;if(ok)accepted[e]++;return {g,accepted:ok};});
  return {scores,weights,routes};
 });
 return {tokens,counts,accepted,dropped:counts.map((n,e)=>n-accepted[e]),total:6*k};
}
export function expertVector(e,h=[1,2]){return [(e+1)*h[0],h[1]-(e+1)];}
export function moeOutput(routes,h=[1,2]){return routes.reduce((out,r,e)=>out.map((v,j)=>v+(r.accepted?r.g*expertVector(e,h)[j]:0)),[0,0]);}
export function laneMix(iterations,bias,x=[1,2,-1]){
 const R=sinkhorn([[Math.exp(bias),.6,1.2],[.4,1.8,.8],[1.5,.3,1]],iterations),mixed=R.map(row=>row.reduce((s,r,j)=>s+r*x[j],0));
 const a=[.5,.25,.25],b=[1,.5,0],aggregate=a.reduce((s,v,i)=>s+v*x[i],0),f=Math.tanh(aggregate);
 return {R,mixed,a,b,aggregate,f,output:mixed.map((v,i)=>v+b[i]*f),rows:R.map(r=>r.reduce((a,b)=>a+b,0)),cols:R[0].map((_,j)=>R.reduce((s,r)=>s+r[j],0))};
}
export const loopMaps={stable:z=>Math.tanh(.6*z+.4),oscillatory:z=>-z,expanding:z=>1.15*z};
export function loopTrace(steps,mode='stable',input=1,early=false,tolerance=.01){
 const values=[-1];let stopped=false;
 for(let k=0;k<steps;k++){const old=values.at(-1),next=mode==='stable'?Math.tanh(.6*old+.4*input):loopMaps[mode](old);values.push(next);if(early&&Math.abs(next-old)<tolerance){stopped=true;break;}}
 return {values,stopped,delta:values.length>1?Math.abs(values.at(-1)-values.at(-2)):null};
}
export const selectiveTokens=[{label:'write 2',x:2,kind:'write'},{label:'noise 9',x:9,kind:'noise'},{label:'noise −3',x:-3,kind:'noise'},{label:'read',x:0,kind:'read'},{label:'write −1',x:-1,kind:'write'},{label:'read',x:0,kind:'read'}];
export function selectiveTrace(selective=true){let s=0;return selectiveTokens.map(token=>{const a=selective?(token.kind==='write'?0:1):.5,b=selective?(token.kind==='write'?1:0):.5,c=token.kind==='read'?1:0,prior=s,retained=a*prior,write=b*token.x;s=retained+write;return {...token,a,b,c,prior,retained,write,s,y:c*s};});}
export function lifTrace({steps=12,beta=.8,drive=.6,threshold=1,pattern='constant'}={}){
 if(beta<0||beta>=1||threshold<=0||steps<1)throw new RangeError('Invalid LIF parameters.');
 let u=0;return Array.from({length:steps+1},(_,t)=>{const z=u>=threshold?1:0,I=pattern==='pulses'?(t%3===0?drive:0):drive,next=beta*u+I-threshold*z,point={t,u,z,I,next};u=next;return point;});
}
export function surrogateDerivative(v,width=.5,scale=1){if(width<=0||scale<0)throw new RangeError('Invalid surrogate.');return scale*Math.max(0,1-Math.abs(v)/width);}
export function liquidGate(x,input,feedback=0){return 2/(1+Math.exp(-(input+feedback*x)));}
export function liquidResponse({tau=1,A=1,input=0,feedback=0,dt=.2,duration=4}={}){
 if(tau<=0||dt<=0||duration<=0)throw new RangeError('Positive time parameters required.');
 const f0=liquidGate(0,input,feedback),rate=1/tau+f0,equilibrium=f0*A/rate,teff=1/rate;
 const rhs=x=>{const f=liquidGate(x,input,feedback);return -(1/tau+f)*x+f*A;};
 let x=0,reference=0,time=0;const points=[{t:0,x:0,reference:0,exact:0,gate:f0}],referenceCurve=[{t:0,x:0}];
 while(time<duration-1e-10){const h=Math.min(dt,duration-time);x+=h*rhs(x);
  const sub=h/100;for(let j=0;j<100;j++){const k1=rhs(reference),k2=rhs(reference+sub*k1/2),k3=rhs(reference+sub*k2/2),k4=rhs(reference+sub*k3);reference+=sub*(k1+2*k2+2*k3+k4)/6;if(j%5===4){const t=time+(j+1)*sub;referenceCurve.push({t,x:feedback===0?equilibrium*(1-Math.exp(-rate*t)):reference});}}
  time+=h;points.push({t:time,x,reference,exact:equilibrium*(1-Math.exp(-rate*time)),gate:liquidGate(x,input,feedback)});
 }
 return {points,referenceCurve,f0,teff,equilibrium,rate,error:Math.abs(x-(feedback===0?points.at(-1).exact:reference))};
}
