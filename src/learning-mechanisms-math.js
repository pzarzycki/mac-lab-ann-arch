// Synthetic mechanism examples, never empirical model measurements.
export function backwardTrace(weight=-.5,target=.2){
 const x=[1,-1],W=[[.5,-.3],[-.2,.4]],a=W.map(r=>r.reduce((s,v,j)=>s+v*x[j],0)),h=a.map(Math.tanh),w=[.7,weight],prediction=w.reduce((s,v,i)=>s+v*h[i],0),error=prediction-target,derivative=h.map(v=>1-v*v),transport=w.map(v=>v*error),delta=transport.map((v,i)=>v*derivative[i]),gradient=delta.map(v=>x.map(z=>v*z));
 return {x,W,a,h,w,prediction,error,derivative,transport,delta,gradient,loss:.5*error*error};
}
export function creditDirections(weight=-.5,target=.2){
 const s=backwardTrace(weight,target),B=[-.4,.9],C=[.6,.2],auxPrediction=C.reduce((v,c,i)=>v+c*s.h[i],0),auxError=auxPrediction-target;
 const exact=s.delta,random=B.map((v,i)=>v*s.error*s.derivative[i]),head=C.map((v,i)=>v*auxError*s.derivative[i]);
 const next=s.h.map((v,i)=>v-.1*s.w[i]*s.error),residual=next.map((v,i)=>v-s.h[i]),pc=residual.map((v,i)=>-v*s.derivative[i]);
 const cosine=v=>{const d=Math.hypot(...exact)*Math.hypot(...v);return d?exact.reduce((q,e,i)=>q+e*v[i],0)/d:null;};
 return {...s,B,C,auxPrediction,auxError,exact,random,head,next,residual,pc,cosines:{random:cosine(random),head:cosine(head),pc:cosine(pc)}};
}
export function pcActivityStep(h=[.8,.56],lam=[0,0]){
 const r=[h[0]-.8,h[1]-.7*h[0]],e=r.map((v,i)=>v+lam[i]),output=1.2*h[1]-.2,gradient=[e[0]-.7*e[1],e[1]+1.2*output],next=h.map((v,i)=>v-.06*gradient[i]);
 return {r,e,output,gradient,next};
}
const entropy=p=>-p.reduce((s,v)=>s+(v? v*Math.log(v):0),0);
export function activeInference(accuracy=.7,preference=.8){
 if(!Number.isFinite(accuracy)||!Number.isFinite(preference)||accuracy<.5||accuracy>1||preference<=0||preference>=1)throw new RangeError('Invalid likelihood or preference.');
 const belief=[accuracy,1-accuracy],preferences=[preference,1-preference],A=[[accuracy,1-accuracy],[1-accuracy,accuracy]];
 const policies=[{name:'Stay',state:belief},{name:'Reset',state:[.85,.15]}].map(p=>{const outcomes=A.map(row=>row.reduce((s,v,j)=>s+v*p.state[j],0)),ambiguity=entropy([accuracy,1-accuracy]),information=entropy(outcomes)-ambiguity,cost=-outcomes.reduce((s,v,j)=>s+v*Math.log(preferences[j]),0),risk=cost-entropy(outcomes);return {...p,outcomes,ambiguity,information,cost,risk,G:cost-information};});
 const weights=policies.map(p=>Math.exp(-4*p.G)),sum=weights.reduce((a,b)=>a+b,0);return {A,belief,preferences,policies:policies.map((p,i)=>({...p,probability:weights[i]/sum})),chosen:policies[0].G<=policies[1].G?0:1};
}
export const TASK_TARGETS=[[1,0],[-1,1],[0,-1]];
export function retention(method='plain',strength=1){
 if(!['plain','ewc','replay','isolation'].includes(method)||!Number.isFinite(strength)||strength<0)throw new RangeError('Invalid retention configuration.');
 let theta=[0,0];const states=[],matrix=[];
 TASK_TARGETS.forEach((target,t)=>{if(method==='isolation'){states.push([...target]);matrix.push(TASK_TARGETS.map((v,j)=>j<=t?1:null));return;}
 theta=target.map((v,k)=>{if(t===0||method==='plain')return v;if(method==='ewc'){const p=strength*[5,.2][k];return (v+p*theta[k])/(1+p);}const old=TASK_TARGETS.slice(0,t).reduce((s,a)=>s+a[k],0)/t;return (v+strength*old)/(1+strength);});
 states.push([...theta]);matrix.push(TASK_TARGETS.map(v=>Math.exp(-v.reduce((s,a,k)=>s+(a-theta[k])**2,0)/2)));});
 const final=matrix.at(-1),forgetting=[0,1].map(j=>Math.max(...matrix.slice(j,2).map(row=>row[j]))-final[j]),backward=[0,1].map(j=>final[j]-matrix[j][j]);
 return {states,matrix,forgetting,backward,meanForgetting:forgetting.reduce((a,b)=>a+b,0)/2};
}
export function memoryLifecycle(operations=[]){
 let state={context:false,record:false,adapter:false,saved:false,ttt:false};const history=[{operation:'Base',...state}];
 for(const operation of operations){if(operation==='inject')state={context:true,record:true,adapter:true,saved:true,ttt:true};if(operation==='session')state={...state,context:false};if(operation==='parameters')state={...state,adapter:false,ttt:false};if(operation==='reload')state={...state,adapter:state.saved};if(operation==='retrieve')state={...state,context:state.record};history.push({operation,...state});}return history;
}
export const VERIFIER_CANDIDATES=[{id:'A',steps:[[2,3,5],[5,4,20]],answer:20,summary:'2 + 3 = 5; 5 × 4 = 20'},{id:'B',steps:[[2,3,6],[6,4,20]],answer:20,summary:'2 + 3 = 6; 6 × 4 = 20'},{id:'C',steps:[[2,3,5],[5,4,9]],answer:9,summary:'2 + 3 = 5; 5 × 4 = 9'}];
export function verifyCandidates(strict=false){return VERIFIER_CANDIDATES.map(c=>{const valid=c.steps[0][0]+c.steps[0][1]===c.steps[0][2]&&c.steps[1][0]*c.steps[1][1]===c.steps[1][2]&&c.steps[0][2]===c.steps[1][0];return {...c,valid,accepted:c.answer===20&&(!strict||valid)};});}
