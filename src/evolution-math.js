// Synthetic deterministic evolutionary examples, not benchmark implementations.
const clone=g=>g.map(a=>({...a}));
export const parentA=[{id:1,from:1,to:3,w:.8,on:false},{id:2,from:2,to:3,w:-.2,on:true},{id:3,from:1,to:4,w:1,on:true},{id:4,from:4,to:3,w:.8,on:true}];
export const parentB=[{id:1,from:1,to:3,w:1,on:true},{id:2,from:2,to:3,w:.2,on:true},{id:3,from:1,to:4,w:.8,on:true},{id:5,from:2,to:4,w:.4,on:true},{id:6,from:4,to:3,w:.5,on:true}];
export function alignGenes(a=parentA,b=parentB){const maxA=Math.max(...a.map(g=>g.id),0),maxB=Math.max(...b.map(g=>g.id),0);return [...new Set([...a,...b].map(g=>g.id))].sort((a,b)=>a-b).map(id=>{const A=a.find(g=>g.id===id),B=b.find(g=>g.id===id);return {id,A,B,type:A&&B?'matching':id>Math.min(maxA,maxB)?'excess':'disjoint'};});}
export function compatibility(a=parentA,b=parentB){const rows=alignGenes(a,b),matching=rows.filter(r=>r.type==='matching'),E=rows.filter(r=>r.type==='excess').length,D=rows.filter(r=>r.type==='disjoint').length,W=matching.length?matching.reduce((s,r)=>s+Math.abs(r.A.w-r.B.w),0)/matching.length:0,N=Math.max(a.length,b.length)<20?1:Math.max(a.length,b.length);return {E,D,W,N,distance:(E+D)/N+.4*W};}
export function crossover(a=parentA,b=parentB,fitter='A',matchingParent='A'){return alignGenes(a,b).flatMap(r=>{const g=r.type==='matching'?(matchingParent==='A'?r.A:r.B):(fitter==='A'?r.A:r.B);if(!g)return [];return [{...g,on:r.A&&r.B&&(!r.A.on||!r.B.on)?false:g.on}];});}
export function genomeOutput(genes,x1,x2){const node={1:x1,2:x2,4:0,3:0};for(const target of [4,3]){let sum=0;for(const g of genes)if(g.on&&g.to===target)sum+=node[g.from]*g.w;node[target]=target===4?Math.tanh(sum):sum;}return node[3];}
export function splitGene(genes){const g=clone(genes),old=g.find(a=>a.id===1);if(!old||!old.on)return g;old.on=false;g.push({id:3,from:1,to:4,w:1,on:true},{id:4,from:4,to:3,w:old.w,on:true});return g;}
const task=[[-1,-1],[-.5,1],[.5,-1],[1,1]];
export function genomeFitness(g){const mse=task.reduce((s,[x,y])=>s+(genomeOutput(g,x,y)-(.8*Math.tanh(x)-.2*y))**2,0)/task.length;return {mse,fitness:1/(1+mse)};}
export function neatGeneration(n=0){
 if(!Number.isInteger(n)||n<0||n>30)throw new RangeError('Generation outside toy budget');
 let population=[.2,.5,.8,1.1].map((w,i)=>({name:'P'+(i+1),genes:[{id:1,from:1,to:3,w,on:true},{id:2,from:2,to:3,w:i%2?.1:-.2,on:true}]}));
 for(let g=0;g<n;g++){
  const ranked=population.map(p=>({...p,...genomeFitness(p.genes)})).sort((a,b)=>b.fitness-a.fitness),best=ranked[0].genes,second=ranked[1].genes;
  population=[{name:'elite',genes:clone(best)},{name:'split',genes:splitGene(best)},{name:'child +',genes:crossover(best,second,'A','B').map(a=>({...a,w:a.w+.16*Math.sin((g+1)*1.7+a.id)}))},{name:'child −',genes:clone(best).map(a=>({...a,w:a.w-.16*Math.sin((g+1)*1.7+a.id)}))}];
 }
 const evaluated=population.map(p=>({...p,...genomeFitness(p.genes)})).sort((a,b)=>b.fitness-a.fitness);return {population:evaluated,best:evaluated[0]};
}
export function cppn(s,t,params={a:1,b:0,c:0}){const d=s-t,g=Math.exp(-2*d*d),p=Math.sin(Math.PI/2*(s+t)),linear=t,weight=params.a*g+params.b*p+params.c*linear-.3;return {d,g,p,linear,weight};}
export function coordinates(size=3){if(![3,5].includes(size))throw new RangeError('Toy substrate size must be 3 or 5');return Array.from({length:size},(_,i)=>-1+2*i/(size-1));}
export function substrate(params,size=3){const coords=coordinates(size),weights=coords.map(s=>coords.map(t=>cppn(s,t,params).weight)),input=coords.map(x=>x),output=coords.map((_,j)=>coords.reduce((s,x,i)=>s+x*weights[i][j],0)),mse=output.reduce((s,y,j)=>s+(y-coords[j])**2,0)/size;return {coords,weights,input,output,mse,fitness:1/(1+mse)};}
export function hyperGeneration(n=0,symmetric=true,size=3){
 if(!Number.isInteger(n)||n<0||n>30)throw new RangeError('Generation outside toy budget');
 let candidates=[.5,.8,1,1.3].map(a=>({a,b:symmetric?0:.25,c:symmetric?0:.35}));let scored;
 for(let g=0;g<=n;g++){
  scored=candidates.map(params=>({params,...substrate(params,size)})).sort((a,b)=>b.fitness-a.fitness);
  if(g<n){const p=scored[0].params,d=.2/(1+.15*g);candidates=[p,{...p,a:p.a+d},{...p,a:p.a-d},{...p,b:symmetric?0:p.b-d,c:symmetric?0:p.c-d}];}
 }
 return {candidates:scored,best:scored[0]};
}
