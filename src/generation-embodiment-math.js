// Transparent numerical examples. None of these helpers executes a learned model.
export function deterministicDenoise(x, alpha, previousAlpha, predictedNoise) {
  if (!(alpha > 0 && alpha < 1 && previousAlpha >= alpha && previousAlpha <= 1)) throw new RangeError('Require 0 < alpha < 1 and alpha <= previousAlpha <= 1');
  const clean = (x - Math.sqrt(1 - alpha) * predictedNoise) / Math.sqrt(alpha);
  return {clean, next: Math.sqrt(previousAlpha) * clean + Math.sqrt(1 - previousAlpha) * predictedNoise};
}

// Euler for dx/dt=1, dy/dt=x; exact solution from (0,0) is (t,t²/2).
export function flowTrace(steps) {
  if (!Number.isInteger(steps) || steps < 1) throw new RangeError('Positive integer steps required');
  const points = [[0, 0]];
  for (let k = 0; k < steps; k++) {
    const [x,y] = points.at(-1);
    points.push([x + 1 / steps, y + x / steps]);
  }
  return points;
}

const words = ['The','robot','moves','to','the','blue','cup','.'];
const baseConfidence = [.94,.65,.72,.90,.55,.83,.50,.96];
export function maskedTrace(rounds, remask = true) {
  let visible = Array(8).fill(null), acceptedConfidence = Array(8).fill(0);
  const trace = [{visible:[...visible], proposals:[], accepted:[], remasked:[]}];
  for (let round = 1; round <= rounds; round++) {
    const proposals = visible.flatMap((v,i) => v === null ? [{position:i,word:i===6 && round===1?'box':words[i],confidence:Math.min(.99,baseConfidence[i]+.03*(round-1)+.06*Number(Boolean(visible[i-1]))+.06*Number(Boolean(visible[i+1])))}] : []);
    const accepted = [...proposals].sort((a,b)=>b.confidence-a.confidence || a.position-b.position).slice(0,2);
    const old = visible.flatMap((v,i)=>v!==null?[i]:[]);
    accepted.forEach(p=>{visible[p.position]=p.word;acceptedConfidence[p.position]=p.confidence;});
    const remasked = remask && round===2 && old.length ? [old.sort((a,b)=>acceptedConfidence[a]-acceptedConfidence[b])[0]] : [];
    remasked.forEach(i=>{visible[i]=null;});
    trace.push({visible:[...visible], proposals, accepted:accepted.map(p=>p.position), remasked});
  }
  return trace;
}

export const mixtureEmbeddings = [[1,0],[0,2],[-1,0]];
export function embeddingMixture(firstWeight) {
  const weights = [firstWeight, .6*(1-firstWeight), .4*(1-firstWeight)];
  return {weights,vector:[0,1].map(d=>weights.reduce((sum,p,i)=>sum+p*mixtureEmbeddings[i][d],0))};
}
export function latentTrace(steps, bypass = true) {
  const states = [[.2,.8]], inputs=[];
  for (let k=0;k<steps;k++) {
    const h=states.at(-1),logits=[h[0],h[1],-h[0]-h[1]],token=logits.indexOf(Math.max(...logits));
    const embedding=bypass?h:mixtureEmbeddings[token];
    inputs.push({token,embedding:[...embedding]});
    states.push([.5*embedding[0]+.4,.5*embedding[1]-.1]);
  }
  return {states,inputs};
}
export function quantizeCoordinate(value, lo=-10, hi=10, bins=8) {
  if (!(hi>lo && Number.isInteger(bins) && bins>0)) throw new RangeError('Invalid quantization interval');
  const width=(hi-lo)/bins,bin=Math.max(0,Math.min(bins-1,Math.floor((value-lo)/width)));
  return {bin,width,decoded:lo+(bin+.5)*width};
}

export function actionTrace(mode, steps=4, firstCoordinate=2) {
  if (!Number.isInteger(steps) || steps<1) throw new RangeError('Positive integer steps required');
  const center=[[firstCoordinate,1],[4,2],[6,3],[8,4]],noise=[[-6,4],[3,-5],[-2,6],[5,-3]];
  if(mode==='tokens') return {center,noise,chunk:center.map(row=>row.map(v=>quantizeCoordinate(v).decoded)),history:[],bins:center.map(row=>row.map(v=>quantizeCoordinate(v).bin))};
  let chunk=noise.map(row=>[...row]); const history=[chunk];
  // Synthetic conditioned functions: flow contracts to μ(c), denoiser predicts μ(c).
  for(let k=0;k<steps;k++) {
    if(mode==='flow') chunk=chunk.map((row,i)=>row.map((v,j)=>v+(center[i][j]-v)/steps));
    else if(mode==='diffusion') {
      const a=.1+.9*k/steps,previous=.1+.9*(k+1)/steps;
      // At k=0 the state is the analytically corrupted conditional center.
      if(k===0) {chunk=center.map((row,i)=>row.map((v,j)=>Math.sqrt(a)*v+Math.sqrt(1-a)*noise[i][j]));history[0]=chunk;}
      chunk=chunk.map((row,i)=>row.map((v,j)=>deterministicDenoise(v,a,previous,(v-Math.sqrt(a)*center[i][j])/Math.sqrt(1-a)).next));
    } else throw new RangeError('Unknown sampler');
    history.push(chunk);
  }
  return {center,noise,chunk,history};
}

export function rolloutTiming(inference=40, prefix=2, pipelined=false) {
  const capture=12,transport=8,dispatch=5,period=20;
  const first={capture:[0,capture],transport:[capture,capture+transport],inference:[capture+transport,capture+transport+inference],dispatch:[capture+transport+inference,capture+transport+inference+dispatch]};
  first.use=first.dispatch[1]; first.age=first.use; first.execute=[first.use,first.use+prefix*period];
  const start=pipelined?25:first.execute[1],sensorsEnd=start+capture+transport;
  const inferenceStart=Math.max(sensorsEnd,first.inference[1]);
  const second={capture:[start,start+capture],transport:[start+capture,sensorsEnd],inference:[inferenceStart,inferenceStart+inference],dispatch:[inferenceStart+inference,inferenceStart+inference+dispatch]};
  second.use=Math.max(second.dispatch[1],first.execute[1]);second.age=second.use-start;second.execute=[second.use,second.use+prefix*period];
  return {frames:[first,second],period,queue:inferenceStart-sensorsEnd};
}
export function wilsonInterval(successes,trials,z=1.96) {
  if(!Number.isInteger(trials) || trials<1 || !Number.isInteger(successes) || successes<0 || successes>trials) throw new RangeError('Invalid counts');
  const p=successes/trials,denom=1+z*z/trials,center=(p+z*z/(2*trials))/denom,half=z*Math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denom;
  return [Math.max(0,center-half),Math.min(1,center+half)];
}
export function jepaExample(mask='block', error=1, momentum=.9) {
  const positions=mask==='block'?[7,8,12,13]:[1,8,11,18];
  const rows=positions.map(position=>{const target=[position/10,1-position/20],prediction=[target[0]+.2*error,target[1]-.1*error];return {position,target,prediction,absolute:prediction.map((v,i)=>Math.abs(v-target[i]))};});
  const mean=rows.reduce((s,r)=>s+r.absolute.reduce((a,b)=>a+b,0),0)/(rows.length*2);
  const featureMean=[0,1].map(j=>rows.reduce((s,r)=>s+r.target[j],0)/rows.length);
  const variance=rows.reduce((s,r)=>s+r.target.reduce((t,v,j)=>t+(v-featureMean[j])**2,0),0)/(rows.length*2);
  return {rows,loss:mean,variance,ema:[momentum*.4+(1-momentum)*1,momentum*.8+(1-momentum)*.2]};
}
export const planningCandidates=[{name:'A',points:[0,-18,-35]},{name:'B',points:[0,10,34]},{name:'C',points:[0,6,12]}];
export function rankPlans(goal) {
  return planningCandidates.map(c=>({...c,score:Math.abs(c.points.at(-1)-goal)})).sort((a,b)=>a.score-b.score || a.name.localeCompare(b.name));
}
