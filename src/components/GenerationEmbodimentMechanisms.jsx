import React,{useId,useState} from 'react';
import {Figure,Box,ArrowDefs,DiffusionFigure} from './Figures';
import {JepaFigure,WorldModelFigure} from './JepaFigures';
import {ConceptFigure} from './ConceptFigures';
import {deterministicDenoise,flowTrace,maskedTrace,embeddingMixture,latentTrace,quantizeCoordinate,actionTrace,rolloutTiming,wilsonInterval,jepaExample,planningCandidates,rankPlans} from '../generation-embodiment-math';

const f=(x,n=2)=>Number(x).toFixed(n),vec=v=>`[${v.map(x=>f(x)).join(', ')}]`;
const colors=['var(--blue)','var(--orange)','var(--green)'];
function Text({x,y,children,size=14,color='var(--ink)',anchor='middle',...rest}){return <text x={x} y={y} fill={color} fontSize={size} textAnchor={anchor} fontFamily="var(--sans)" {...rest}>{children}</text>;}
function Wire({d,id,color='var(--fig-line)',dashed=false}){return <path d={d} fill="none" stroke={color} strokeWidth="1.7" strokeDasharray={dashed?'6 4':undefined} markerEnd={id?`url(#${id})`:undefined}/>;}
function Range({label,value,onChange,min=0,max=1,step=.1}){return <label className="range-control">{label} <output>{value}</output><input type="range" aria-label={label} value={value} min={min} max={max} step={step} onChange={e=>onChange(Number(e.target.value))}/></label>;}
function Choices({label,value,onChange,options}){return <div className="segments" role="group" aria-label={label}>{options.map(([v,t])=><button key={v} aria-pressed={v===value} className={v===value?'selected':''} onClick={()=>onChange(v)}>{t}</button>)}</div>;}
function Svg({id,title,description,children,height=650}){return <svg viewBox={`0 0 1000 ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}><title id={`${id}-title`}>{title}</title><desc id={`${id}-desc`}>{description}</desc><ArrowDefs id={id}/>{children}</svg>;}
function Card({x,y,w=210,h=52,title,sub,color='var(--fig-line)'}){return <g><Box x={x} y={y} w={w} h={h} label="" color={color}/><Text x={x} y={y+22} size={15}>{title}</Text>{sub&&<Text x={x} y={y+42} size={12} color="var(--muted)">{sub}</Text>}</g>;}
function Disclosure({title,children}){return <details style={{marginTop:18}}><summary>{title}</summary>{children}</details>;}

export function DiffusionFlowMechanism(){
 const id=useId().replace(/:/g,'')+'generation';
 const [eps,setEps]=useState(1),[steps,setSteps]=useState(4),[round,setRound]=useState(2),[remask,setRemask]=useState(true);
 const x=.5*2+Math.sqrt(.75),d=deterministicDenoise(x,.25,.64,eps),trace=flowTrace(steps),m=maskedTrace(round,remask).at(-1);
 const px=x=>565+330*x,py=y=>280-310*y;
 return <><Figure number="12" title="Learn a reverse operation; choose a sampling rule" controls={<><Range label="Predicted Gaussian noise" value={eps} onChange={setEps} min={0} max={2} step={.1}/><Range label="Euler steps" value={steps} onChange={setSteps} min={1} max={8} step={1}/><Range label="Mask-predict rounds" value={round} onChange={setRound} min={0} max={5} step={1}/><label className="small-note"><input type="checkbox" checked={remask} onChange={e=>setRemask(e.target.checked)}/> Remask one old token in round 2</label></>} caption="Three independent, invented examples. Gaussian view uses a known training pair and an editable noise estimate; its deterministic reverse update omits stochastic variance. Flow view integrates an explicitly defined field, not a fitted network. Mask predictions use the printed arithmetic rule; they are neither LLaDA outputs nor calibrated probabilities.">
 <Svg id={id} title="Gaussian reverse arithmetic, curved flow integration and masked-token revision" description={`Gaussian clean estimate ${f(d.clean)} and reverse state ${f(d.next)}. ${steps} Euler steps end at ${vec(trace.at(-1))}. After ${round} mask evaluations ${m.visible.filter(Boolean).length} of eight positions are accepted.`} height={690}>
  <Text x={255} y={31} size={18}>Gaussian: a noise target trains a denoiser</Text>
  <Text x={255} y={64}>Training pair: clean = 2; ε = 1; ᾱ = 0.25</Text>
  <Text x={255} y={94}>x = √0.25 × 2 + √0.75 × 1 = {f(x,3)}</Text>
  <Card x={130} y={121} w={190} title={`ε̂θ = ${f(eps)}`} sub={`noise loss (1 − ε̂)² = ${f((1-eps)**2)}`} color="var(--purple)"/>
  <Wire d="M225 147 H270" id={id}/><Card x={370} y={121} w={190} title={`clean estimate ${f(d.clean,3)}`} sub="(x − √0.75 ε̂) / √0.25" color="var(--blue)"/>
  <Wire d="M370 174 V196" id={id}/>
  <Card x={255} y={203} w={390} title={`Less noisy level ᾱ′ = 0.64 → x′ = ${f(d.next,3)}`} sub="x′ = √0.64 × clean estimate + √0.36 × ε̂" color="var(--green)"/>
  <Text x={255} y={283} size={12}>Inference has x and ε̂θ; the true clean sample is unavailable.</Text>
  <Text x={755} y={31} size={18}>Flow: solver steps follow a vector field</Text>
  <Text x={755} y={61} size={13}>Training pair z = (0,0), y = (1,0.5): target y − z</Text>
  <Text x={755} y={84} size={13}>Independent inference toy: v(x,y,t) = (1,x)</Text>
  <Wire d="M565 98 V280 H940"/>
  {[0,.25,.5,.75,1].map(a=><g key={a}><Text x={px(a)} y={300} size={11}>{a}</Text>{[0,.25,.5].map(b=><Wire key={b} d={`M${px(a)} ${py(b)} l24 ${-24*a}`} id={id} color="var(--border)"/>)}</g>)}
  {[0,.25,.5].map(y=><Text key={y} x={547} y={py(y)+4} size={11}>{y}</Text>)}
  <path d={`M${px(0)} ${py(0)} ${Array.from({length:21},(_,i)=>`L${px(i/20)} ${py((i/20)**2/2)}`).join(' ')}`} fill="none" stroke="var(--muted)" strokeDasharray="5 4"/>
  <polyline points={trace.map(([a,b])=>`${px(a)},${py(b)}`).join(' ')} fill="none" stroke="var(--purple)" strokeWidth="3"/>
  {trace.map(([a,b],i)=><circle key={i} cx={px(a)} cy={py(b)} r="4" fill="var(--purple)"/>)}
  <Text x={755} y={325} size={13}>Solid Euler endpoint {vec(trace.at(-1))}; exact (1, 0.50)</Text>
  <Text x={755} y={348} size={12}>Δt = 1/{steps}; x ← x + Δt; y ← y + Δt × old x</Text>
  <path d="M35 367 H965" stroke="var(--border)"/>
  <Text x={500} y={397} size={18}>Masked prediction: evaluate every masked position, accept only two</Text>
  {Array.from({length:8},(_,i)=>{const p=m.proposals.find(p=>p.position===i),accepted=m.accepted.includes(i),reopened=m.remasked.includes(i);return <g key={i}>
   <Text x={77+i*121} y={425} size={12}>position {i+1}</Text>
   <Card x={77+i*121} y={438} w={105} h={50} title={m.visible[i]??'[MASK]'} sub={reopened?'remasked':accepted?'accepted now':m.visible[i]?'retained':'not accepted'} color={reopened?'var(--orange)':accepted?'var(--green)':'var(--fig-line)'}/>
   <Text x={77+i*121} y={517} size={13}>{p?`${p.word} · ${f(p.confidence)}`:'—'}</Text>
  </g>;})}
  <Text x={500} y={550} size={13}>Round {round}: {m.proposals.length} proposals; {m.accepted.length} accepts; {m.remasked.length} old token reopened.</Text>
  <Text x={500} y={580} size={12}>Toy confidence = min(0.99, baseᵢ + 0.03(round − 1) + 0.06 × visible neighbors)</Text>
  <Text x={500} y={606} size={12}>base = [0.94, 0.65, 0.72, 0.90, 0.55, 0.83, 0.50, 0.96]; accept largest two.</Text>
  <Text x={500} y={632} size={12}>At round 2, optionally reopen the lower-confidence old token. Draft at position 7: “box” in round 1, then “cup”.</Text>
  <Text x={500} y={659} size={12}>A network evaluation makes proposals; an acceptance schedule decides the next conditioning state.</Text>
 </Svg></Figure><Disclosure title="Supporting view: left-to-right versus fixed masked reveal order"><DiffusionFigure/></Disclosure></>;
}

export function SoftLatentMechanism(){
 const id=useId().replace(/:/g,'')+'latent';
 const [prompt,setPrompt]=useState(0),[weight,setWeight]=useState(.5),[mode,setMode]=useState('latent'),[steps,setSteps]=useState(2),[stage,setStage]=useState(1);
 const mix=embeddingMixture(weight),grad=prompt-.5,next=prompt-.25*grad,latent=latentTrace(steps,mode==='latent');
 return <Figure number="13" title="A learned prompt, a convex mixture and a latent step compute different things" controls={<><Range label="Scalar prompt P" value={prompt} onChange={setPrompt} min={-1} max={1} step={.1}/><Range label="Mixture weight p₁" value={weight} onChange={setWeight} min={0} max={1} step={.1}/><Choices label="Next embedding construction" value={mode} onChange={setMode} options={[["tokens","Decode → embed"],["latent","Bypass vocabulary"]]}/><Range label="Feedback steps" value={steps} onChange={setSteps} min={1} max={3} step={1}/><Range label="Curriculum stage k" value={stage} onChange={setStage} min={0} max={3} step={1}/></>} caption="All numbers are synthetic width-two or scalar examples, not Coconut activations. Prompt and mixture columns show exact arithmetic. The lower trace uses hnext = 0.5 × input + [0.4, −0.1], with a retained prefix at every position. Coconut's trained language/latent boundaries and gradual replacement curriculum motivate the timeline; this toy recurrence makes no reasoning claim.">
 <Svg id={id} title="Prompt-only gradient, calculated embedding mixture and normal versus latent feedback" description={`Prompt gradient ${f(grad)}, updated P ${f(next)}. Mixture ${vec(mix.vector)}. ${steps} ${mode} steps end at ${vec(latent.states.at(-1))}. Curriculum stage ${stage} replaces that many of three language reasoning steps.`} height={705}>
  <Text x={245} y={30} size={18}>Task parameters: only P learns</Text>
  <Text x={245} y={63}>Frozen toy backbone: ŷ = P + 0.5</Text>
  <Text x={245} y={91}>Target 1; L = ½(ŷ − 1)² = {f(.5*grad*grad,3)}</Text>
  <Card x={120} y={116} w={160} title={`P = ${f(prompt)}`} sub="trainable prompt" color="var(--purple)"/>
  <Wire d="M200 141 H275" id={id}/><Card x={365} y={116} w={180} title={`ŷ = ${f(prompt+.5)}`} sub="θ stays fixed" color="var(--blue)"/>
  <Wire d="M365 170 V215 H120 V176" id={id} color="var(--green)" dashed/>
  <Text x={245} y={205} size={13}>∂L/∂P = {f(grad)}; learning rate 0.25</Text>
  <Text x={245} y={244}>One update: P′ = {f(next,3)}; Δθ = 0</Text>
  <Text x={750} y={30} size={18}>Probability mixture: v = Σ pᵢEᵢ</Text>
  <Text x={750} y={63}>E₁ = [1,0]; E₂ = [0,2]; E₃ = [−1,0]</Text>
  <Text x={750} y={91}>p = {vec(mix.weights)}; Σp = 1</Text>
  <Wire d="M600 220 H895 M745 230 V110"/>
  <polygon points="670,220 745,120 820,220" fill="var(--accent-soft)" stroke="var(--border)"/>
  <Text x={655} y={238} size={12}>E₃</Text><Text x={745} y={111} size={12}>E₂</Text><Text x={835} y={238} size={12}>E₁</Text>
  <circle cx={745+75*mix.vector[0]} cy={220-50*mix.vector[1]} r="6" fill="var(--purple)"/>
  <Text x={750} y={269}>Calculated v = {vec(mix.vector)}</Text>
  <path d="M35 292 H965" stroke="var(--border)"/>
  <Text x={500} y={322} size={18}>{mode==='latent'?'Latent mode skips language-head decoding and token embedding':'Language mode chooses a token, then looks up its embedding'}</Text>
  <Card x={120} y={348} w={165} title="Final-position h" sub="after normalization" color="var(--blue)"/>
  {mode==='tokens'?<><Wire d="M203 374 H255" id={id}/><Card x={365} y={348} w={220} title="softmax(W h) → argmax" sub="W rows: (1,0), (0,1), (−1,−1)" color="var(--orange)"/><Wire d="M475 374 H525" id={id}/><Card x={625} y={348} w={200} title="Vocabulary embedding Eᵢ" sub="same E₁, E₂, E₃ as above" color="var(--orange)"/><Wire d="M725 374 H780" id={id}/></>:<><Wire d="M203 374 H780" id={id} color="var(--purple)"/><Text x={490} y={361} size={14}>direct vector h → next input; no sampled word</Text></>}
  <Card x={875} y={348} w={190} title="Next model position" sub="retain all earlier K/V states" color="var(--green)"/>
  <Text x={500} y={432} size={13}>Synthetic h₀ = [0.20, 0.80]; hnext = 0.5 × next input + [0.4, −0.1]</Text>
  {latent.states.map((h,i)=><g key={i}><Card x={130+i*245} y={452} w={205} title={`h${i} = ${vec(h)}`} sub={i===0?'prefix ends at entry marker':mode==='latent'?`input was h${i-1}`:`token ${latent.inputs[i-1].token+1}: ${vec(latent.inputs[i-1].embedding)}`} color={i===steps?'var(--green)':'var(--fig-line)'}/>{i<steps&&<Wire d={`M${233+i*245} 478 H${270+i*245}`} id={id}/>}</g>)}
  <Text x={500} y={533} size={13}>Coconut layout: question → &lt;bot&gt; → {steps} latent positions → &lt;eot&gt; → normal answer tokens</Text>
  <Text x={500} y={562} size={12}>Prefix/cache is appended to, never replaced by a single state. The next state comes from the final position only.</Text>
  <Text x={500} y={607} size={18}>Training curriculum: c = 1, three original reasoning steps</Text>
  {Array.from({length:3},(_,i)=><Card key={i} x={255+i*245} y={629} w={220} title={i<stage?`Latent position ${i+1}`:`Language step ${i+1}`} sub={i<stage?'no token cross-entropy here':'supervise remaining text'} color={i<stage?'var(--purple)':'var(--orange)'}/>)}
  <Text x={500} y={696} size={12}>Stage {stage}: {stage} language steps replaced; answer remains supervised. This training changes model parameters.</Text>
 </Svg></Figure>;
}

function ActionMechanism({foundation=false}){
 const id=useId().replace(/:/g,'')+'action';
 const [mode,setMode]=useState('tokens'),[steps,setSteps]=useState(4),[h,setH]=useState(2),[coordinate,setCoordinate]=useState(2);
 const toy=actionTrace(mode,steps,coordinate),q=quantizeCoordinate(coordinate),H=4;
 const paths=mode==='tokens'?[]:toy.history.map((chunk,k)=>[k/steps,chunk[0][0]]);
 const chartX=t=>625+290*t,chartY=a=>415-7*a;
 return <Figure number={foundation?'14':'15'} title={foundation?'A VLA action interface closes both sensor loops':'Action tokens and action-chunk samplers have separate clocks'} controls={<><Choices label="Action representation" value={mode} onChange={setMode} options={[["tokens","Action tokens"],["diffusion","Noise prediction"],["flow","Velocity flow"]]}/><Range label="Sampling steps N" value={steps} onChange={setSteps} min={1} max={6} step={1}/><Range label="First Δx coordinate (mm)" value={coordinate} onChange={setCoordinate} min={-10} max={10} step={1}/><Range label="Executed prefix h" value={h} onChange={setH} min={1} max={4} step={1}/></>} caption="Generic synthetic policy, not a reconstruction of a named VLA. Each action has two Cartesian displacement coordinates in mm; the controller convention is assumed. H = 4, dₐ = 2, physical period = 20 ms. Eight bins make quantization visible; RT-2/OpenVLA use 256. The diffusion toy uses an analytic conditional center and zero stochastic variance; the flow toy uses v = μ(c) − a. Sampling compute and physical execution are separate.">
 <Svg id={id} title="Token bins or iterative action sampling followed by prefix execution and refreshed sensors" description={`${mode} produces four two-dimensional actions. First action ${vec(toy.chunk[0])} millimeters; execute ${h} actions over ${h*20} milliseconds. Feedback refreshes both camera observations and proprioception before synchronization.`} height={735}>
  <Card x={135} y={32} w={200} title="Camera oₜ₊₁" sub="fresh image timestamp" color="var(--blue)"/>
  <Card x={135} y={106} w={200} title="Robot state qₜ₊₁" sub="proprioception timestamp" color="var(--blue)"/>
  <Wire d="M235 58 H270 V96 H300 M235 132 H270 V96" id={id}/>
  <Card x={390} y={69} w={180} title="Synchronize" sub="o, q share a reference time" color="var(--blue)"/>
  <Wire d="M480 95 H520" id={id}/><Card x={625} y={69} w={210} title="Vision–language policy" sub="instruction ℓ persists" color="var(--purple)"/>
  <Wire d="M730 95 H770" id={id}/><Card x={870} y={69} w={200} title={mode==='tokens'?'Token decoder':mode==='diffusion'?'Noise predictor':'Velocity expert'} sub={mode==='tokens'?'bins → physical coordinates':'iterative H × dₐ state'} color="var(--orange)"/>
  <Text x={500} y={192} size={18}>{mode==='tokens'?'Coordinate → bin → predicted token → dequantized command':mode==='diffusion'?'Noise predictions drive a deterministic reverse schedule':'An explicit velocity field changes the whole chunk'}</Text>
  {mode==='tokens'?<>
   <Text x={500} y={224}>Toy interval [−10,10] mm; B = 8; width w = 2.5 mm</Text>
   {Array.from({length:8},(_,i)=><g key={i}><rect x={100+i*100} y={245} width="100" height="65" fill={i===q.bin?'var(--accent-soft)':'var(--surface)'} stroke={i===q.bin?'var(--accent)':'var(--border)'}/><Text x={150+i*100} y={269} size={14}>bin {i}</Text><Text x={150+i*100} y={294} size={12}>{f(-10+(i+.5)*2.5)} mm</Text></g>)}
   <Text x={500} y={341}>b = clip(floor(({coordinate} + 10) / 2.5), 0, 7) = {q.bin}; decode = −10 + ({q.bin}+½)2.5 = {f(q.decoded)} mm</Text>
   <Text x={500} y={376} size={14}>Autoregressive coordinate order: a₀ˣ → a₀ʸ → a₁ˣ → a₁ʸ → a₂ˣ → a₂ʸ → a₃ˣ → a₃ʸ</Text>
   <Text x={500} y={409} size={14}>This toy's token targets: {toy.bins.flat().join(' → ')}; action supervision: −Σⱼ log pθ(bⱼ | c,b&lt;ⱼ)</Text>
   <Text x={500} y={441} size={12}>One model action can need several token predictions. Eight coordinates here do not imply eight robot time steps.</Text>
  </>:<>
   <Text x={280} y={228} size={14}>Conditional center μ(c) = [{toy.center.map(v=>`(${v.join(',')})`).join(', ')}] mm</Text>
   <Text x={280} y={259} size={14}>Noise seed = [(−6,4), (3,−5), (−2,6), (5,−3)]</Text>
   <Text x={280} y={295} size={14}>{mode==='flow'?'v(a,τ,c) = μ(c) − a;  Δτ = 1/N':'ᾱₖ = 0.1 + 0.9k/N; denoiser predicts μ(c)'}</Text>
   <Text x={280} y={326} size={14}>{mode==='flow'?'aₖ₊₁ = aₖ + (μ(c) − aₖ)/N':'ε̂ = (aₖ − √ᾱₖ μ(c)) / √(1−ᾱₖ)'}</Text>
   <Text x={280} y={358} size={14}>{mode==='flow'?'Finite N contracts toward the center; it need not arrive.':'aₖ₊₁ = √ᾱₖ₊₁ μ(c) + √(1−ᾱₖ₊₁) ε̂'}</Text>
   <Text x={280} y={397} size={13}>All H × dₐ entries update in each synthetic evaluation.</Text>
   <Text x={280} y={427} size={12}>A real learned predictor can be wrong; no demonstration endpoint is supplied at inference.</Text>
   <Wire d="M625 260 V460 H950"/>
   {[0,.5,1].map(t=><Text key={t} x={chartX(t)} y={480} size={12}>τ={t}</Text>)}
   {[-6,0,6,10].map(v=><g key={v}><Text x={606} y={chartY(v)+4} size={12}>{v}</Text><path d={`M625 ${chartY(v)} H935`} stroke="var(--border)"/></g>)}
   <polyline points={paths.map(([t,a])=>`${chartX(t)},${chartY(a)}`).join(' ')} fill="none" stroke="var(--purple)" strokeWidth="3"/>
   {paths.map(([t,a],i)=><g key={i}><circle cx={chartX(t)} cy={chartY(a)} r="4" fill="var(--purple)"/><Text x={chartX(t)} y={chartY(a)-12} size={11}>{f(a,1)}</Text></g>)}
   <Text x={785} y={244} size={14}>First coordinate over generative time τ</Text>
  </>}
  <path d="M35 498 H965" stroke="var(--border)"/>
  <Text x={500} y={527} size={18}>Aₜ ∈ ℝ⁴ˣ²: physical action index is a different axis</Text>
  {toy.chunk.map((row,i)=><g key={i}><Card x={210+i*205} y={548} w={185} h={58} title={`aₜ₊${i} = ${vec(row)}`} sub={i<h?`EXECUTE · ${i*20}–${(i+1)*20} ms`:'proposed; discarded on replan'} color={i<h?'var(--green)':'var(--fig-line)'}/></g>)}
  <Wire d={`M115 627 H${115+h*205}`} id={id} color="var(--green)"/>
  <Text x={500} y={654} size={14}>Execute h = {h} of H = 4 → next observation at physical t + {h*20} ms (sampling latency excluded)</Text>
  <Card x={500} y={672} w={450} title="Robot + environment → acquire both sensors again" sub="instruction remains task conditioning; synchronize refreshed o and q" color="var(--green)"/>
  <Wire d="M275 698 H18 V16 H135 V27 M18 16 V184 H135 V164" id={id} color="var(--green)"/>
 </Svg></Figure>;
}
export function VLAFoundationMechanism(){return <ActionMechanism foundation/>;}
export function ActionSamplingMechanism(){return <ActionMechanism/>;}

export function RobotEvaluationMechanism(){
 const id=useId().replace(/:/g,'')+'timing';
 const [pipeline,setPipeline]=useState('serial'),[inference,setInference]=useState(40),[prefix,setPrefix]=useState(2),[metric,setMetric]=useState('autonomous');
 const result=rolloutTiming(inference,prefix,pipeline==='pipeline'),frames=result.frames,total=Math.max(...frames.map(r=>r.execute[1])),scale=t=>100+790*t/total;
 const successes=metric==='autonomous'?1:2,interval=wilsonInterval(successes,3),v=.2,age=frames[1].age;
 const stages=[['capture','var(--blue)'],['transport','var(--orange)'],['inference','var(--purple)'],['dispatch','var(--green)'],['execute','var(--fig-line)']];
 return <><Figure number="16" title="Observation age and rollout denominators are measurable choices" controls={<><Choices label="Two-frame schedule" value={pipeline} onChange={setPipeline} options={[["serial","Sequential cycles"],["pipeline","Pipeline next frame"]]}/><Range label="Inference duration (ms)" value={inference} onChange={setInference} min={20} max={80} step={5}/><Range label="Actions per executed prefix" value={prefix} onChange={setPrefix} min={1} max={4} step={1}/><Choices label="Reported success definition" value={metric} onChange={setMetric} options={[["autonomous","Autonomous success"],["assisted","Completion incl. assistance"]]}/></>} caption="Invented two-frame schedule: capture 12 ms, transport 8 ms, one inference server, dispatch 5 ms, actions every 20 ms. Capture timestamp means start of exposure here. Pipeline frame 1 starts at 25 ms and may queue; dispatch can finish before the previous executed prefix ends. Observation age is measured at actual use. Displacement assumes constant 0.2 m/s and is neither a stability nor a safety model. Three toy trials are too few to estimate a real policy reliably.">
 <Svg id={id} title="Computed sensor-to-execution timing and transparent autonomous versus assisted outcomes" description={`Frame 1 begins at ${frames[1].capture[0]} milliseconds and is used at ${frames[1].use}; its observation age is ${age} milliseconds. ${successes} of three episodes meet the chosen success definition, with a Wilson interval from ${f(interval[0]*100,1)} to ${f(interval[1]*100,1)} percent.`} height={705}>
  <Text x={500} y={31} size={18}>Two observations; one inference server; physical execution is explicit</Text>
  {stages.map(([name,color],i)=><g key={name}><rect x={78+i*180} y={52} width="17" height="12" fill={color} opacity=".7"/><Text x={105+i*180} y={64} anchor="start" size={13}>{name}</Text></g>)}
  {[0,1,2,3,4].map(i=>{const t=total*i/4;return <g key={i}><path d={`M${scale(t)} 90 V313`} stroke="var(--border)"/><Text x={scale(t)} y={334} size={12}>{f(t,0)} ms</Text></g>;})}
  {frames.map((frame,i)=>{const y=112+i*111;return <g key={i}><Text x={81} y={y+18} anchor="end" size={13}>frame {i}</Text>{stages.map(([name,color],j)=>{const span=frame[name];return <g key={name}><rect x={scale(span[0])} y={y+(j===4?34:0)} width={Math.max(2,scale(span[1])-scale(span[0]))} height="24" rx="2" fill={color} opacity=".7"/><Text x={(scale(span[0])+scale(span[1]))/2} y={y+(j===4?51:17)} size={j===0||j===1||j===3?10:12} color="var(--ink)">{span[1]-span[0]}</Text></g>;})}<Wire d={`M${scale(frame.capture[0])} ${y+77} H${scale(frame.use)}`} id={id} color="var(--accent)"/><Text x={(scale(frame.capture[0])+scale(frame.use))/2} y={y+100} size={13}>age at use = {frame.age} ms</Text><path d={`M${scale(frame.use)} ${y-8} V${y+63}`} stroke="var(--accent)" strokeWidth="2"/></g>;})}
  <Text x={500} y={367} size={14}>Frame 1: capture timestamp {frames[1].capture[0]} ms → action use {frames[1].use} ms; inference queue {result.queue} ms</Text>
  <Text x={500} y={399} size={16}>Constant-velocity illustration: 0.2 m/s × {f(age/1000,3)} s = {f(v*age,1)} mm displacement</Text>
  <Text x={500} y={430} size={12}>Pipelining can improve throughput while increasing age through queueing or waiting for the old prefix.</Text>
  <path d="M35 450 H965" stroke="var(--border)"/>
  <Text x={500} y={479} size={18}>Fixed three-trial denominator; report assistance separately</Text>
  <Text x={145} y={510} size={13}>initial-state ID</Text><Text x={365} y={510} size={13}>termination / takeover</Text><Text x={640} y={510} size={13}>autonomous / complete</Text><Text x={865} y={510} size={13}>p95 delay</Text>
  {[["S01","goal confirmed / none","yes / yes","62 ms"],["S02","goal after takeover","no / yes","80 ms"],["S03","timeout / none","no / no","73 ms"]].map((r,i)=><g key={i}>{[145,365,640,865].map((x,j)=><Text key={j} x={x} y={542+i*34} size={14}>{r[j]}</Text>)}</g>)}
  <Text x={500} y={655} size={18}>{metric==='autonomous'?'Autonomous success':'Assisted-inclusive completion'} = {successes}/3 = {f(successes/3*100,1)}%; 95% Wilson [{f(interval[0]*100,1)}, {f(interval[1]*100,1)}]%</Text>
  <Text x={500} y={688} size={12}>Takeover rate = 1/3; timeout rate = 1/3; no partial credit. Each episode remains in the denominator.</Text>
 </Svg></Figure><Disclosure title="Rollout architecture: observations, policy, execution and logging"><ConceptFigure type="robot-evaluation"/></Disclosure></>;
}

export function JepaMechanism(){
 const id=useId().replace(/:/g,'')+'jepanumeric';
 const [mask,setMask]=useState('block'),[error,setError]=useState(1),[momentum,setMomentum]=useState(.9);
 const j=jepaExample(mask,error,momentum);
 return <><JepaFigure/><Figure number="17" title="Match target-position rows before reducing the feature loss" controls={<><Choices label="Numerical target-position set" value={mask} onChange={setMask} options={[["block","Block positions"],["scattered","Scattered positions"]]}/><Range label="Prediction error multiplier" value={error} onChange={setError} min={0} max={2} step={.1}/><Range label="EMA momentum m" value={momentum} onChange={setMomentum} min={0} max={.99} step={.01}/></>} caption="Synthetic feature width d = 2, four target positions, absolute-error reduction over all eight entries. Target row at position i is [i/10, 1−i/20]; prediction adds [0.2e, −0.1e]. These numbers are not trained embeddings. A separate two-parameter EMA example makes the target update explicit; its vectors are parameters, not feature rows.">
 <Svg id={id} title="Position-aligned feature regression with separate gradient and EMA updates" description={`Target positions ${j.rows.map(r=>r.position).join(', ')} have mean absolute prediction loss ${f(j.loss,3)}. Momentum ${momentum} updates target parameters to ${vec(j.ema)}. Gradient route reaches both predictor and context encoder.`} height={610}>
  <Text x={500} y={32} size={18}>Keep the same original position ID on both sides of the loss</Text>
  <Text x={130} y={72}>position i</Text><Text x={350} y={72}>predicted Ŷᵢ</Text><Text x={625} y={72}>target sg(Yᵢ)</Text><Text x={855} y={72}>|Ŷᵢ − Yᵢ|</Text>
  {j.rows.map((r,i)=><g key={r.position}><rect x="55" y={91+i*47} width="890" height="40" rx="4" fill={i%2?'var(--surface)':'var(--accent-soft)'}/><Text x={130} y={117+i*47}>{r.position}</Text><Text x={350} y={117+i*47}>{vec(r.prediction)}</Text><Text x={625} y={117+i*47}>{vec(r.target)}</Text><Text x={855} y={117+i*47}>{vec(r.absolute)}</Text></g>)}
  <Card x={785} y={302} w={300} title={`L₁ = sum / (4 × 2) = ${f(j.loss,3)}`} sub="target features are stop-gradient" color="var(--green)"/>
  <Card x={355} y={302} w={190} title="Predictor φ" sub="receives ∂L/∂φ" color="var(--purple)"/><Card x={120} y={302} w={190} title="Context encoder θ" sub="receives ∂L/∂θ" color="var(--blue)"/>
  <Wire d="M635 328 H455 M260 328 H220" id={id} dashed color="var(--green)"/>
  <Text x={500} y={388} size={16}>EMA is a parameter update: θ̄′ = mθ̄ + (1−m)θ</Text>
  <Text x={500} y={421}>θ̄ = [0.4,0.8], θ = [1.0,0.2], m = {f(momentum)} → θ̄′ = {vec(j.ema)}</Text>
  <Text x={500} y={460} size={16}>Why a matching loss alone is insufficient</Text>
  <Text x={500} y={492}>If both learned branches output [0,0] for every input: matching loss = 0 and feature variance = 0.</Text>
  <Text x={500} y={526} size={14}>Displayed target mean feature variance across positions = {f(j.variance,4)}; constant-output variance = 0.</Text>
  <Text x={500} y={558} size={13}>Variance is a diagnostic, not proof of useful representations or a guarantee against collapse.</Text>
  <Text x={500} y={590} size={12}>Mask patterns and target-space comparison above remain independent explanatory controls.</Text>
 </Svg></Figure></>;
}

export function WorldPlanningMechanism(){
 const id=useId().replace(/:/g,'')+'planning';
 const [goal,setGoal]=useState(34),[h,setH]=useState(1); const ranked=rankPlans(goal),winner=ranked[0].name;
 const px=t=>255+t*260,py=z=>290-z*3.1;
 return <><Figure number="18" title="Read every candidate score from one common feature axis" controls={<><Range label="Scalar feature goal" value={goal} onChange={setGoal} min={-40} max={40} step={1}/><Range label="Executed prefix of two proposed actions" value={h} onChange={setH} min={1} max={2} step={1}/></>} caption={`Invented scalar latent trajectories, not V-JEPA 2 predictions or robot coordinates. All candidates start at feature 0 and share the same feature/time axes. Score is absolute endpoint distance with λ = 0. Candidate ${winner} wins; only ${h} of its two proposed actions execute before a new observation. Action values and robot state transitions are abstracted, so this is a scoring illustration, not a controller.`}>
 <Svg id={id} title="Common-axis latent rollouts, endpoint distances and external feedback routing" description={`Goal ${goal}. ${planningCandidates.map(c=>`${c.name}: ${c.points.join(', ')}; distance ${Math.abs(c.points[2]-goal)}`).join('. ')}. Select ${winner}, execute prefix ${h}, acquire a new observation and replan.`} height={650}>
  <Card x={185} y={30} w={245} title="Encode observed context" sub="zₜ = 0 in this scalar example" color="var(--blue)"/>
  <Wire d="M308 56 H385" id={id}/><Card x={535} y={30} w={300} title="Roll out candidate A, B, C" sub="same initial feature; different proposed actions" color="var(--purple)"/>
  <Text x={520} y={118} size={18}>Scalar feature z versus predicted step k</Text>
  <Wire d="M255 153 V426 H805"/>
  {[-40,-20,0,20,40].map(z=><g key={z}><path d={`M255 ${py(z)} H790`} stroke="var(--border)"/><Text x={232} y={py(z)+5} size={13}>{z}</Text></g>)}
  {[0,1,2].map(t=><g key={t}><path d={`M${px(t)} 153 V426`} stroke="var(--border)"/><Text x={px(t)} y={449} size={13}>k = {t}</Text></g>)}
  <path d={`M255 ${py(goal)} H790`} stroke="var(--orange)" strokeWidth="2" strokeDasharray="6 4"/><Text x={890} y={py(goal)+5} size={14}>goal {goal}</Text>
  {planningCandidates.map((c,i)=><g key={c.name}><polyline points={c.points.map((z,t)=>`${px(t)},${py(z)}`).join(' ')} fill="none" stroke={colors[i]} strokeWidth={c.name===winner?4:2}/>{c.points.map((z,t)=><g key={t}><circle cx={px(t)} cy={py(z)} r={c.name===winner?6:4} fill={colors[i]}/>{t>0&&<Text x={px(t)+(t===2?26:0)} y={py(z)+(i===2?20:-12)} size={13} color={colors[i]}>{c.name}:{z}</Text>}</g>)}</g>)}
  <Text x={500} y={481} size={16}>A: [0, −18, −35] · |−35 − ({goal})| = {Math.abs(-35-goal)}</Text>
  <Text x={500} y={509} size={16}>B: [0, 10, 34] · |34 − ({goal})| = {Math.abs(34-goal)}</Text>
  <Text x={500} y={537} size={16}>C: [0, 6, 12] · |12 − ({goal})| = {Math.abs(12-goal)}</Text>
  <Card x={535} y={563} w={510} title={`Select ${winner}; execute ${h}/2 actions; observe again`} sub="the outer feedback route never crosses the candidate plot" color="var(--green)"/>
  <Wire d="M280 590 H22 V56 H57" id={id} color="var(--green)"/>
  <Text x={500} y={638} size={12}>Ties use alphabetical order only for this demonstration. Distance is not a probability of task success.</Text>
 </Svg></Figure><Disclosure title="Architecture comparison: action-free video prediction and action-conditioned planning"><WorldModelFigure/></Disclosure></>;
}
