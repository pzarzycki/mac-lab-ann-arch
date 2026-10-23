import React,{useId,useState} from 'react';
import {Figure,Box,ArrowDefs} from './Figures';
import {people,relationTrace,additiveRead,scalarBlock,scalingLoss,fusionCounts,similarityMatrix,contrastiveTrace,maskedTokenLoss,temporalTrace} from '../representation-mechanisms-math';
const fmt=(v,d=3)=>Number(v).toFixed(d),pct=v=>`${fmt(v*100,1)}%`,vec=v=>`[${v.map(x=>fmt(x,2)).join(', ')}]`;
function Segs({label,value,onChange,options}){return <div className="segments" role="group" aria-label={label}>{options.map(([v,l])=><button key={v} aria-pressed={v===value} className={v===value?'selected':''} onClick={()=>onChange(v)}>{l}</button>)}</div>}
function Range({label,value,onChange,min,max,step=1}){return <label className="range-control">{label} <input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))}/><output>{fmt(value,step<1?2:0)}</output></label>}
function T({x,y,children,left=false,size=16,color}){return <text x={x} y={y} className="svg-label" textAnchor={left?'start':'middle'} style={{fontSize:size,fill:color,textAnchor:left?'start':'middle'}}>{children}</text>}
function Wire({d,id,color='var(--fig-line)',dash}){return <path d={d} fill="none" stroke={color} strokeWidth="2" strokeDasharray={dash} markerEnd={`url(#${id})`}/>}
function Canvas({id,title,desc,height=560,children}){return <svg className="representation-mechanisms" viewBox={`0 0 1000 ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}><title id={`${id}-title`}>{title}</title><desc id={`${id}-desc`}>{desc}</desc><ArrowDefs id={id}/><style>{`.representation-mechanisms .svg-label{font-size:16px}.representation-mechanisms .svg-small{font-size:13px}`}</style>{children}</svg>}
function Strip({y,items}){return <g><path d={`M30 ${y-25}H970`} stroke="var(--fig-line)" opacity=".4"/>{items.map((item,i)=><T key={item} x={185+i*315} y={y+5} size={14}>{item}</T>)}</g>}
export function DistributedRepresentationsFigure({compact=false}){
 const id=useId().replace(/:/g,'')+'dist',[person,setPerson]=useState(1),[eta,setEta]=useState(1),[updated,setUpdated]=useState(false),trace=relationTrace(person,updated?eta:0),code=people[person].code;
 return <Figure title="A shared feature update reaches another fact" compact={compact} controls={<><Segs label="Person in has-child query" value={person} onChange={setPerson} options={people.slice(0,2).map((p,i)=>[i,p.name])}/><Segs label="Apply one training fact" value={updated} onChange={setUpdated} options={[[false,'Before update'],[true,'After Ada → Cora']]}/><Range label="Learning rate η" value={eta} onChange={setEta} min={0} max={2} step={.1}/></>} caption={<>Synthetic two-candidate has-child model; the codes are chosen for explanation. One Ada → Cora gradient step changes Ben’s distribution because Ada and Ben share coordinates. A separate one-hot output column for Ben receives no update. Continuous vectors alone guarantee neither useful features nor held-out accuracy.</>}>
 <Canvas id={id} title="One-hot identities versus shared codes" desc={`Query ${people[person].name} has-child. Cora probability ${pct(trace.probabilities[0])}; isolated-column comparison ${pct(trace.isolated[0])}.`} height={620}>
 <T x={30} y={34} left size={18}>ILLUSTRATIVE MODEL · relation fixed to has-child · candidates Cora / Dan</T>
 <T x={225} y={85}>Isolated identity eᵢ ∈ R⁴</T><T x={725} y={85}>Distributed code vᵢ ∈ R³</T>
 {people.map((p,i)=><g key={p.name} opacity={i===person?1:.65}><T x={65} y={135+i*48} left>{p.name}</T>{people.map((_,j)=><g key={j}><rect x={140+j*55} y={111+i*48} width="45" height="34" rx="4" fill={i===j?'var(--accent-soft)':'var(--surface)'} stroke="var(--fig-line)"/><T x={162+j*55} y={134+i*48}>{i===j?1:0}</T></g>)}<T x={520} y={135+i*48} left>{p.name}</T>{p.code.map((v,j)=><g key={j}><rect x={610+j*105} y={111+i*48} width="90" height="34" rx="4" fill={i===person?'var(--accent-soft)':'var(--surface)'} stroke="var(--fig-line)"/><T x={655+j*105} y={134+i*48}>{fmt(v,1)}</T></g>)}</g>)}
 <T x={655} y={322} size={13}>older</T><T x={760} y={322} size={13}>branch A</T><T x={865} y={322} size={13}>branch B</T>
 <Wire d="M740 338V365" id={id}/><Box x={740} y={372} w={430} h={60} label={`z = Wv · scores ${vec(trace.scores)}`} sub={`v = ${vec(code)} · W rows shown below`}/>
 <T x={245} y={385} size={16}>One Ada training fact, target Cora</T><T x={245} y={418} size={14}>ΔWₖ = η (targetₖ − pₖ) vAdaᵀ</T><T x={245} y={451} size={14}>W rows initially [0, .2, 0] / [0, 0, .2]</T>
 <T x={740} y={464} size={14}>W Cora = {vec(trace.weights[0])}</T><T x={740} y={493} size={14}>W Dan = {vec(trace.weights[1])}</T>
 <T x={245} y={510}>Isolated: p(Cora) = {pct(trace.isolated[0])}</T><T x={740} y={535}>Shared: p(Cora) = {pct(trace.probabilities[0])}</T>
 <Strip y={590} items={['1986 · relational codes','Backprop · learn shared parameters','2003 · shared word features + LM']}/>
 </Canvas></Figure>;
}
export function SequenceAttentionHistoryFigure({compact=false}){
 const id=useId().replace(/:/g,'')+'align',[query,setQuery]=useState(-1),r=additiveRead(query);
 return <Figure title="One memory, two decoder-dependent reads" compact={compact} controls={<Segs label="Previous decoder state" value={query} onChange={setQuery} options={[[-1,'Target step A · s = −1'],[1,'Target step B · s = +1']]}/>} caption={<>Synthetic scalar encoder values, width-two additive alignment. eⱼ = tanh(hⱼ − s + 1) − tanh(hⱼ − s − 1); α = softmax(e). The decoder state changes the read of the same memory. A toy update s′ = tanh(0.5s + c + 0.2) then predicts two arbitrary output symbols. This is a mechanism trace, not a translation experiment.</>}>
 <Canvas id={id} title="Additive attention includes the decoder query and value paths" desc={`Weights ${vec(r.weights)}, context ${fmt(r.context)}, next decoder state ${fmt(r.nextState)}.`} height={650}>
 <T x={35} y={35} left size={17}>FIXED SUMMARY: encoder sequence → one summary → recurrent decoder</T>
 <path d="M30 60H970" stroke="var(--fig-line)"/>
 <Box x={160} y={85} w={235} h={65} label={`Previous decoder s = ${query}`} sub="query · target index t"/>
 <T x={640} y={96}>Retained source states · source index j = 1, 2, 3</T>
 {r.states.map((h,j)=><g key={j}><Box x={420+j*225} y={120} w={185} h={45} label={`h${j+1} = ${h}`}/><Wire d={`M${420+j*225} 165V216`} id={id}/><T x={420+j*225} y={198} size={14}>query + source → score</T><Wire d={`M${513+j*225} 142H${525+j*225}V366H${513+j*225}`} id={id} color="var(--blue)"/><Box x={420+j*225} y={224} w={185} h={60} label={`e${j+1} = ${fmt(r.scores[j])}`} sub={`α${j+1} = ${pct(r.weights[j])}`}/><Wire d={`M${420+j*225} 284V336`} id={id}/><T x={420+j*225} y={315} size={14}>α{j+1} × h{j+1}</T><Box x={420+j*225} y={344} w={185} h={44} label={fmt(r.contributions[j])}/><Wire d={`M${420+j*225} 388V410H640V433`} id={id}/></g>)}
 <Wire d="M160 150V253H322" id={id}/><T x={165} y={208} size={14}>s enters every score</T><T x={165} y={235} size={13}>blue bypasses retain h values</T><T x={165} y={287} size={14}>common alignment width = 2</T><T x={165} y={318} size={14}>Ws = [−1, −1]ᵀ</T><T x={165} y={346} size={14}>Wh = [1, 1]ᵀ</T><T x={165} y={374} size={14}>b = [1, −1]ᵀ; v = [1, −1]ᵀ</T>
 <Box x={640} y={440} w={360} h={46} label={`Context c = Σ αⱼ hⱼ = ${fmt(r.context)}`}/><Wire d="M640 486V513" id={id}/><Box x={640} y={520} w={440} h={56} label={`Next decoder state s′ = ${fmt(r.nextState)}`} sub={`p(symbol A / B) = ${pct(r.output[0])} / ${pct(r.output[1])}`}/><Wire d="M42 118V548H411" id={id} dash="5 4"/>
 <Strip y={619} items={['2017 encoder · self-attention','2017 decoder · masked self + cross','GPT branch · causal decoder only']}/>
 </Canvas></Figure>;
}
export function ResnetLineageFigure({compact=false}){
 const id=useId().replace(/:/g,'')+'resnet',[j,setJ]=useState(.25),[relu,setRelu]=useState(false),rows=[['Plain',0],['Identity skip',1],['Projection skip',.5]];
 return <Figure title="The skip adds a derivative; it can still cancel" compact={compact} controls={<><Range label="Branch derivative JF" value={j} onChange={setJ} min={-2} max={2} step={.05}/><Segs label="Derivative scope" value={relu} onChange={setRelu} options={[[false,'Additive junction'],[true,'Include post-add ReLU']]}/><button className="lab-play" onClick={()=>setJ(-1)}>Set cancellation JF = −1</button></>} caption={<>Synthetic scalar blocks: x = 1, F(x) = JF · x and upstream g = 1. Plain, identity and projection skips use P = 0, 1 and 0.5. The plotted derivative is (P + JF)g, optionally multiplied by the post-add ReLU gate; its derivative at exactly zero is chosen as zero. A direct path helps optimization but cannot prevent cancellation or explosion.</>}>
 <Canvas id={id} title="Compare plain and residual forward and backward paths" desc={`Branch derivative ${j}; post-add ReLU ${relu?'included':'excluded'}. Identity gradient ${scalarBlock(j,1,1,1,relu).gradient}.`} height={600}>
 <T x={500} y={36} size={18}>x = 1 · upstream gradient g = 1 · forward and reverse-mode terms</T>
 {rows.map(([name,p],i)=>{const x=175+i*325,r=scalarBlock(j,p,1,1,relu);return <g key={name}><T x={x} y={90} size={19}>{name} · P = {p}</T><Box x={x} y={118} w={170} label="x = 1"/><Wire d={`M${x} 150V197`} id={id}/><Box x={x} y={205} w={170} h={50} label={`F(x) = ${fmt(j,2)}`} sub={`JF = ${fmt(j,2)}`}/>{p!==0&&<><Wire d={`M${x} 164H${x+118}V294H${x+43}`} id={id}/><T x={x+103} y={279} size={12}>P·x</T></>}<Wire d={`M${x} 255V279`} id={id}/><Box x={x} y={286} w={80} h={38} label="+"/><Wire d={`M${x} 324V348`} id={id}/><Box x={x} y={355} w={245} h={57} label={`sum = ${fmt(r.sum,2)} · y = ${fmt(r.output,2)}`} sub={relu?`ReLU gate = ${r.gate}`:'Derivative before activation'}/><T x={x} y={452}>skip term = {fmt(r.skipGradient,2)}</T><T x={x} y={484}>branch term = {fmt(r.branchGradient,2)}</T><T x={x} y={520} size={20}>gₓ = {fmt(r.gradient,2)}</T></g>})}
 <Strip y={574} items={['Plain depth · harder optimization','Residual F = 0 · identity mapping','Original ResNet · post-add ReLU']}/>
 </Canvas></Figure>;
}
export function GptScalingFigure({compact=false}){
 const id=useId().replace(/:/g,'')+'scale',[compute,setCompute]=useState(64),[fraction,setFraction]=useState(.5),N=compute**fraction,r=scalingLoss(N,compute),opt=scalingLoss(Math.sqrt(compute),compute),X=n=>100+800*Math.log(n)/Math.log(compute),Y=l=>445-(l-1)*180;
 const curves=Array.from({length:101},(_,i)=>scalingLoss(compute**(i/100),compute));
 const path=field=>curves.map((d,i)=>`${i?'L':'M'}${X(d.N)},${Y(field==='total'?d.total:1+d[field])}`).join(' ');
 return <Figure title="Fixed compute trades model capacity for training tokens" compact={compact} controls={<><Segs label="Normalized training compute ND" value={compute} onChange={setCompute} options={[[16,'ND = 16'],[64,'ND = 64'],[256,'ND = 256']]}/><Range label="Fraction of log budget allocated to N" value={fraction} onChange={setFraction} min={0} max={1} step={.01}/></>} caption={<>Illustrative normalized units, not Kaplan or Chinchilla fitted coefficients: L = 1 + N⁻⁰·⁵ + D⁻⁰·⁵, ND = C. More parameters at fixed C means fewer tokens. The minimum balances the two reducible loss terms. Vertical values represent a synthetic cross-entropy-like loss; real fits depend on model family, data and training conditions.</>}>
 <Canvas id={id} title="Schematic language loss on a fixed compute frontier" desc={`N ${fmt(r.N,2)}, D ${fmt(r.D,2)}, loss ${fmt(r.total)}; balanced optimum N and D ${fmt(opt.N,2)}.`} height={650}>
 <T x={100} y={30} left size={18}>L = 1 + N⁻⁰·⁵ + D⁻⁰·⁵ · D = C/N · C = {compute}</T>
 <path d="M100 70V445H925" fill="none" stroke="var(--fig-line)"/>
 {[1,1.5,2].map(l=><g key={l}><path d={`M100 ${Y(l)}H910`} stroke="var(--fig-line)" opacity=".2"/><T x={68} y={Y(l)+5} size={14}>{l}</T></g>)}
 <path d={path('total')} fill="none" stroke="var(--accent)" strokeWidth="3"/><path d={path('model')} fill="none" stroke="var(--blue)" strokeWidth="2" strokeDasharray="7 4"/><path d={path('data')} fill="none" stroke="var(--orange)" strokeWidth="2" strokeDasharray="2 4"/>
 <path d={`M${X(opt.N)} 70V445`} stroke="var(--fig-line)" strokeDasharray="5 5"/><circle cx={X(r.N)} cy={Y(r.total)} r="8" fill="var(--accent)" stroke="var(--ink)"/>
 <T x={500} y={475}>N · log axis (D decreases to the right)</T><T x={100} y={466} size={13}>1</T><T x={900} y={466} size={13}>{compute}</T>
 <T x={275} y={95} size={14}>solid: total loss</T><T x={275} y={122} size={14}>dashed: 1 + model term</T><T x={275} y={149} size={14}>dotted: 1 + data term</T>
 <Box x={500} y={510} w={850} h={55} label={`N = ${fmt(N,2)} · D = ${fmt(r.D,2)} · ND = ${fmt(N*r.D,0)} · L = ${fmt(r.total)}`} sub={`model term ${fmt(r.model)} + data term ${fmt(r.data)} + floor 1 · optimum N = D = ${fmt(opt.N,2)}`}/>
 <Strip y={615} items={['Kaplan 2020 · model-heavier allocation','Hoffmann 2022 · scale N and D together','These curves · synthetic coefficients']}/>
 </Canvas></Figure>;
}
function Tokens({x,y,count,prefix,color='var(--accent)',width=420}){const cell=Math.min(48,width/count);return <g>{Array.from({length:count},(_,i)=><g key={i}><rect x={x+i*cell} y={y} width={cell-5} height="32" rx="4" fill="var(--surface)" stroke={color}/><T x={x+i*cell+(cell-5)/2} y={y+21} size={13}>{prefix}{i+1}</T></g>)}</g>}
export function MultimodalFusionFigure({compact=false}){
 const id=useId().replace(/:/g,'')+'fusion',[mode,setMode]=useState('prefix'),[p,setP]=useState(8),[m,setM]=useState(2),[resample,setResample]=useState(false),n=4,r=fusionCounts(mode,p,m,n,resample),compressed=mode==='compressed'||(mode==='cross'&&resample);
 return <Figure title="Which tokens belong to the language sequence?" compact={compact} controls={<><Segs label="Fusion interface" value={mode} onChange={setMode} options={[["prefix","Projected prefix"],["compressed","Query → prefix"],["cross","External cross-attention"]]}/><Range label="Encoded visual tokens p" value={p} onChange={setP} min={4} max={12}/><Segs label="Query outputs m" value={m} onChange={setM} options={[[2,'m = 2'],[4,'m = 4']]}/>{mode==='cross'&&<label><input type="checkbox" checked={resample} onChange={e=>setResample(e.target.checked)}/> Resample external memory first</label>}</>} caption={<>Illustrative counts and widths: p visual encoder tokens, m query outputs, n = 4 language tokens. Prefix tokens join self-attention; external memory does not. Pair counts show dense score-matrix entries per head, before masking and excluding projections/MLPs. Compression can feed either interface, as resampling plus external cross-attention does in Flamingo.</>}>
 <Canvas id={id} title="Prefix membership, query compression and separate Q/K/V ports" desc={`${mode}, p ${p}, m ${m}. Language length ${r.length}, external memory ${r.memory}; self score entries ${r.selfPairs}, cross entries ${r.crossPairs}.`} height={650}>
 <T x={35} y={35} left size={18}>Visual encoder Z ∈ Rᵖˣᵈᵛ</T><Tokens x={40} y={65} count={p} prefix="v" width={520}/><T x={775} y={81}>Text H ∈ R⁴ˣᵈ</T><Tokens x={675} y={104} count={4} prefix="t" width={260} color="var(--blue)"/>
 <Wire d="M290 98V157" id={id}/>
 <Box x={290} y={165} w={470} h={65} label={compressed?`m = ${m} learned queries → read p = ${p} features`:`Project p = ${p} tokens to compatible width d`} sub={compressed?'Q from learned queries; K,V from encoded vision':'V = ZP; P ∈ Rᵈᵛˣᵈ'}/>
 {compressed&&<><Wire d="M290 230V255" id={id}/><Box x={290} y={263} w={420} h={48} label="Project query width dq → interface width d"/></>}
 {mode!=='cross'?<>
 <Wire d={`M290 ${compressed?311:230}V350H410V380`} id={id}/><Wire d="M800 136V348H840V380" id={id} color="var(--blue)"/>
 <T x={500} y={365}>ONE language sequence · [{r.visual} visual tokens | {n} text tokens]</T>
 <rect x="30" y="389" width="940" height="86" rx="8" fill="var(--fig-panel)" stroke="var(--fig-line)"/>
 <Tokens x={55} y={411} count={r.visual} prefix="v" width={590}/><Tokens x={710} y={411} count={n} prefix="t" width={240} color="var(--blue)"/>
 <T x={350} y={500}>visual prefix read rule: visual ↔ visual, no text reads</T><T x={350} y={529}>text reads all visual + text at positions ≤ its own</T><T x={785} y={505}>length = {r.length}</T><T x={785} y={535}>self Q,K,V: all {r.length} positions</T>
 </>:<>
 <Wire d={`M290 ${compressed?311:230}V345`} id={id}/><T x={290} y={335}>EXTERNAL memory · {r.memory} tokens</T><Tokens x={60} y={353} count={r.memory} prefix="v" width={470}/>
 <Wire d="M800 136V385" id={id} color="var(--blue)"/><T x={840} y={305} size={14}>language sequence only</T><T x={840} y={335} size={14}>n = 4 positions</T>
 <Box x={765} y={393} w={330} h={68} label="Cross-attention" sub={`Q: 4 × dk · K,V: ${r.memory} memory rows`}/>
 <Wire d="M300 385V429H592" id={id}/><T x={420} y={416} size={14}>K,V from memory</T><T x={835} y={378} size={14}>Q from language</T>
 <Wire d="M765 461V495" id={id} color="var(--blue)"/><Box x={765} y={503} w={370} h={44} label="4 updated language states → LM"/>
 <T x={290} y={485}>Memory never becomes a text position.</T><T x={290} y={516} size={14}>QKᵀ ∈ R⁴ˣʳ · softmax(QKᵀ/√dk)V</T>
 </>}
 <Box x={500} y={578} w={920} h={46} label={`Dense score entries: self ${r.length}² = ${r.selfPairs} · external read ${n}×${r.memory} = ${r.crossPairs} · ${r.resamplerPairs===0?'resampler inactive = 0':`resampler ${m}×${p} = ${r.resamplerPairs}`}`} />
 </Canvas></Figure>;
}
export function MultimodalObjectivesFigure({compact=false}){
 const id=useId().replace(/:/g,'')+'obj',[direction,setDirection]=useState('image'),[index,setIndex]=useState(0),[temp,setTemp]=useState(.3),[position,setPosition]=useState(5),r=contrastiveTrace(direction,index,temp),labels=['red square','blue circle','red object'],tokens=['[image]','User:','What','color?','Assistant:','red','[stop]'],loss=maskedTokenLoss([1,1,1,1,1,.8,.5],[0,0,0,0,0,1,1]);
 return <Figure title="Pair classification and answer-token prediction" compact={compact} controls={<><Segs label="Contrastive normalization direction" value={direction} onChange={setDirection} options={[["image","Image → text (row)"],["text","Text → image (column)"]]}/><Segs label="Batch query" value={index} onChange={setIndex} options={labels.map((l,i)=>[i,`${i+1} · ${l}`])}/><Range label="Contrastive temperature τ" value={temp} onChange={setTemp} min={.1} max={1} step={.05}/><Segs label="Next token target" value={position} onChange={setPosition} options={[[5,'Predict red'],[6,'Predict stop']]}/></>} caption={<>Synthetic batch: normalized image and text vectors [1,0], [0,1], [0.8,0.6]. The diagonal is the paired target, even when another caption is semantically plausible. Selected row/column probabilities are softmax(similarity/τ). Below, an illustrative answer mask scores red and stop; prompt positions still condition predictions. Token probabilities are supplied assumptions, not measured model outputs.</>}>
 <Canvas id={id} title="Explicit contrastive matrix and shifted supervised-token mask" desc={`Selected ${direction} ${index+1}, correct-pair probability ${pct(r.probabilities[index])}, cross entropy ${fmt(r.loss)}. Current token target ${tokens[position]}, answer-only mean loss ${fmt(loss.mean)}.`} height={700}>
 <T x={35} y={30} left size={18}>sᵢⱼ = normalized vᵢ · tⱼ · ✓ denotes paired target</T>
 {labels.map((l,j)=><T key={l} x={250+j*143} y={67} size={14}>T{j+1}: {l}</T>)}
 {similarityMatrix.map((row,i)=><g key={i}><T x={30} y={117+i*76} left size={14}>I{i+1}: {labels[i]}</T>{row.map((s,j)=>{const selected=direction==='image'?i===index:j===index;return <g key={j}><rect x={184+j*143} y={83+i*76} width="131" height="65" rx="5" fill={selected?'var(--accent-soft)':'var(--surface)'} stroke={selected?'var(--accent)':'var(--fig-line)'} strokeWidth={selected?2:1}/><T x={250+j*143} y={109+i*76}>{fmt(s,2)} {i===j?'✓':''}</T><T x={250+j*143} y={134+i*76} size={13}>logit {fmt(s/temp,2)}</T></g>})}</g>)}
 <T x={800} y={70} size={17}>Selected {direction==='image'?'row':'column'} probabilities</T>
 {r.probabilities.map((p,j)=><g key={j}><T x={670} y={119+j*65} left size={14}>{j+1}{j===index?' ✓':''}</T><rect x="715" y={100+j*65} width={185*p} height="23" rx="3" fill="var(--accent)"/><T x={935} y={118+j*65} size={14}>{pct(p)}</T></g>)}
 <T x={790} y={302}>−log p(correct) = {fmt(r.loss)}</T><T x={790} y={331} size={14}>directional batch mean = {fmt(r.batchLoss)}</T>
 <T x={350} y={348} size={14}>A plausible red caption can be a false negative.</T>
 <path d="M30 370H970" stroke="var(--fig-line)"/><T x={35} y={403} left size={18}>Visual instruction tuning · conditional token likelihood + chosen scored spans</T>
 {tokens.map((token,i)=><g key={i}><rect x={34+i*137} y="433" width="125" height="52" rx="5" fill={i>=5?'var(--accent-soft)':'var(--surface)'} stroke={i===position?'var(--accent)':'var(--fig-line)'} strokeWidth={i===position?3:1}/><T x={96+i*137} y={464} size={14}>{token}</T><T x={96+i*137} y={510} size={14}>mask = {i>=5?1:0}</T><T x={96+i*137} y={538} size={13}>{i<position?'context':i===position?'TARGET':'future'}</T></g>)}
 <Wire d={`M${96+(position-1)*137} 554V578H${96+position*137}V554`} id={id}/>
 <T x={500} y={615}>Predict {tokens[position]} from positions before it · assumed p = {position===5?'.8':'.5'}</T>
 <T x={500} y={653}>Answer loss = −log(.8) − log(.5) = {fmt(loss.sum)} · mean over 2 scored tokens = {fmt(loss.mean)}</T>
 </Canvas></Figure>;
}
export function TemporalModalitiesFigure({compact=false}){
 const id=useId().replace(/:/g,'')+'time',[fps,setFps]=useState(2),[phase,setPhase]=useState(0),[offset,setOffset]=useState(0),r=temporalTrace(fps,phase,offset),X=t=>160+180*t;
 return <Figure title="Sampling loses events; clock offsets misalign evidence" compact={compact} controls={<><Segs label="Video frames per second" value={fps} onChange={setFps} options={[[1,'1 fps'],[2,'2 fps'],[4,'4 fps']]}/><Range label="Sample phase (fraction of frame interval)" value={phase} onChange={setPhase} min={0} max={.9} step={.1}/><Range label="Audio clock offset Δ seconds" value={offset} onChange={setOffset} min={-.5} max={.5} step={.05}/></>} caption={<>Synthetic 4 s clip, instantaneous video samples and a 0.20 s event at [1.15,1.35). Audio windows here are 0.20 s wide every 0.50 s, deliberately coarser than Whisper’s frontend. Their reported time is physical time + Δ. A toy read allows audio centers within 0.26 s of video query 1.25 s. Correcting the clock can align retained evidence; it cannot recover an unsampled visual event.</>}>
 <Canvas id={id} title="Shared timeline with sampled event and audio-clock shift" desc={`Video samples ${r.frames.length}, event hits ${r.hits.length}, token count ${r.tokens}. Event audio window ${r.eventAudioAllowed?'is':'is not'} allowed by the timestamp read rule.`} height={650}>
 <T x={35} y={34} left size={18}>Shared timestamp axis · event physical time 1.15–1.35 s</T>
 {Array.from({length:9},(_,i)=>i*.5).map(t=><g key={t}><path d={`M${X(t)} 65V440`} stroke="var(--fig-line)" opacity=".24"/><T x={X(t)} y={62} size={13}>{t}s</T></g>)}
 <T x={25} y={120} left size={15}>World event</T><rect x={X(1.15)} y="95" width={.2*180} height="42" fill="var(--accent-soft)" stroke="var(--accent)"/><T x={600} y={123} size={14}>short state change (synthetic)</T>
 <T x={25} y={215} left size={15}>Video frames</T><path d={`M${X(0)} 210H${X(4)}`} stroke="var(--fig-line)"/>
 {r.frames.map((t,i)=>{const hit=t>=1.15&&t<1.35;return <g key={i}><rect x={X(t)-9} y="185" width="18" height="48" rx="3" fill={hit?'var(--accent)':'var(--surface)'} stroke="var(--fig-line)"/><T x={X(t)} y={260+(i%2)*21} size={12}>{fmt(t,2)}</T>{hit&&<T x={X(t)} y={178} size={12}>HIT</T>}</g>})}
 <T x={25} y={340} left size={15}>Audio windows</T><path d={`M${X(0)} 336H${X(4)}`} stroke="var(--fig-line)"/>
 {r.audio.map((w,i)=><g key={i}><rect x={X(w.start)} y="311" width="36" height="50" rx="4" fill={w.isEvent?'var(--accent-soft)':'var(--surface)'} stroke={w.allowed?'var(--accent)':'var(--fig-line)'} strokeWidth={w.allowed?3:1}/><T x={X(w.reported)} y="342" size={12}>{w.isEvent?'event':i+1}</T><T x={X(w.reported)} y={390+(i%2)*22} size={12}>{fmt(w.reported,2)}</T></g>)}
 <path d={`M${X(1.25)} 90V438`} stroke="var(--accent)" strokeDasharray="4 5"/><T x={X(1.25)} y={459} size={14}>video query 1.25 s</T><T x={750} y={455} size={14}>thick audio borders = allowed reads</T>
 <Box x={500} y={488} w={920} h={55} label={`Video event ${r.hits.length?'observed':'MISSED'} · ${r.frames.length} frames × 196 patches = ${r.tokens.toLocaleString('en-US')} tokens`} sub={`audio event center: physical 1.25 s → reported ${fmt(1.25+offset,2)} s · timestamp mask ${r.eventAudioAllowed?'admits':'excludes'} it`}/>
 <T x={500} y={583}>Try 2 fps, phase 0 → missed; phase 0.5 → sample at 1.25 s.</T><T x={500} y={615} size={14}>Shift Δ to +0.5 s: observations stay present while correspondence changes.</T>
 </Canvas></Figure>;
}
