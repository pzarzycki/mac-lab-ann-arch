import React,{useId,useState} from 'react';
import {Figure,Box,ArrowDefs,ResidualFigure} from './Figures';
import {allowedAttention,edgeCount} from '../math';
import {add,dot,normStats,rotate2,swigluToy,attentionTrace,onlineAttention,quantizeAffine,speculativeMass,speculativeCycle,kvMemory,headGroup} from '../transformer-mechanisms-math';

const f=(x,n=3)=>Number.isFinite(x)?(Math.abs(x)<.5*10**-n?0:x).toFixed(n):x===-Infinity?'−∞':'∞';
const vector=(v,n=2)=>`[${v.map(x=>f(x,n)).join(', ')}]`;
const C=['var(--accent)','var(--orange)','var(--blue)','var(--purple)'];
function Segments({label,value,onChange,options}){return <div className="segments" role="group" aria-label={label}>{options.map(([v,t])=><button key={v} aria-pressed={v===value} className={v===value?'selected':''} onClick={()=>onChange(v)}>{t}</button>)}</div>;}
function Range({label,value,onChange,min,max,step=1,display}){return <label className="range-control">{label} <b>{display??value}</b><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(+e.target.value)}/></label>;}
function Text({x,y,children,small=false,anchor='middle',...rest}){return <text x={x} y={y} textAnchor={anchor} className={small?'svg-small':'svg-label'} {...rest}>{children}</text>;}
function Wire({d,id,color='var(--fig-line)',...rest}){return <path d={d} fill="none" stroke={color} strokeWidth="1.8" markerEnd={id?`url(#${id})`:undefined} {...rest}/>;}
function Panel({x,y,w,h,title}){return <g><rect x={x} y={y} width={w} height={h} rx="9" fill="var(--fig-panel)" stroke="var(--border)"/><text x={x+18} y={y+27} className="svg-overline" textAnchor="start">{title}</text></g>;}
function Svg({title,description,height=480,children}){const id=useId().replace(/:/g,'');return <svg viewBox={`0 0 1000 ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}><title id={`${id}-title`}>{title}</title><desc id={`${id}-desc`}>{description}</desc>{children}</svg>;}

export function ResidualMechanismFigure(){
 const [token,setToken]=useState(2),[write,setWrite]=useState(.5),hs=[[1,0,-1],[0,1,.5],[1,-1,.5]],h=hs[token],a=[write,.25,-.5],m=[-.25,.5,1],mid=add(h,a),out=add(mid,m);
 return <><ResidualFigure/><Figure title="Trace the additions in one token" controls={<><Segments label="Vector inspector token" value={token} onChange={setToken} options={['The','model','learns'].map((t,i)=>[i,`${i+1} · ${t}`])}/><Range label="Attention write · coordinate 1" min={-1.5} max={1.5} step={.25} value={write} onChange={setWrite}/></>} caption="Synthetic width-3 writes, held fixed for arithmetic inspection. This panel isolates pre-norm residual additions; it does not simulate learned attention or MLP functions. The topology controls above are independent.">
 <Svg height={450} title="Coordinate-wise residual accumulation" description={`Token ${token+1}: incoming ${vector(h)}, attention write ${vector(a)}, intermediate ${vector(mid)}, MLP write ${vector(m)}, final ${vector(out)}. Only coordinate one changes with the write slider.`}>
 <Text x={500} y={35}>One token · three coordinates · writes add without replacing the whole state</Text>
 {['incoming h','attention Δa','h + Δa','MLP Δm','h + Δa + Δm'].map((t,i)=><Text key={t} x={110+i*195} y={85}>{t}</Text>)}
 {[h,a,mid,m,out].map((v,i)=><g key={i}>{v.map((v,j)=><g key={j}><rect x={43+i*195} y={112+j*60} width="134" height="44" rx="5" fill={i===2||i===4?'var(--accent-soft)':'var(--fig-panel)'} stroke={j===0?'var(--accent)':'var(--border)'}/><Text x={110+i*195} y={140+j*60}>{f(v,2)}</Text></g>)}{i<4&&<Text x={207+i*195} y={202}>{i%2===0?'+':'='}</Text>}</g>)}
 <Text x={500} y={335}>Coordinate 1: {f(h[0],2)} + {f(a[0],2)} + ({f(m[0],2)}) = {f(out[0],2)}</Text>
 <Text x={500} y={375} small>Coordinates 2 and 3 stay fixed as the first write coordinate changes.</Text>
 <Text x={500} y={410} small>Pre-LN: normalization is inside each write branch. Post-LN: it also transforms the sum on the next-block path.</Text>
 </Svg></Figure></>;
}

function MaskGraph(){
 const [mode,setMode]=useState('causal'),[window,setWindow]=useState(4),[row,setRow]=useState(8),n=12;
 return <Figure title="Inspect the permitted-read graph" controls={<><Segments label="Graph mask" value={mode} onChange={setMode} options={[[ 'causal','Causal'],['window','Sliding window'],['full','Full']]}/><Range label="Graph destination" min={0} max={11} value={row} display={row+1} onChange={setRow}/>{mode==='window'&&<Range label="Graph window width" min={1} max={12} value={window} onChange={setWindow}/>}</>} caption="Connectivity only. A dot marks every allowed cell, including unselected rows; × marks a blocked cell. These symbols distinguish the graph in both themes. Numeric examples use separate four-source toy inputs.">
 <Svg height={490} title={`${mode} connectivity with ${edgeCount(n,mode,window)} edges`} description={`Destination ${row+1} can read ${Array.from({length:n},(_,j)=>allowedAttention(row,j,mode,window)?j+1:null).filter(Boolean).join(', ')}. Allowed cells contain a dot; blocked cells contain an x.`}>
 <text x="290" y="29" className="svg-overline">SOURCE j →</text><Text x={25} y={255} small transform="rotate(-90 25 255)">DESTINATION i →</Text>
 {Array.from({length:n},(_,i)=><g key={i}><Text x={88+i*31} y={55} small>{i+1}</Text><Text x={50} y={82+i*31} small>{i+1}</Text>{Array.from({length:n},(_,j)=>{const allowed=allowedAttention(i,j,mode,window),x=75+j*31,y=65+i*31;return <g key={j}><rect x={x} y={y} width="27" height="27" rx="3" fill={allowed?'var(--accent-soft)':'var(--fig-bg)'} stroke={i===row?'var(--accent)':'var(--border)'} strokeWidth={i===row?2:1}/>{allowed?<circle cx={x+13.5} cy={y+13.5} r={i===row?4.5:3.2} fill="var(--accent)"/>:<path d={`M${x+10} ${y+10}l7 7m0 -7l-7 7`} stroke="var(--muted)" strokeWidth="1"/>}</g>;})}</g>)}
 <Text x={535} y={100} anchor="start">PERMITTED EDGES</Text><text x="535" y="159" className="svg-stat" textAnchor="start">{edgeCount(n,mode,window)} / {n*n}</text><Text x={535} y={217} anchor="start">Destination {row+1} reads:</Text>
 <Text x={535} y={253} anchor="start">{Array.from({length:n},(_,j)=>allowedAttention(row,j,mode,window)?j+1:null).filter(Boolean).join(', ')}</Text>
 <Text x={535} y={319} small anchor="start">Causal: n(n + 1)/2 = 78</Text><Text x={535} y={351} small anchor="start">Window: Σ min(i + 1, w)</Text><Text x={535} y={391} small anchor="start">A smaller graph does not guarantee a faster kernel.</Text>
 <Text x={500} y={466} small>● permitted read     × blocked read     outlined row = selected destination</Text>
 </Svg></Figure>;
}
function AttentionRow(){
 const [row,setRow]=useState(2),[offset,setOffset]=useState(0),[value,setValue]=useState(2),[mode,setMode]=useState('causal');
 const queryBase=[[0,1],[1,0],[Math.SQRT2,0],[1,-1]],query=[queryBase[row][0]+offset,queryBase[row][1]],keys=[[1,0],[0,1],[1,1],[-1,0]],values=[[1,0],[0,2],[value,1],[-1,1]],allowed=keys.map((_,j)=>mode==='full'||j<=row),s=attentionTrace(query,keys,values,allowed);
 return <Figure title="One head: matching becomes a weighted value sum" controls={<><Segments label="Numeric attention mask" value={mode} onChange={setMode} options={[[ 'causal','Causal'],['full','Full']]}/><Range label="Numeric destination" min={0} max={3} value={row} display={row+1} onChange={setRow}/><Range label="Query coordinate 1 offset" min={-2} max={2} step={.25} value={offset} onChange={setOffset}/><Range label="Source 3 value coordinate 1" min={-2} max={4} step={.25} value={value} onChange={setValue}/></>} caption="Computed toy inputs after Q/K/V projection, dₖ = dᵥ = 2. Editing the query changes matching and routing; editing a value changes the mixed content while coefficients stay fixed. No trained-model attention is represented.">
 <Svg height={540} title="Scaled logits, masked softmax and value contributions" description={`Query ${vector(query)}. Weights ${vector(s.weights,3)}. Weighted output ${vector(s.output,3)}.`}>
 <Box x={500} y={22} w={420} h={48} label={`Destination ${row+1} query q = ${vector(query)}`} sub="read request · score = q · k / √2" color="var(--accent)"/>
 {['source · key','raw score','masked score','softmax Aᵢⱼ','value vⱼ','Aᵢⱼ × vⱼ'].map((t,i)=><Text key={t} x={[112,300,443,580,748,905][i]} y={117}>{t}</Text>)}
 {keys.map((k,j)=><g key={j}><rect x="24" y={140+j*70} width="952" height="54" rx="6" fill={allowed[j]?'var(--accent-soft)':'var(--fig-panel)'} stroke="var(--border)" strokeDasharray={allowed[j]?undefined:'4 4'}/><Text x={112} y={173+j*70}>{j+1} · {vector(k,0)}</Text><Text x={300} y={173+j*70}>{f(s.logits[j])}</Text><Text x={443} y={173+j*70}>{allowed[j]?f(s.masked[j]):'−∞ · blocked'}</Text><Text x={580} y={173+j*70}>{f(s.weights[j])}</Text><Text x={748} y={173+j*70}>{vector(values[j])}</Text><Text x={905} y={173+j*70}>{vector(s.contributions[j])}</Text></g>)}
 <Text x={500} y={459}>Row sum = {f(s.weights.reduce((a,b)=>a+b,0))} · Oᵢ = Σ Aᵢⱼvⱼ = {vector(s.output,3)}</Text>
 <Text x={500} y={507} small>All four raw scores are visible. Masked positions contribute zero even when their raw score is large.</Text>
 </Svg></Figure>;
}
export function AttentionMechanismFigure(){return <><AttentionRow/><MaskGraph/></>;}

function NormTransform(){
 const [shift,setShift]=useState(0),[last,setLast]=useState(2),x=[-1+shift,1+shift,last+shift],s=normStats(x);
 return <Figure title="Role 1 · normalize one token’s features" controls={<><Range label="Shared feature offset" min={-2} max={3} step={.5} value={shift} onChange={setShift}/><Range label="Third feature before offset" min={-2} max={4} step={.5} value={last} onChange={setLast}/></>} caption="Synthetic width-3 vector. γ = 1, β = 0, ε = 10⁻⁵. LN centers before scaling; RMSNorm retains the mean. A common feature offset leaves LN unchanged here, while RMSNorm usually changes.">
 <Svg height={430} title="Computed LayerNorm versus RMSNorm" description={`Input ${vector(x)}, mean ${f(s.mean)}, variance ${f(s.variance)}, RMS denominator ${f(s.rms)}. LN ${vector(s.ln)}, RMSNorm ${vector(s.rmsnorm)}.`}>
 <Text x={500} y={38}>Statistics use the feature axis of this token, never other positions.</Text>
 {[['input x',x],['LayerNorm',s.ln],['RMSNorm',s.rmsnorm]].map(([name,values],row)=><g key={name}><Text x={125} y={107+row*96}>{name}</Text>{values.map((v,j)=><g key={j}><rect x={240+j*245} y={73+row*96} width="208" height="64" rx="6" fill="var(--fig-panel)" stroke="var(--border)"/><line x1={344+j*245} x2={344+j*245} y1={78+row*96} y2={131+row*96} stroke="var(--border)"/><rect x={344+j*245+Math.min(0,Math.max(-94,v*18))} y={119+row*96} width={Math.min(94,Math.max(1,Math.abs(v)*18))} height="8" fill={C[row]}/><Text x={344+j*245} y={103+row*96}>{f(v)}</Text></g>)}</g>)}
 <Text x={500} y={381}>μ = {f(s.mean)} · variance = {f(s.variance)} · √(mean x² + ε) = {f(s.rms)}</Text>
 <Text x={500} y={416} small>Learned γ (and LN β) can change the final scale; neither output is promised unit RMS after affine parameters.</Text>
 </Svg></Figure>;
}
function RotaryTransform(){
 const [i,setI]=useState(0),[j,setJ]=useState(1),[shift,setShift]=useState(0),omega=Math.PI/6,q=[1,0],k=[0,1],qr=rotate2(q,(i+shift)*omega),kr=rotate2(k,(j+shift)*omega),relative=rotate2(k,(j-i)*omega),left=dot(qr,kr),right=dot(q,relative),id=useId().replace(/:/g,'')+'rope';
 const arrow=(v,color,dash)=> <Wire d={`M255 247L${255+v[0]*133} ${247-v[1]*133}`} id={id} color={color} strokeWidth="3" strokeDasharray={dash}/>;
 return <Figure title="Role 2 · rotate a query/key coordinate pair" controls={<><Range label="Query position i" min={0} max={6} value={i} onChange={setI}/><Range label="Key position j" min={0} max={6} value={j} onChange={setJ}/><Range label="Common position shift" min={0} max={4} value={shift} onChange={setShift}/></>} caption="One 2D pair of a synthetic attention head, ω = π/6 radians per position. Solid arrows show rotated q and k; dashed arrows show the originals. A common shift rotates both and preserves their dot product. A full RoPE head repeats this construction over coordinate pairs with pair-specific frequencies.">
 <Svg height={480} title="Rotary geometry and relative-position identity" description={`Positions ${i+shift} and ${j+shift}; difference ${j-i}. Rotated query ${vector(qr)}, rotated key ${vector(kr)}. Both dot-product constructions equal ${f(left,4)}.`}>
 <ArrowDefs id={id}/><Panel x={25} y={25} w={455} h={410} title="ONE COORDINATE PLANE"/>
 <line x1="75" x2="435" y1="247" y2="247" stroke="var(--border)"/><line x1="255" x2="255" y1="72" y2="415" stroke="var(--border)"/><circle cx="255" cy="247" r="133" fill="none" stroke="var(--border)" strokeDasharray="3 4"/>{arrow(q,'var(--accent)','5 5')}{arrow(k,'var(--orange)','5 5')}{arrow(qr,'var(--accent)')}{arrow(kr,'var(--orange)')}
 <Text x={80} y={390} small anchor="start">q′ · iris</Text><Text x={80} y={415} small anchor="start">k′ · orange</Text><Text x={435} y={270} small>x₁</Text><Text x={275} y={80} small>x₂</Text>
 <Text x={525} y={70} anchor="start">q = [1, 0] · k = [0, 1]</Text><Text x={525} y={119} anchor="start">q′ = {vector(qr,3)}</Text><Text x={525} y={161} anchor="start">k′ = {vector(kr,3)}</Text><Text x={525} y={219} anchor="start">(Rᵢq)ᵀ(Rⱼk) = {f(left,4)}</Text><Text x={525} y={269} anchor="start">qᵀRⱼ₋ᵢk = {f(right,4)}</Text><Text x={525} y={325} anchor="start">j − i = {j-i} · angle difference = {j-i}π/6</Text><Text x={525} y={382} small anchor="start">RᵢᵀRⱼ = Rⱼ₋ᵢ · rotations compose by angle.</Text><Text x={500} y={466} small>The equality is algebraic; it does not establish generalization to unseen sequence lengths.</Text>
 </Svg></Figure>;
}
function GatedTransform(){
 const [v,setV]=useState(1),x=[v,1],s=swigluToy(x),id=useId().replace(/:/g,'')+'gate';
 return <Figure title="Role 3 · two parallel projections meet at a gate" controls={<Range label="MLP input coordinate 1" min={-2} max={2} step={.25} value={v} onChange={setV}/>} caption="Synthetic bias-free SwiGLU, d = 2 and m = 3. Gate and up projections read the same input in parallel. SiLU is not a probability: its values may be negative or exceed one. This position-wise network does not mix tokens.">
 <Svg height={490} title="Computed SwiGLU branches and down projection" description={`Input ${vector(x)}. Gate preactivation ${vector(s.gate)}, SiLU ${vector(s.activated)}, up ${vector(s.up)}, product ${vector(s.product)}, output ${vector(s.output)}.`}>
 <ArrowDefs id={id}/><Box x={100} y={190} w={154} h={56} label={`x = ${vector(x)}`} sub="width d = 2"/><Wire d="M177 218H205V115H240" id={id}/><Wire d="M177 218H205V312H240" id={id}/>
 <Box x={360} y={73} w={240} h={68} label={`Wg x = ${vector(s.gate)}`} sub={`SiLU → ${vector(s.activated)}`} color="var(--accent)"/><Box x={360} y={278} w={240} h={68} label={`Wu x = ${vector(s.up)}`} sub="up branch · width m = 3" color="var(--orange)"/>
 <Wire d="M480 107H530V199H550" id={id}/><Wire d="M480 312H530V235H550" id={id}/><Box x={650} y={185} w={200} h={66} label="Elementwise ×" sub={vector(s.product)} color="var(--purple)"/><Wire d="M750 218H790" id={id}/><Box x={885} y={185} w={190} h={66} label={`Wo → ${vector(s.output)}`} sub="width d = 2" color="var(--blue)"/>
 <Text x={500} y={394} small>Wg = [[1,0],[0,1],[1,−1]] · Wu = [[1,1],[1,−1],[0,1]] · Wo = [[1,0,1],[0,1,−1]]</Text>
 <Text x={500} y={435}>Gate coordinate 1: SiLU({f(s.gate[0],2)}) × {f(s.up[0],2)} = {f(s.product[0])}</Text>
 <Text x={500} y={475} small>Three matrices: 3dm weights. An ordinary two-matrix FFN of the same intermediate width has 2dm.</Text>
 </Svg></Figure>;
}
export function NormalizationMechanismFigure(){return <><NormTransform/><RotaryTransform/><GatedTransform/></>;}

export function TrainingDecodingMechanismFigure(){
 const [position,setPosition]=useState(2),[phase,setPhase]=useState(1),inputs=['BOS','The','model','learns'],targets=['The','model','learns','EOS'],prob=[.7,.5,.6,.8],loss=-Math.log(prob[position]),stages=['Before prefill','Prefill complete','First token sampled','Consume sample / append'],cached=phase===0?0:phase===3?4:3,id=useId().replace(/:/g,'')+'train';
 return <Figure title="Known targets in parallel; generated inputs in sequence" controls={<><Range label="Training output position" min={0} max={3} value={position} display={position+1} onChange={setPosition}/><Range label="Inference phase" min={0} max={3} value={phase} display={stages[phase]} onChange={setPhase}/></>} caption="Illustrative tokens and probabilities, B = 1, T = 4, V = 5. Training probabilities are stipulated toy outputs, not measured predictions. The inference trace uses prompt [BOS, The, model] and a predetermined sample ‘learns’. Weights update during training and remain fixed during this inference trace.">
 <Svg height={640} title="Teacher-forcing alignment and prefill/decode timeline" description={`Training position ${position+1} reads positions 1 to ${position+1}, predicts ${targets[position]} with probability ${prob[position]}, loss ${f(loss)} nats. Inference phase ${stages[phase]}, cache contains ${cached} positions.`}>
 <ArrowDefs id={id}/><Panel x={18} y={18} w={470} h={595} title="TRAINING · SHIFT TARGETS ONE PLACE"/><Panel x={510} y={18} w={472} h={595} title="INFERENCE · FIXED PARAMETERS"/>
 <Text x={252} y={85} small>inputs [1,4] → logits [1,4,5] → targets [1,4]</Text>
 {inputs.map((t,i)=><g key={t}><Box x={84+i*111} y={114} w={92} h={40} label={t} color={i<=position?'var(--accent)':'var(--border)'}/><Wire d={`M${84+i*111} 155V192`} id={id}/><Box x={84+i*111} y={195} w={92} h={48} label={`p(${targets[i]})`} sub={f(prob[i],2)} color={i===position?'var(--orange)':'var(--border)'}/><Wire d={`M${84+i*111} 244V273`} id={id}/><Box x={84+i*111} y={276} w={92} h={36} label={targets[i]} color={i===position?'var(--orange)':'var(--border)'}/></g>)}
 <Text x={252} y={352}>Output {position+1} may read input positions 1…{position+1}</Text>
 {inputs.map((_,i)=><g key={i}><Text x={151+i*49} y={380} small>{i+1}</Text><Text x={100} y={410+i*40} small>{i+1}</Text>{inputs.map((_,j)=><g key={j}><rect x={132+j*49} y={389+i*40} width="38" height="31" rx="4" fill={i===position&&j<=i?'var(--accent-soft)':'var(--fig-bg)'} stroke={i===position?'var(--accent)':'var(--border)'}/><Text x={151+j*49} y={410+i*40} small>{j<=i?'●':'×'}</Text></g>)}</g>)}
 <Text x={252} y={579}>−ln {f(prob[position],2)} = {f(loss)} nats</Text>
 <Text x={746} y={87}>{stages[phase]}</Text>
 {inputs.map((t,i)=><Box key={t} x={571+i*116} y={118} w={98} h={45} label={i===3&&phase<2?'?':t} sub={i<3?'prompt':phase===2?'sampled':'decode input'} active={i<3||phase>=2} color={i===3?'var(--orange)':'var(--accent)'}/>)}
 <Text x={746} y={207}>Per-layer K and V cache · {cached} positions</Text>
 {inputs.map((t,i)=><g key={t}><rect x={532+i*116} y="229" width="98" height="68" rx="6" fill={i<cached?'var(--accent-soft)':'var(--fig-bg)'} stroke={i<cached?'var(--accent)':'var(--border)'} strokeDasharray={i<cached?undefined:'4 4'}/><Text x={581+i*116} y={255}>{i<cached?`K${i+1}, V${i+1}`:'empty'}</Text><Text x={581+i*116} y={280} small>{i<cached?t:'not consumed'}</Text></g>)}
 <Wire d="M746 310V343" id={id}/><Box x={746} y={348} w={410} h={63} label={phase===0?'No logits yet':phase===3?'New-position logits predict the next token':'Final prompt-position logits predict learns'} sub={phase===0?'Call decoder on the three prompt inputs':phase===3?'toy p(EOS) = 0.70 · sample for following call':'toy p(learns) = 0.60; EOS = 0.10; rest = 0.30'} color="var(--accent)"/>
 <Text x={746} y={459}>{phase<2?'The first generated token is not cached yet.':phase===2?'Sample emitted; cache still has three positions.':'One new K/V pair appended in every layer.'}</Text>
 <Text x={746} y={501} small>{phase===3?'New query reads cached positions 1–4.':'Sampling alone does not append K/V.'}</Text><Text x={746} y={549} small>Each decode call consumes the previous sample.</Text><Text x={746} y={579} small>It returns logits for the following sample.</Text>
 <Text x={500} y={635} small>● allowed causal read · × blocked read · training targets are known; future inference inputs are not.</Text>
 </Svg></Figure>;
}

export function KVMechanismFigure(){
 const [v,setV]=useState({batch:1,layers:32,context:8192,kvHeads:8,headDim:128,bytes:2}),[append,setAppend]=useState(false),gib=kvMemory(v)/2**30,qPer=32/v.kvHeads,id=useId().replace(/:/g,'')+'kv';
 const change=(k,x)=>setV(old=>({...old,[k]:x}));
 return <><div className="lab-panel"><div className="lab-heading"><span className="eyebrow">INTERACTIVE CALCULATION</span><h3>The cost of remembering</h3><p>Conventional full-context KV cache · 32 query heads · equal K/V width</p></div><div className="kv-layout"><div className="lab-inputs">{[['batch','Batch size',1,16,1],['layers','Layers',4,96,4],['context','Context tokens',512,65536,512]].map(([key,label,min,max,step])=><label key={key}>{label}<output>{v[key].toLocaleString()}</output><input aria-label={label} type="range" min={min} max={max} step={step} value={v[key]} onChange={e=>change(key,+e.target.value)}/></label>)}<div className="lab-selects">{[['kvHeads','KV heads',[1,4,8,16,32]],['headDim','Head width',[64,128,256]],['bytes','Bytes / scalar',[.5,1,2,4]]].map(([key,label,options])=><label key={key}>{label}<select aria-label={label} value={v[key]} onChange={e=>change(key,+e.target.value)}>{options.map(n=><option key={n} value={n}>{n}</option>)}</select></label>)}</div></div><div className="kv-result" aria-live="polite"><span className="eyebrow">ESTIMATED CACHE</span><div className="big-number">{f(gib,gib<1?3:2)}<span>GiB</span></div><p>2 × B × L × T × H<sub>KV</sub> × d<sub>h</sub> × bytes</p><div className="comparison-bars">{[['MHA',32],['Selected',v.kvHeads],['MQA',1]].map(([label,h])=><div key={label}><span>{label}</span><div><i style={{width:`${Math.max(2,100*h/32)}%`}}/></div><b>{f(kvMemory({...v,kvHeads:h})/2**30,2)}</b></div>)}</div><small>GiB = 2³⁰ bytes. Excludes weights, padding, allocation overhead, temporary buffers and quantization metadata. MLA requires a different formula.</small></div></div></div>
 <Figure title="What the KV-head selector shares" controls={<Segments label="Toy cache transition" value={append} onChange={setAppend} options={[[false,'After prefill · T = 3'],[true,'Append new token · T = 4']]}/>} caption="The 32-query-head sharing diagram is tied to the calculator’s KV-head selector. The four-position timeline is a separate small illustrative context; its T = 3/4 does not replace the calculator’s context length. Connections show sharing, not attention weights.">
 <Svg height={650} title="Grouped query heads and cache append" description={`${v.kvHeads} KV heads, ${qPer} query heads per group. Query head h maps to KV head floor(h/${qPer}). Cache tensors after this toy step have shape [${v.batch},${v.kvHeads},${append?4:3},${v.headDim}].`}>
 <ArrowDefs id={id}/><Text x={500} y={35}>32 query heads · {v.kvHeads===32?'MHA':v.kvHeads===1?'MQA':'GQA'} · {qPer} queries share each K/V pair</Text><Text x={500} y={76} small>QUERY HEAD INDEX</Text>
 {Array.from({length:32},(_,q)=>{const x=50+q*29,group=headGroup(q,32,v.kvHeads),center=50+(group*qPer+(qPer-1)/2)*29;return <g key={q}><rect x={x-12} y="94" width="24" height="34" rx="3" fill="var(--fig-bg)" stroke={C[group%4]}/><Text x={x} y={117} small>{q}</Text><Wire d={`M${x} 128V153L${center} 209`} color={C[group%4]} opacity={.8}/></g>;})}
 {Array.from({length:v.kvHeads},(_,g)=>{const x=50+(g*qPer+(qPer-1)/2)*29,w=Math.max(24,qPer*29-5);return <g key={g}><rect x={x-w/2} y="211" width={w} height="39" rx="4" fill="var(--accent-soft)" stroke={C[g%4]}/><Text x={x} y={236} small>{v.kvHeads>16?g:`K/V ${g}`}</Text></g>;})}
 <Text x={500} y={285} small>K/V HEAD INDEX · each box stores both a key-head sequence and a value-head sequence</Text>
 <Panel x={22} y={318} w={956} h={298} title="ONE LAYER · PROJECT THE NEW TOKEN, APPEND K/V ONCE, READ THE CACHE"/>
 <Text x={500} y={378}>K, V each: [B, Hkv, T, dh] = [{v.batch}, {v.kvHeads}, {append?4:3}, {v.headDim}]</Text>
 {[0,1,2,3].map(t=><g key={t}><Box x={125+t*185} y={412} w={158} h={58} label={t<3?`cached K${t+1}, V${t+1}`:append?'append K4, V4':'next K4, V4'} sub={t<3?'reused unchanged':append?'from new input':'not projected yet'} active={t<3||append} color={t===3?'var(--orange)':'var(--accent)'}/>{append&&<Wire d={`M${125+t*185} 471V497H830`} color={t===3?'var(--orange)':'var(--accent)'}/>}</g>)}
 <Box x={886} y={473} w={145} h={50} label={append?'new Q4 reads 1…4':'next query pending'} sub="32 query heads" color="var(--orange)"/>
 <Text x={500} y={560} small>New K/V each have shape [B,Hkv,1,dh]; concatenate along T.</Text><Text x={500} y={595} small>Compute the new query and attention output. Reuse previous positions’ K/V; do not cache their queries for this purpose.</Text>
 </Svg></Figure></>;
}

function TiledAttention(){
 const [tile,setTile]=useState(2),[score,setScore]=useState(2),logits=[0,1,score,-1],values=[[1,0],[0,2],[2,1],[-1,1]],states=onlineAttention(logits,values,2),s=states[tile-1],dense=states.at(-1).output,id=useId().replace(/:/g,'')+'tile';
 return <Figure title="Exact attention · change the memory schedule" controls={<><Segments label="Processed key-value tiles" value={tile} onChange={setTile} options={[[1,'Tile 1 only'],[2,'Tiles 1 + 2']]}/><Range label="Source 3 scaled logit" min={-2} max={5} step={.5} value={score} onChange={setScore}/></>} caption="Computed one-query, four-key example, two keys per tile, two value coordinates. Scores are synthetic scaled dot products; no mask or dropout is used. Both routes retain all four edges. The running output after one tile is only a prefix result; after both tiles it agrees with dense attention.">
 <Svg height={640} title="HBM versus SRAM tiling with online softmax" description={`Scores ${vector(logits)}. Processed ${tile} tiles. Running maximum ${f(s.m)}, normalizer ${f(s.ell)}, numerator ${vector(s.numerator)}, output ${vector(s.output)}; dense result ${vector(dense)}.`}>
 <ArrowDefs id={id}/><Panel x={20} y={18} w={960} h={184} title="HBM · LARGE GLOBAL MEMORY"/>
 <Box x={143} y={80} w={205} h={64} label="Q [1,2] · K,V [4,2]" sub="both routes read these inputs"/>
 <Box x={490} y={80} w={350} h={64} label="Materialize route: S, A each [1,4]" sub="full n-query case: each is n × n" color="var(--orange)"/>
 <Box x={835} y={80} w={226} h={64} label={`Dense O = ${vector(dense)}`} sub="softmax([0,1,s₃,−1]) V" color="var(--orange)"/>
 <Wire d="M246 112H315" id={id}/><Wire d="M665 112H722" id={id}/>
 <Wire d="M143 144V243" id={id} color="var(--accent)"/><Text x={175} y={226} small anchor="start">load Q + current K/V tile</Text>
 <Panel x={20} y={251} w={960} h={267} title="SRAM / ON-CHIP STATE · NEVER WRITE THE FULL S OR A TO HBM"/>
 <Box x={160} y={318} w={226} h={73} label={`Current tile ${tile}: sources ${2*tile-1}, ${2*tile}`} sub={`scores ${vector(logits.slice((tile-1)*2,tile*2))}`} color="var(--accent)"/>
 <Wire d="M273 354H341" id={id}/><Box x={500} y={309} w={315} h={92} label={`m = ${f(s.m)} · ℓ = ${f(s.ell)}`} sub={`u = Σ exp(s−m)v = ${vector(s.numerator)}`} color="var(--accent)"/>
 <Wire d="M657 354H711" id={id}/><Box x={836} y={318} w={246} h={73} label={`O = u / ℓ = ${vector(s.output)}`} sub={tile===2?'complete · equals dense result':'partial · two sources remain'} color="var(--accent)"/>
 <Text x={500} y={449}>Rescale old state by exp(mold − mnew) = {f(s.rescale)}</Text>
 <Text x={500} y={488} small>For a Br-query tile: Q [Br,dk], K [Bc,dk], V [Bc,dv], score tile [Br,Bc].</Text>
 <Text x={500} y={561}>Tile 1: m={f(states[0].m)}, ℓ={f(states[0].ell)}, O={vector(states[0].output)}</Text><Text x={500} y={600}>Tile 2 merge: m={f(states[1].m)}, ℓ={f(states[1].ell)}, O={vector(states[1].output)}</Text>
 <Text x={500} y={633} small>This isolates online-softmax arithmetic, not GPU timing, tile-size tuning or the backward pass.</Text>
 </Svg></Figure>;
}
export function EfficientAttentionMechanismFigure(){return <><TiledAttention/><MaskGraph/></>;}

function AffineQuantizer(){
 const [bits,setBits]=useState(4),[scale,setScale]=useState(.25),weights=[-1.3,-.35,.2,.9,2.1],s=quantizeAffine(weights,bits,scale),lo=-s.zero*scale,hi=(s.max-s.zero)*scale;
 return <Figure title="Numerical compression · inspect rounding and clipping" controls={<><Segments label="Affine quantizer bit width" value={bits} onChange={setBits} options={[[2,'2 bit'],[4,'4 bit'],[8,'8 bit']]}/><Range label="Quantizer scale s" min={.1} max={.5} step={.05} value={scale} display={f(scale,2)} onChange={setScale}/></>} caption="Synthetic unsigned affine quantizer, round to nearest with JavaScript’s half-tie convention, zero point z = 2^(bits−1). This illustrates basic quantization, not GPTQ calibration or a hardware kernel. Storage assumes one float32 scale and one uint8 zero point for the five-weight group.">
 <Svg height={490} title="Computed integer codes, reconstruction and residual error" description={`${bits} bits, scale ${scale}, zero point ${s.zero}. Codes ${s.rows.map(r=>r.code).join(', ')}. Payload ${s.packedBytes} bytes plus ${s.metadataBytes} metadata bytes. MSE ${f(s.mse,5)}.`}>
 <Text x={500} y={36}>q = clip(round(w/s) + z, 0, {s.max}) · ŵ = s(q − {s.zero})</Text>
 {['original w','raw integer','stored q','dequantized ŵ','error ŵ − w','clipping'].map((t,i)=><Text key={t} x={[105,265,420,592,777,917][i]} y={92}>{t}</Text>)}
 {s.rows.map((r,i)=><g key={i}><rect x="25" y={112+i*49} width="950" height="40" rx="5" fill={r.clipped?'var(--accent-soft)':'var(--fig-panel)'} stroke="var(--border)"/><Text x={105} y={138+i*49}>{f(r.w,2)}</Text><Text x={265} y={138+i*49}>{r.raw}</Text><Text x={420} y={138+i*49}>{r.code}</Text><Text x={592} y={138+i*49}>{f(r.reconstructed,2)}</Text><Text x={777} y={138+i*49}>{f(r.error,2)}</Text><Text x={917} y={138+i*49}>{r.clipped?'clipped':'within range'}</Text></g>)}
 <Text x={500} y={391}>Representable range [{f(lo,2)}, {f(hi,2)}] · MSE = {f(s.mse,5)}</Text>
 <Text x={500} y={434}>Payload ceil(5 × {bits} / 8) = {s.packedBytes} B + 4 B scale + 1 B zero point = {s.packedBytes+5} B</Text>
 <Text x={500} y={473} small>Five float32 weights use 20 B. This tiny group exposes metadata overhead; real group sizes and formats differ.</Text>
 </Svg></Figure>;
}
function SpeculativeTrace(){
 const [scenario,setScenario]=useState('second'),p=[.2,.5,.3],q=[.5,.3,.2],names=['A','B','C'],proposals=[0,0,2],uniforms=scenario==='first'?[.8,.2,.2]:scenario==='second'?[.2,.8,.2]:[.2,.2,.2],s=speculativeCycle(p,q,proposals,uniforms,.7),mass=speculativeMass(p,q),id=useId().replace(/:/g,'')+'spec';
 const tokens=xs=>xs.map(i=>names[i]).join(' ')||'∅';
 return <Figure title="Scheduling · verify a draft and follow every branch" controls={<Segments label="Speculative verification outcomes" value={scenario} onChange={setScenario} options={[[ 'first','Reject proposal 1'],['second','Accept 1, reject 2'],['all','Accept all three']]}/>} caption="Computed three-token vocabulary, constant p = [0.2,0.5,0.3] and q = [0.5,0.3,0.2] at every prefix only to keep this toy trace compact. Real pᵢ and qᵢ are separately conditioned on each proposed prefix and include the actual sampling transformations. Draft samples A,A,C and uniforms are stipulated; the final categorical draw uses u = 0.70.">
 <Svg height={700} title="Complete speculative-decoding acceptance and correction paths" description={`Proposals A A C. ${s.rejected<0?'All accepted; sample a bonus token from target p.':`First rejection at proposal ${s.rejected+1}; retain ${tokens(s.prefix)}, discard ${tokens(s.discarded)} and sample correction.`} Output appended to the prior prefix: ${tokens(s.result)}.`}>
 <ArrowDefs id={id}/><Text x={500} y={35}>Draft proposes A → A → C · target evaluates all proposed prefixes in one verification call</Text>
 {proposals.map((token,i)=>{const c=s.checks[i],x=177+i*323;return <g key={i}><Box x={x} y={75} w={250} h={76} label={`Proposal ${i+1}: ${names[token]}`} sub={c?`u=${f(c.u,2)} < α=${f(c.ratio,2)} ? ${c.accepted?'YES':'NO'}`:'not checked after first rejection'} active={!!c} color={c?.accepted?'var(--accent)':'var(--orange)'}/>{i<2&&<Wire d={`M${x+125} 113H${x+198}`} id={id} color="var(--accent)" strokeDasharray={!c?.accepted?'4 4':undefined}/>}</g>;})}
 <Text x={500} y={187} small>α(x) = min(1,p(x)/q(x)); α(A)=0.40, α(B)=1, α(C)=1. Accept → check the next proposal.</Text>
 <Panel x={25} y={220} w={460} h={208} title="FIRST REJECTION · STOP CHECKING"/><Panel x={515} y={220} w={460} h={208} title="ALL ACCEPTED · BONUS TARGET TOKEN"/>
 <Text x={255} y={279}>Retain accepted prefix; discard rejected token</Text><Text x={255} y={310}>and every later proposal. Draw from (p−q)₊ / Z.</Text><Text x={255} y={352}>{s.rejected<0?'This branch is inactive.':`Keep: ${tokens(s.prefix)} · discard: ${tokens(s.discarded)}`}</Text><Text x={255} y={394}>{s.rejected<0?'No correction is required.':`Correction = [0, 2/3, 1/3] → sample ${names[s.extra]}`}</Text>
 <Text x={745} y={279}>Retain every proposed token.</Text><Text x={745} y={310}>Draw one extra token from target p at that prefix.</Text><Text x={745} y={352}>{s.rejected<0?`Keep: ${tokens(proposals)} · bonus: ${names[s.extra]}`:'This branch is inactive.'}</Text><Text x={745} y={394}>{s.rejected<0?'Target p = [0.20, 0.50, 0.30]':'A rejection prevents the bonus branch.'}</Text>
 <Wire d={s.rejected<0?'M745 429V452H500V468':'M255 429V452H500V468'} id={id} color="var(--accent)"/>
 <Box x={500} y={472} w={760} h={53} label={`Append to original context: ${tokens(s.result)}`} sub="Then begin a new draft/verify cycle from this resulting prefix." color="var(--accent)"/>
 <Text x={500} y={563}>One-step probability mass · A / B / C</Text><Text x={500} y={599}>Accepted draft mass min(p,q) = {vector(mass.accepted,2)}</Text><Text x={500} y={634}>Rejection mass {f(mass.rejection,2)} × correction {vector(mass.correction,3)}</Text><Text x={500} y={675}>Total = {vector(mass.recovered,2)} = target p · exact distribution, different possible random samples</Text>
 </Svg></Figure>;
}
export function CompressionDecodingMechanismFigure(){return <><AffineQuantizer/><SpeculativeTrace/></>;}
