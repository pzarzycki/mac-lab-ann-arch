import test from 'node:test';
import assert from 'node:assert/strict';
import {deterministicDenoise,flowTrace,maskedTrace,embeddingMixture,latentTrace,quantizeCoordinate,actionTrace,rolloutTiming,wilsonInterval,jepaExample,rankPlans} from '../src/generation-embodiment-math.js';
const near=(a,b,t=1e-10)=>assert.ok(Math.abs(a-b)<t,`${a} ≠ ${b}`);
test('Gaussian reverse step recovers a known clean point and reaches the clean endpoint',()=>{
 const x=.5*2+Math.sqrt(.75)*(-1),r=deterministicDenoise(x,.25,1,-1);near(r.clean,2);near(r.next,2);
 near(deterministicDenoise(x,.25,.25,-1).next,x);
 assert.throws(()=>deterministicDenoise(1,0,1,0),RangeError);
});
test('Euler transport has independently derived O(1/N) endpoint error',()=>{
 for(const n of [1,2,4,16]){const [x,y]=flowTrace(n).at(-1);near(x,1);near(.5-y,.5/n);}
 assert.throws(()=>flowTrace(0),RangeError);
});
test('masked evaluation differs from acceptance; remasking reopens a position',()=>{
 const t=maskedTrace(5);assert.equal(t[1].proposals.length,8);assert.equal(t[1].accepted.length,2);assert.equal(t[2].remasked.length,1);
 assert.equal(t[2].visible[t[2].remasked[0]],null);assert.equal(t.at(-1).visible.filter(Boolean).length,8);
 assert.equal(maskedTrace(0)[0].visible.filter(Boolean).length,0);
 assert.equal(maskedTrace(2,false)[2].visible.filter(Boolean).length,4);
});
test('embedding mixtures remain convex; bypass preserves the continuous state',()=>{
 const m=embeddingMixture(.5);near(m.weights.reduce((a,b)=>a+b,0),1);near(m.vector[0],.3);near(m.vector[1],.6);
 assert.deepEqual(embeddingMixture(1).vector,[1,0]);
 near(latentTrace(1,true).states[1][0],.5);near(latentTrace(1,true).states[1][1],.3);
 assert.deepEqual(latentTrace(1,false).inputs[0].embedding,[0,2]);
 assert.deepEqual(latentTrace(0).states,[[.2,.8]]);
});
test('quantization bounds and per-bin reconstruction error',()=>{
 assert.equal(quantizeCoordinate(2).bin,4);near(quantizeCoordinate(2).decoded,1.25);
 assert.equal(quantizeCoordinate(-999).bin,0);assert.equal(quantizeCoordinate(10).bin,7);
 for(let v=-10;v<=10;v+=.125) assert.ok(Math.abs(quantizeCoordinate(v).decoded-v)<=1.25);
});
test('action samplers preserve H×d and distinct finite-step behavior',()=>{
 for(const mode of ['tokens','flow','diffusion']){const t=actionTrace(mode,4);assert.equal(t.chunk.length,4);assert.ok(t.chunk.every(r=>r.length===2 && r.every(Number.isFinite)));}
 const f=actionTrace('flow',4);near(f.chunk[0][0],2+(-6-2)*(.75**4));
 for(const n of [1,4,6]){const d=actionTrace('diffusion',n);d.chunk.forEach((r,i)=>r.forEach((v,j)=>near(v,d.center[i][j])));}
});
test('pipeline scheduling exposes queueing and age rather than assuming lower latency',()=>{
 const serial=rolloutTiming(40,2,false),piped=rolloutTiming(40,2,true);
 assert.equal(serial.frames[0].use,65);assert.equal(serial.frames[1].age,65);
 assert.equal(serial.frames[1].use,170);assert.equal(piped.frames[1].use,105);assert.equal(piped.frames[1].age,80);assert.equal(piped.queue,15);
 assert.ok(piped.frames[1].inference[0]>=piped.frames[0].inference[1]);
});
test('Wilson bounds handle all-failure/all-success and tiny illustrative cohort',()=>{
 const [lo,hi]=wilsonInterval(1,3);near(lo,.061490315276160515);near(hi,.7923450448735121);
 near(wilsonInterval(0,3)[0],0);near(wilsonInterval(3,3)[1],1);assert.throws(()=>wilsonInterval(1,0),RangeError);
});
test('JEPA row IDs align with a mean absolute feature loss and EMA arithmetic',()=>{
 const j=jepaExample();near(j.loss,.15);assert.deepEqual(j.rows.map(r=>r.position),[7,8,12,13]);near(j.ema[0],.46);near(j.ema[1],.74);
 near(jepaExample('scattered',0).loss,0);assert.ok(jepaExample('scattered').variance>j.variance);
 near(jepaExample('block',1,0).ema[0],1);
});
test('goal scores are reproducible from a shared scalar trace and ties are deterministic',()=>{
 assert.equal(rankPlans(34)[0].name,'B');assert.equal(rankPlans(-35)[0].name,'A');
 assert.deepEqual(rankPlans(-35).map(p=>p.score),[0,47,69]);assert.equal(rankPlans(23)[0].name,'B');
});
