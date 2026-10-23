import test from 'node:test';
import assert from 'node:assert/strict';
import {add,dot,normStats,rotate2,swigluToy,softmax,attentionTrace,onlineAttention,quantizeAffine,speculativeMass,speculativeCycle,kvMemory,headGroup} from '../src/transformer-mechanisms-math.js';
const close=(a,b,eps=1e-10)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
const vec=(a,b)=>a.forEach((v,i)=>close(v,b[i]));
test('residual writes preserve untouched coordinates',()=>{vec(add(add([1,-1,.5],[.5,.25,-.5]),[-.25,.5,1]),[1.25,-.25,1]);});
test('LN centers, is shift invariant; RMSNorm retains mean and handles zero',()=>{
 const a=normStats([-1,1,2]),b=normStats([4,6,7]);vec(a.ln,b.ln);close(a.ln.reduce((s,v)=>s+v,0),0);
 assert.ok(a.rmsnorm.reduce((s,v)=>s+v,0)>0);vec(normStats([0,0]).ln,[0,0]);vec(normStats([0,0]).rmsnorm,[0,0]);
});
test('RoPE common shift and relative-angle identity, including negative displacement',()=>{
 for(const [i,j] of [[0,2],[3,1],[4,4]]){const q=[1,.3],k=[-.2,1],w=Math.PI/6;
 close(dot(rotate2(q,i*w),rotate2(k,j*w)),dot(q,rotate2(k,(j-i)*w)));
 close(dot(rotate2(q,(i+7)*w),rotate2(k,(j+7)*w)),dot(rotate2(q,i*w),rotate2(k,j*w)));}
});
test('parallel gated transform has independently derived zero-input-coordinate example',()=>{
 const s=swigluToy([0,1]),sig=1/(1+Math.E);vec(s.gate,[0,1,-1]);vec(s.up,[1,-1,1]);vec(s.output,[-sig,-Math.E/(1+Math.E)+sig]);
});
test('scaled logits, masking and weighted sums agree with hand derivation',()=>{
 const s=attentionTrace([Math.SQRT2,0],[[1,0],[0,1],[1,1],[-1,0]],[[1,0],[0,2],[2,1],[-1,1]],[true,true,true,false]),z=2*Math.E+1;
 vec(s.logits,[1,0,1,-1]);vec(s.weights,[Math.E/z,1/z,Math.E/z,0]);vec(s.output,[3*Math.E/z,(2+Math.E)/z]);
 const edited=attentionTrace([Math.SQRT2,0],[[1,0],[0,1],[1,1],[-1,0]],[[1,0],[0,2],[4,1],[-1,1]],[true,true,true,false]);vec(edited.weights,s.weights);close(edited.output[0]-s.output[0],2*Math.E/z);
});
test('online softmax matches dense for partition sizes, extreme logits and masked tiles',()=>{
 const logits=[1001,-Infinity,1000,999,-Infinity],values=[[1,2],[8,8],[-1,4],[3,-2],[9,9]],w=softmax(logits),dense=[0,1].map(j=>w.reduce((s,v,i)=>s+v*values[i][j],0));
 for(const size of [1,2,3,8])vec(onlineAttention(logits,values,size).at(-1).output,dense);
 vec(onlineAttention([-Infinity,-Infinity,0],[[99],[99],[7]],2).at(-1).output,[7]);
 vec(onlineAttention([-Infinity,-Infinity],[[1],[2]],1).at(-1).output,[0]);
 assert.throws(()=>onlineAttention([0],[[1]],0),RangeError);
});
test('quantizer exposes clipping, exact codes and actual payload plus metadata',()=>{
 const q=quantizeAffine([-1.3,-.35,.2,.9,2.1],4,.25);assert.deepEqual(q.rows.map(r=>r.code),[3,7,9,12,15]);vec(q.rows.map(r=>r.reconstructed),[-1.25,-.25,.25,1,1.75]);assert.equal(q.rows.at(-1).clipped,true);assert.equal(q.packedBytes,3);assert.equal(q.metadataBytes,5);assert.throws(()=>quantizeAffine([1],4,0),RangeError);
});
test('speculative mass recovers target; all accept and first/second rejection retain correct prefix',()=>{
 const p=[.2,.5,.3],q=[.5,.3,.2],s=speculativeMass(p,q);vec(s.accepted,[.2,.3,.2]);close(s.rejection,.3);vec(s.correction,[0,2/3,1/3]);vec(s.recovered,p);
 const first=speculativeCycle(p,q,[0,0,2],[.8,.2,.2]);assert.equal(first.rejected,0);assert.deepEqual(first.result,[2]);assert.deepEqual(first.discarded,[0,0,2]);
 const second=speculativeCycle(p,q,[0,0,2],[.2,.8,.2]);assert.equal(second.rejected,1);assert.deepEqual(second.result,[0,2]);
 const all=speculativeCycle(p,q,[0,0,2],[.2,.2,.2]);assert.equal(all.rejected,-1);assert.deepEqual(all.result,[0,0,2,2]);
 close(speculativeMass(p,p).rejection,0);vec(speculativeMass(p,p).recovered,p);
});
test('cache accounting and grouped-head endpoints',()=>{
 const v={batch:1,layers:32,context:8192,kvHeads:8,headDim:128,bytes:2};assert.equal(kvMemory(v),2**30);assert.equal(kvMemory({...v,kvHeads:32}),4*2**30);
 assert.deepEqual(Array.from({length:8},(_,i)=>headGroup(i,8,2)),[0,0,0,0,1,1,1,1]);assert.equal(headGroup(31,32,1),0);assert.equal(headGroup(31,32,32),31);assert.throws(()=>headGroup(2,8,3),RangeError);
});
