import test from 'node:test';
import assert from 'node:assert/strict';
import {toyEmbedding,toyLens,toyJacobian,patchRecovery} from '../src/observability-math.js';
test('categorical lookup and a convex soft mixture preserve the stated toy geometry',()=>{assert.deepEqual(toyEmbedding(2),[-1,0]);assert.deepEqual(toyEmbedding(0,.25),[.25,.75]);assert.throws(()=>toyEmbedding(0,1.1),RangeError);});
test('a coordinate translator changes the decoded token, while softmax stays normalized',()=>{const a=toyLens('raw'),b=toyLens('translated');assert.deepEqual(a.logits,[0,1,-1]);assert.deepEqual(b.logits,[1,0,-1]);assert.ok(a.probabilities[1]>a.probabilities[0]);assert.ok(b.probabilities[0]>b.probabilities[1]);assert.ok(Math.abs(b.probabilities.reduce((s,p)=>s+p,0)-1)<1e-12);});
test('mean Jacobians can cancel a direction and patch recovery can overshoot',()=>{assert.deepEqual(toyJacobian(.5),[[1,0],[0,0]]);assert.deepEqual(toyJacobian(1),[[1,0],[0,1]]);assert.equal(patchRecovery(4,0,5),1.25);assert.equal(patchRecovery(1,1,2),null);});
