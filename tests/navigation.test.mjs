import test from 'node:test';
import assert from 'node:assert/strict';
import {countLabel,glossaryMatches,legacyDomains} from '../src/navigation.js';
test('glossary resolves common technical spellings and expansions',()=>{
 const ln=['Layer Normalization (LN)','Feature normalization across one token.'];
 assert.equal(glossaryMatches(ln,'LayerNorm'),true);
 assert.equal(glossaryMatches(ln,'layer norm'),true);
 assert.equal(glossaryMatches(['Query / key / value','Three projections.'],'QKV'),true);
 assert.equal(glossaryMatches(['GQA','Shared key/value heads.'],'grouped query attention'),true);
 assert.equal(glossaryMatches(ln,'unknown mechanism'),false);
 assert.equal(glossaryMatches(ln,''),true);
});
test('counts and legacy scope distinguish singular wording and retained group bridges',()=>{
 assert.equal(countLabel(1,'topic'),'1 topic');
 assert.equal(countLabel(0,'topic'),'0 topics');
 assert.equal(countLabel(1,'family','families'),'1 family');
 assert.deepEqual(legacyDomains.learning,['learning-adaptation','inference-control']);
});
