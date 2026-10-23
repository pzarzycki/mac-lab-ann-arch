import test from 'node:test';
import assert from 'node:assert/strict';
import {canonicalAuthorList,bibtex} from '../src/bibliography.js';
test('BibTeX author lists distinguish people, omitted names and corporate authors',()=>{
 assert.equal(canonicalAuthorList('Rumelhart, Hinton & Williams'),'Rumelhart and Hinton and Williams');
 assert.equal(canonicalAuthorList('Nora Belrose et al.'),'Nora Belrose and others');
 assert.equal(canonicalAuthorList('Emmanuel Ameisen, Jack Lindsey, Adam Pearce et al.'),'Emmanuel Ameisen and Jack Lindsey and Adam Pearce and others');
 assert.equal(canonicalAuthorList('Meta FAIR'),'{Meta FAIR}');
 assert.equal(canonicalAuthorList('DeepSeek-AI et al.'),'{DeepSeek-AI} and others');
 assert.equal(canonicalAuthorList('Seely & Gould · Sakana AI'),'Seely and Gould');
});
test('explicit structured author order wins over abbreviated display labels',()=>{
 const out=bibtex([{id:'example-paper',title:'A & B_1',authors:'Smith et al.',bibtexAuthors:'Smith, Alice and Jones, Bob',year:2026,url:'https://example.org/a_b',checked:'2026-10-22'}]);
 assert.ok(out.includes('author = {Smith, Alice and Jones, Bob}'));
 assert.ok(out.includes('title = {A \\& B\\_1}'));
 assert.ok(out.includes('url = {https://example.org/a_b}'));
});
