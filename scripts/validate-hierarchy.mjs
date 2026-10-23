import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const read=n=>JSON.parse(readFileSync(new URL(`../src/content/${n}.json`,import.meta.url),'utf8'));
const articles=read('articles'), roots=read('hierarchy'), actual=[],ids=new Set();
function visit(n){assert(n.id&&n.title&&n.description,`Incomplete family ${n.id}`);assert(!ids.has(n.id),`Duplicate family ${n.id}`);ids.add(n.id);assert(n.children?.length||n.articles?.length,`Empty family ${n.id}`);actual.push(...(n.articles||[]));for(const c of n.children||[])visit(c);}
roots.forEach(visit);
assert.equal(actual.length,new Set(actual).size,'An article appears in multiple primary families');
assert.deepEqual([...actual].sort(),articles.map(a=>a.slug).sort(),'Hierarchy must contain every article exactly once');
const app=readFileSync(new URL('../src/App.jsx',import.meta.url),'utf8');
assert(!/href="\.\/manuscript.md"/.test(app),'Manuscript links must open the reading page');
console.log(`Hierarchy validated: ${roots.length} domains, ${ids.size} domain/family pages; all ${actual.length} article URLs retained exactly once. Manuscript links render in the site.`);
