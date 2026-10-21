import {readFileSync} from 'node:fs';
import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import katex from 'katex';
const read=name=>JSON.parse(readFileSync(new URL(`../src/content/${name}.json`,import.meta.url),'utf8'));
const articles=read('articles'),sources=read('sources'),categories=read('categories'),visuals=read('visuals'),collections=read('collections');
const sourceMap=new Map(sources.map(s=>[s.id,s])),slugs=new Set(articles.map(a=>a.slug)),cats=new Set(categories.map(c=>c.id));
let mathCount=0;const errors=[];
if(slugs.size!==articles.length)errors.push('Duplicate article slug');
if(sourceMap.size!==sources.length)errors.push('Duplicate source ID');
for(const c of categories){
 if(!articles.some(a=>a.category===c.id))errors.push(`Empty topic group ${c.id}`);
 for(const id of c.related||[])if(!cats.has(id)||id===c.id)errors.push(`Invalid related group ${c.id} -> ${id}`);
}
for(const c of collections)for(const slug of c.articles)if(!slugs.has(slug))errors.push(`Collection ${c.id}: missing topic ${slug}`);
for(const s of sources){if(!s.url.startsWith('https://')||!s.title||!s.authors||!s.evidence||!s.checked)errors.push(`Incomplete source ${s.id}`);}
for(const a of articles){
 const body=readFileSync(new URL(`../src/content/articles/${a.slug}.md`,import.meta.url),'utf8');
 if(!cats.has(a.category))errors.push(`Unknown category ${a.category}`);
 if(!a.diagram||!visuals[a.diagram])errors.push(`Missing visual assignment: ${a.slug}`);
 if(a.keyPoints?.length!==3||a.keyPoints.some(x=>!x.trim()))errors.push(`Missing three preview points: ${a.slug}`);
 if(!a.tags?.length)errors.push(`Missing technique tags: ${a.slug}`);
 if(a.related?.length<3)errors.push(`Missing related topics: ${a.slug}`);
 const related=new Set();for(const r of a.related||[]){if(!slugs.has(r.slug)||r.slug===a.slug||!r.reason||related.has(r.slug))errors.push(`Invalid related topic ${a.slug} -> ${r.slug}`);related.add(r.slug);}
 if(body.split(/\s+/).length<200)errors.push(`Article too short: ${a.slug}`);
 if(!a.references.length)errors.push(`Uncited article: ${a.slug}`);
 const h=[...body.matchAll(/^## (.+)$/gm)].map(m=>m[1]);if(h.length<4)errors.push(`Missing article structure: ${a.slug}`);
 for(const id of a.references)if(!sourceMap.has(id))errors.push(`${a.slug}: missing reference ${id}`);
 for(const m of body.matchAll(/\[(\d+)\]\(#ref-([\w-]+)\)/g)){if(!a.references.includes(m[2]))errors.push(`${a.slug}: undeclared inline reference ${m[2]}`);if(a.references[+m[1]-1]!==m[2])errors.push(`${a.slug}: reference number mismatch for ${m[2]}`);}
 for(const m of body.matchAll(/#\/article\/([\w-]+)/g))if(!slugs.has(m[1]))errors.push(`${a.slug}: broken cross-link ${m[1]}`);
 const tree=unified().use(remarkParse).use(remarkMath).parse(body);
 function visit(node){if(node.type==='math'||node.type==='inlineMath'){mathCount++;try{katex.renderToString(node.value,{throwOnError:true,strict:false,displayMode:node.type==='math'});}catch(e){errors.push(`${a.slug}: invalid TeX ${node.value}: ${e.message}`);}}for(const child of node.children||[])visit(child);}visit(tree);
}
if(errors.length){console.error(errors.join('\n'));process.exit(1);}
console.log(`Content validated: ${articles.length} articles, ${categories.length} topic groups, ${sources.length} primary references, ${mathCount} TeX expressions; previews, visuals, related topics and numbered citations resolve.`);
