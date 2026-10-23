import {readFileSync} from 'node:fs';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';

// Render the actual registry-selected components so a declared but unresolved
// figure cannot silently leave an article empty.
const articles=JSON.parse(readFileSync(new URL('../src/content/articles.json',import.meta.url),'utf8'));
const server=await createServer({server:{middlewareMode:true},appType:'custom'});
try{
 const {default:Visual,articleVisualTypes}=await server.ssrLoadModule('/src/components/ArticleVisual.jsx');
 const {conceptDiagrams}=await server.ssrLoadModule('/src/components/ConceptFigures.jsx');
 for(const a of articles){
  if(!articleVisualTypes.has(a.diagram))throw Error(`No explicit worked-visual renderer: ${a.slug} / ${a.diagram}`);
  const html=renderToStaticMarkup(createElement(Visual,{type:a.diagram}));
  if(/(?:NaN|Infinity)/.test(html))throw Error(`Non-finite rendered value: ${a.slug}`);
  if(html.length<400)throw Error(`Empty visual: ${a.slug} / ${a.diagram}`);
  if(!['kv','timeline'].includes(a.diagram)&&!/<svg\b[^>]*viewBox=/.test(html))throw Error(`Missing vector diagram: ${a.slug}`);
 }
 for(const [slug,d] of Object.entries(conceptDiagrams)){
  const ids=new Set(d.nodes.map(n=>n.id));
  for(const n of d.nodes)if(n.x-n.w/2<0||n.x+n.w/2>840||n.y<0||n.y+n.h>d.height)throw Error(`Out-of-bounds node ${slug}/${n.id}`);
  for(const e of d.edges)if(!ids.has(e.from)||!ids.has(e.to))throw Error(`Unresolved graph edge ${slug}`);
 }
 console.log(`Rendered all ${articles.length} article visuals; ${Object.keys(conceptDiagrams).length} concept graphs have resolved edges and bounded nodes.`);
}finally{await server.close();}
