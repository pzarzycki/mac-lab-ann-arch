import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';
const read=n=>JSON.parse(readFileSync(new URL(`../src/content/${n}.json`,import.meta.url),'utf8'));
const articles=read('articles'),categories=read('categories'),roots=read('hierarchy');
const nodes=roots.flatMap(function flatten(n){return [n,...(n.children||[]).flatMap(flatten)];});
globalThis.location={hash:''};
// Only browser-independent initial rendering runs here; effects and interactions
// are checked in the actual browser preview.
const server=await createServer({server:{middlewareMode:true},appType:'custom'});
try{
 const {default:App}=await server.ssrLoadModule('/src/App.jsx');
 const routes=['/','/manuscript','/explore','/papers','/glossary','/saved','/about',...articles.map(a=>'/article/'+a.slug),...nodes.map(n=>'/domain/'+n.id),...categories.map(c=>'/section/'+c.id)];
 for(const route of routes){globalThis.location.hash='#'+route;const html=renderToStaticMarkup(React.createElement(App));assert(!html.includes('This page is not in the atlas'),`Missing route ${route}`);assert(!html.includes('katex-error'),`Math render error ${route}`);assert(html.includes('<h1'),`Missing page heading ${route}`);}
 console.log(`Rendered ${routes.length} routes: all ${articles.length} articles, ${nodes.length} domain/family pages, manuscript, reference views and ${categories.length} legacy sections.`);
}finally{await server.close();}
