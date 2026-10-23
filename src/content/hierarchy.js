import hierarchy from './hierarchy.json';
export {hierarchy};
export const nodes=hierarchy.flatMap(function flatten(n){return [n,...(n.children||[]).flatMap(flatten)];});
export const nodeMap=Object.fromEntries(nodes.map(n=>[n.id,n]));
export const descendantSlugs=n=>[...(n.articles||[]),...(n.children||[]).flatMap(descendantSlugs)];
export function nodePath(id,list=hierarchy,path=[]){for(const n of list){const next=[...path,n];if(n.id===id)return next;const found=nodePath(id,n.children||[],next);if(found.length)return found;}return [];}
export const articlePaths=Object.fromEntries(nodes.flatMap(n=>(n.articles||[]).map(slug=>[slug,nodePath(n.id)])));
