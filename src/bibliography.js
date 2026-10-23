// Display author labels can be abbreviated. BibTeX needs explicit name
// separators and the special `and others` sentinel, rather than literal et al.
export function canonicalAuthorList(display){
 const label=String(display).split(' · ')[0].trim();
 if(['Meta FAIR','Anthropic','DeepSeek-AI'].includes(label))return `{${label}}`;
 const omitted=/\bet al\.?\s*$/.test(label);
 const names=label.replace(/\s+et al\.?\s*$/,'').split(/\s*&\s*|\s*,\s*|\s+and\s+/).filter(Boolean);
 return [...names.map(n=>n==='DeepSeek-AI'?`{${n}}`:n),...(omitted?['others']:[])].join(' and ');
}
const text=value=>String(value).replace(/[&%_$#]/g,c=>'\\'+c);
export function bibtex(list){return list.map(s=>`@misc{neuralatlas_${s.id.replace(/-/g,'_')},\n  title = {${text(s.title)}},\n  author = {${s.bibtexAuthors||canonicalAuthorList(s.authors)}},\n  year = {${s.year}},\n  url = {${s.url}},\n  note = {Source checked ${s.checked}}\n}`).join('\n\n');}
