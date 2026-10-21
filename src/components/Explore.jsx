import React,{useEffect,useMemo,useRef,useState} from 'react';
import {ArrowRight,ArrowUpRight,Bookmark,Search,X,Orbit,GitBranch} from 'lucide-react';
import {articles,categories,articleMap,collections} from '../content';
import ArticleVisual from './ArticleVisual';

export function TopicCard({article:a,onPreview,reason,compact=false}){
 const group=categories.find(c=>c.id===a.category);
 return <article className={`topic-card ${compact?'topic-card-compact':''}`}>
  <div className="topic-card-top"><a href={`#/section/${a.category}`}>{group.title}</a><span><Orbit size={11}/> Visual guide</span></div>
  {reason&&<p className="connection-reason"><GitBranch size={12}/>{reason}</p>}
  <h3><a href={`#/article/${a.slug}`}>{a.title}</a></h3>
  <p className="topic-summary">{a.abstract}</p>
  {!compact&&<ul className="key-points">{a.keyPoints.map(p=><li key={p}>{p}</li>)}</ul>}
  <div className="topic-tags">{a.tags.slice(0,3).map(t=><span key={t}>{t}</span>)}</div>
  <div className="topic-card-actions"><button onClick={()=>onPreview(a.slug)}>Quick view <Orbit size={13}/></button><a href={`#/article/${a.slug}`}>Deep dive <ArrowUpRight size={14}/></a></div>
 </article>;
}

export function TopicGrid({list,onPreview,compact=false}){
 return <div className="topic-grid">{list.map(a=><TopicCard key={a.slug} article={a} onPreview={onPreview} compact={compact}/>)}</div>;
}

export function GroupPage({category,onPreview}){
 const [query,setQuery]=useState('');
 useEffect(()=>setQuery(''),[category.id]);
 const all=articles.filter(a=>a.category===category.id), words=query.toLowerCase().trim();
 const list=all.filter(a=>`${a.title} ${a.abstract} ${a.tags.join(' ')}`.toLowerCase().includes(words));
 return <div className="section-page explore-group"><header className="page-intro"><div className="eyebrow">TOPIC GROUP · {all.length} DEEP DIVES</div><h1>{category.title}</h1><p>{category.description}</p></header>
  <div className="group-question"><span className="eyebrow">A QUESTION TO EXPLORE</span><p>{category.question}</p></div>
  <div className="directory-toolbar"><label className="filter-search"><Search size={16}/><input aria-label={`Search ${category.title}`} value={query} onChange={e=>setQuery(e.target.value)} placeholder="Find a mechanism or technique in this group…"/></label><span>{list.length} topics</span></div>
  <TopicGrid list={list} onPreview={onPreview}/>
  {!list.length&&<p className="directory-empty">No topics match. Try a broader mechanism or clear the search.</p>}
  <nav className="related-groups" aria-label="Connected topic groups"><span className="eyebrow">CONNECTED GROUPS</span>{category.related.map(id=>{const c=categories.find(c=>c.id===id);return <a href={`#/section/${id}`} key={id}>{c.title}<ArrowRight size={14}/></a>;})}</nav>
 </div>;
}

export function ExplorePage({onPreview}){
 const [query,setQuery]=useState(''),[group,setGroup]=useState('all'),[question,setQuestion]=useState('all');
 const selected=collections.find(c=>c.id===question);
 const list=useMemo(()=>articles.filter(a=>(group==='all'||a.category===group)&&(!selected||selected.articles.includes(a.slug))&&query.toLowerCase().trim().split(/\s+/).every(w=>`${a.title} ${a.subtitle} ${a.abstract} ${a.tags.join(' ')} ${a.keyPoints.join(' ')}`.toLowerCase().includes(w))),[query,group,selected]);
 return <div className="explore-page"><header className="page-intro"><div className="eyebrow">CONNECTIONS, MECHANISMS, QUESTIONS</div><h1>Follow an idea.</h1><p>Browse a mechanism, compare approaches, or follow a question across fields. Quick views expose the key distinctions and a live diagram; every topic opens into a source-linked deep dive.</p></header>
  <section className="question-collections" aria-label="Explore by question"><div className="section-top"><span className="eyebrow">EXPLORE BY QUESTION</span><button className="text-link" onClick={()=>{setQuestion('all');setGroup('all');setQuery('');}}>Show all topics <ArrowRight size={13}/></button></div><div className="question-grid">{collections.map(c=><button key={c.id} aria-pressed={question===c.id} className={question===c.id?'selected':''} onClick={()=>{setQuestion(question===c.id?'all':c.id);setGroup('all');}}><h2>{c.title}</h2><p>{c.description}</p><span>Explore connections <ArrowUpRight size={13}/></span></button>)}</div></section>
  <div className="filter-bar directory-filters"><label className="filter-search"><Search size={17}/><input aria-label="Filter topics and techniques" placeholder="Search topics and techniques: JEPA, MoE, backprop…" value={query} onChange={e=>setQuery(e.target.value)}/></label><select aria-label="Filter topics by group" value={group} onChange={e=>setGroup(e.target.value)}><option value="all">All topic groups</option>{categories.map(c=><option key={c.id} value={c.id}>{c.title}</option>)}</select></div>
  <div className="results-heading">{list.length} topics<span>{selected?selected.title:'BROWSE IN ANY DIRECTION'}</span></div><TopicGrid list={list} onPreview={onPreview}/>
  {!list.length&&<p className="directory-empty">No matches in this selection. Clear the question or choose another group.</p>}
 </div>;
}

export function RelatedTopics({article,onPreview}){
 return <section className="related-topics" id="related-topics"><div className="section-top"><div><span className="eyebrow">CONTINUE THE EXPLORATION</span><h2>Related ideas.</h2></div><a className="text-link" href="#/explore">Browse the atlas <ArrowRight size={14}/></a></div><div className="topic-grid">{article.related.map(r=><TopicCard key={r.slug} article={articleMap[r.slug]} reason={r.reason} onPreview={onPreview} compact/>)}</div></section>;
}

export function PreviewDialog({slug,onClose,onBookmark,saved}){
 const a=articleMap[slug],panel=useRef(),previous=useRef(document.activeElement);
 useEffect(()=>{panel.current?.querySelector('button')?.focus();const old=document.body.style.overflow;document.body.style.overflow='hidden';return()=>{document.body.style.overflow=old;previous.current?.focus?.();};},[]);
 useEffect(()=>{const handler=e=>{if(e.key==='Escape'&&!document.querySelector('.figure-expanded'))onClose();};addEventListener('keydown',handler);return()=>removeEventListener('keydown',handler);},[onClose]);
 function keys(e){if(e.key!=='Tab')return;const nodes=[...panel.current.querySelectorAll('button,a,input,select')].filter(el=>!el.disabled),first=nodes[0],last=nodes.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}
 return <div className="preview-overlay" onMouseDown={e=>{if(e.target===e.currentTarget)onClose();}}><section className="preview-dialog" ref={panel} role="dialog" aria-modal="true" aria-labelledby="preview-title" onKeyDown={keys}><div className="preview-top"><span className="eyebrow">TOPIC PREVIEW</span><button className="icon-button" aria-label="Close topic preview" onClick={onClose}><X size={19}/></button></div><div className="preview-content"><a className="eyebrow" href={`#/section/${a.category}`} onClick={onClose}>{categories.find(c=>c.id===a.category).title}</a><h2 id="preview-title">{a.title}</h2><p className="preview-abstract">{a.abstract}</p><ul className="key-points">{a.keyPoints.map(p=><li key={p}>{p}</li>)}</ul><div className="preview-actions"><a className="primary-button" href={`#/article/${a.slug}`} onClick={onClose}>Open deep dive<ArrowRight size={15}/></a><button className="text-link" onClick={()=>onBookmark(a.slug)}><Bookmark size={14} fill={saved.includes(a.slug)?'currentColor':'none'}/>{saved.includes(a.slug)?'Saved':'Save for later'}</button><span>{a.references.length} primary sources</span></div><ArticleVisual type={a.diagram}/><div className="preview-related"><span className="eyebrow">CONNECTED IDEAS</span>{a.related.slice(0,3).map(r=><a href={`#/article/${r.slug}`} key={r.slug} onClick={onClose}>{articleMap[r.slug].title}<ArrowUpRight size={13}/></a>)}</div></div></section></div>;
}
