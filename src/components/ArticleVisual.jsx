import React from 'react';
import ArticleFigure from './Figures';
import {ConceptFigure,conceptFigureTypes} from './ConceptFigures';
import {JepaFigure,WorldModelFigure} from './JepaFigures';
import {NeatFigure,HyperNeatFigure} from './EvolutionFigures';
import {EmbeddingFigure,ObservabilityFigure,LensReadoutFigure,JacobianFigure,PatchingFigure} from './ObservabilityFigures';
import visualFamilies from '../content/visuals.json';

const extra={'embedding-space':EmbeddingFigure,'observability-map':ObservabilityFigure,'lens-readouts':LensReadoutFigure,'jacobian-transport':JacobianFigure,'causal-patching':PatchingFigure,jepa:JepaFigure,'world-models':WorldModelFigure,neat:NeatFigure,hyperneat:HyperNeatFigure};
export default function ArticleVisual({type}){
 const Component=extra[type];
 if(Component)return <Component/>;
 if(visualFamilies[type]==='concept'&&conceptFigureTypes[type])return <ConceptFigure type={type}/>;
 return <ArticleFigure type={type}/>;
}
