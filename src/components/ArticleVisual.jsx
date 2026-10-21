import React from 'react';
import ArticleFigure from './Figures';
import {ConceptFigure,conceptFigureTypes} from './ConceptFigures';
import {JepaFigure,WorldModelFigure} from './JepaFigures';
import {NeatFigure,HyperNeatFigure} from './EvolutionFigures';
import visualFamilies from '../content/visuals.json';

const extra={jepa:JepaFigure,'world-models':WorldModelFigure,neat:NeatFigure,hyperneat:HyperNeatFigure};
export default function ArticleVisual({type}){
 const Component=extra[type];
 if(Component)return <Component/>;
 if(visualFamilies[type]==='concept'&&conceptFigureTypes[type])return <ConceptFigure type={type}/>;
 return <ArticleFigure type={type}/>;
}
