import React from 'react';
import ArticleFigure from './Figures';
import {ConceptFigure,conceptFigureTypes} from './ConceptFigures';
import {NeatFigure,HyperNeatFigure} from './EvolutionFigures';
import {EmbeddingFigure,ObservabilityFigure,LensReadoutFigure,JacobianFigure,PatchingFigure} from './ObservabilityFigures';
import visualFamilies from '../content/visuals.json';
import {ResidualMechanismFigure,AttentionMechanismFigure,NormalizationMechanismFigure,TrainingDecodingMechanismFigure,KVMechanismFigure,EfficientAttentionMechanismFigure,CompressionDecodingMechanismFigure} from './TransformerMechanisms';
import {DynamicsMoEFigure,DynamicsMHCFigure,DynamicsRecurrentFigure,DynamicsStateSpaceFigure,DynamicsSpikingFigure,DynamicsLiquidFigure} from './DynamicsMechanisms';
import {BackpropMechanism,LocalLearningMechanism,PCInferenceMechanism,PCALMMechanism,ActiveInferenceMechanism,ContinualLearningMechanism,MemoryLifecycleMechanism,SelfImprovementMechanism} from './LearningMechanisms';
import {DistributedRepresentationsFigure,SequenceAttentionHistoryFigure,ResnetLineageFigure,GptScalingFigure,MultimodalFusionFigure,MultimodalObjectivesFigure,TemporalModalitiesFigure} from './RepresentationMechanisms';
import {DiffusionFlowMechanism,SoftLatentMechanism,VLAFoundationMechanism,ActionSamplingMechanism,RobotEvaluationMechanism,JepaMechanism,WorldPlanningMechanism} from './GenerationEmbodimentMechanisms';

const extra={'embedding-space':EmbeddingFigure,'observability-map':ObservabilityFigure,'lens-readouts':LensReadoutFigure,'jacobian-transport':JacobianFigure,'causal-patching':PatchingFigure,'jepa':JepaMechanism,'world-models':WorldPlanningMechanism,'neat':NeatFigure,'hyperneat':HyperNeatFigure,'residual':ResidualMechanismFigure,'attention':AttentionMechanismFigure,'normalization-position':NormalizationMechanismFigure,'training-decoding':TrainingDecodingMechanismFigure,'kv':KVMechanismFigure,'efficient-attention':EfficientAttentionMechanismFigure,'compression-decoding':CompressionDecodingMechanismFigure,'moe':DynamicsMoEFigure,'mhc':DynamicsMHCFigure,'recurrent':DynamicsRecurrentFigure,'state-space':DynamicsStateSpaceFigure,'spiking-networks':DynamicsSpikingFigure,'liquid-networks':DynamicsLiquidFigure,'backprop-biology':BackpropMechanism,'local-learning':LocalLearningMechanism,'credit-pc':PCInferenceMechanism,'credit-alm':PCALMMechanism,'active-inference':ActiveInferenceMechanism,'continual-learning':ContinualLearningMechanism,'memory-adaptation':MemoryLifecycleMechanism,'self-improvement':SelfImprovementMechanism,'distributed-representations':DistributedRepresentationsFigure,'sequence-attention-history':SequenceAttentionHistoryFigure,'resnet-lineage':ResnetLineageFigure,'gpt-scaling':GptScalingFigure,'multimodal-fusion':MultimodalFusionFigure,'multimodal-objectives':MultimodalObjectivesFigure,'temporal-modalities':TemporalModalitiesFigure,'diffusion':DiffusionFlowMechanism,'soft-latent-reasoning':SoftLatentMechanism,'vla':VLAFoundationMechanism,'action-sampling':ActionSamplingMechanism,'robot-evaluation':RobotEvaluationMechanism};
export const articleVisualTypes=new Set(Object.keys(extra));
export default function ArticleVisual({type}){
 const Component=extra[type];
 if(Component)return <Component/>;
 if(visualFamilies[type]==='concept'&&conceptFigureTypes[type])return <ConceptFigure type={type}/>;
 return <ArticleFigure type={type}/>;
}
