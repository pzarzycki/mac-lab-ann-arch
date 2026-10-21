## Search the network as well as its weights
NeuroEvolution of Augmenting Topologies (NEAT), introduced by Kenneth Stanley and Risto Miikkulainen, searches over connection weights and network structure through population-based evolution. Candidates are evaluated, selected and varied across generations. A differentiable task loss is not required: a controller can receive a fitness score from an episode. This changes the optimization procedure; it does not turn an individual neuron's forward computation into a biological simulation. [1](#ref-neat)

A useful comparison asks three independent questions: what is encoded, what produces variation, and what supplies selection? Evolving a policy's weights differs from searching for an architecture and subsequently training it with gradients. Neither automatically includes synaptic plasticity during the individual's lifetime. A hybrid could combine these processes, but that additional learning rule and its evaluation cost must be specified.

## A genome records structure and ancestry
NEAT uses a direct encoding: node genes identify units, while connection genes record endpoints, weight, enabled status and an innovation identifier. Historical markings let crossover align genes descended from the same structural innovation, even when parental genomes have different lengths. An innovation identifier is bookkeeping, not a fitness score or a label for a semantic capability. [1](#ref-neat)

The adjacent figure lets you inspect one invented genome before and after splitting a connection. Its gene identifiers are illustrative. They show why a disabled connection can remain in the genotype while disappearing from the expressed network. The figure does not execute selection or reproduce a published run.

## Complexification changes the search space
Weight mutation changes numeric parameters; structural mutation can add a connection or split an existing connection with a new node. Starting with simple networks allows additional structure to accumulate through search. The original add-node construction disables the old edge and inserts two new edges, assigning the incoming edge weight one and retaining the old weight on the outgoing edge. An inserted nonlinear activation means this is not generally an exact function-preserving identity transformation. [2](#ref-neat-followup)

For an illustrative original edge with weight 0.8 and input 0.5, its contribution is 0.4. After inserting a tanh unit with those weights, the contribution becomes $0.8\tanh(0.5)\approx0.370$. This calculation explains the distinction between preserving an inherited weight and preserving an entire function; it is not the activation convention of every NEAT implementation.

## Speciation protects opportunities, not guaranteed progress
NEAT groups compatible genomes and shares fitness within species, giving new structures time to optimize before direct competition with established structures. Compatibility uses structural differences and matching-weight differences. Speciation therefore addresses premature loss of innovation; it does not guarantee that every species improves or that useful diversity persists indefinitely. [1](#ref-neat)

An experiment can compare the full method with fixed topology, removed historical alignment and removed speciation. Keep evaluation budgets comparable, record unsuccessful seeds, and count environment interactions as well as generations. Population size changes the cost of a generation, so generation count alone is not an efficiency metric.

## Where this route leads
The original evidence concerned particular control benchmarks, not modern language-model pretraining. Report wall-clock cost, parameter count, task success and robustness separately. For noisy environments, reevaluate candidates on fresh episode seeds so selection cannot simply exploit a lucky rollout.

Continue to [HyperNEAT](#/article/hyperneat) for an indirect encoding that generates connections from geometry. Compare with [backpropagation](#/article/backprop-biology) for derivative-based credit and [continual learning](#/article/continual-learning) for retention during sequential updates. Evolution across generations and adaptation within a deployed individual belong on different time axes.
