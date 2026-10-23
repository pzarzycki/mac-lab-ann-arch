## Search the network as well as its weights
NeuroEvolution of Augmenting Topologies (NEAT), introduced by Kenneth Stanley and Risto Miikkulainen, searches over connection weights and network structure through population-based evolution. Candidates are evaluated, selected and varied across generations. A differentiable task loss is not required: a controller can receive a fitness score from an episode. This changes the optimization procedure; it does not turn an individual neuron's forward computation into a biological simulation. [1](#ref-neat)

A useful comparison asks three independent questions: what is encoded, what produces variation, and what supplies selection? Evolving a policy's weights differs from searching for an architecture and subsequently training it with gradients. Neither automatically includes synaptic plasticity during the individual's lifetime. A hybrid could combine these processes, but that additional learning rule and its evaluation cost must be specified.

## A genome records structure and ancestry
NEAT uses a direct encoding: node genes identify units, while connection genes record endpoints, weight, enabled status and an innovation identifier. Historical markings let crossover align genes descended from the same structural innovation, even when parental genomes have different lengths. An innovation identifier is bookkeeping, not a fitness score or a label for a semantic capability. [1](#ref-neat)

The figure separates add-node mutation, aligned crossover, and an evaluated toy population. Its identifiers and outcomes are illustrative. A disabled connection remains in the genotype but disappears from the expressed computation; its dashed trace in the mutation view is explicitly a historical ghost. The population view executes a declared synthetic task and does not reproduce a published NEAT run.

## Align unequal parents before crossing them
The crossover view uses these connection innovations (endpoint, weight, enabled flag):

| ID | Parent A | Parent B | Correspondence |
| --- | --- | --- | --- |
| 1 | $1\to3$, .8, off | $1\to3$, 1, on | Matching |
| 2 | $2\to3$, −.2, on | $2\to3$, .2, on | Matching |
| 3 | $1\to4$, 1, on | $1\to4$, .8, on | Matching |
| 4 | $4\to3$, .8, on | Absent | Disjoint |
| 5 | Absent | $2\to4$, .4, on | Excess |
| 6 | Absent | $4\to3$, .5, on | Excess |

IDs 1–3 align regardless of list length. ID 4 lies inside the other parent's innovation range; IDs 5–6 extend beyond A's largest ID. If A is fitter, a possible child inherits its unmatched ID 4 and chooses each matching gene from either parent. If B is fitter, its unmatched IDs 5–6 are eligible instead. The controls choose one possible matching-weight outcome and execute the resulting child's forward pass. Hidden node 4 uses tanh; output node 3 is linear. The displayed child keeps ID 1 disabled as one possible outcome; the source's crossover rule allows disabled genes to be reenabled stochastically. [2](#ref-neat-followup)

## Complexification changes the search space
Weight mutation changes numeric parameters; structural mutation can add a connection or split an existing connection with a new node. Starting with simple networks allows additional structure to accumulate through search. The original add-node construction disables the old edge and inserts two new edges, assigning the incoming edge weight one and retaining the old weight on the outgoing edge. An inserted nonlinear activation means this is not generally an exact function-preserving identity transformation. [2](#ref-neat-followup)

For an illustrative original edge with weight 0.8 and input 0.5, its contribution is 0.4. After inserting a tanh unit with those weights, the contribution becomes $0.8\tanh(0.5)\approx0.370$. This calculation explains the distinction between preserving an inherited weight and preserving an entire function; it is not the activation convention of every NEAT implementation.

## Speciation protects opportunities, not guaranteed progress
NEAT groups compatible genomes and shares fitness within species, giving new structures time to optimize before direct competition with established structures. Compatibility uses structural differences and matching-weight differences. Speciation therefore addresses premature loss of innovation; it does not guarantee that every species improves or that useful diversity persists indefinitely. [1](#ref-neat)

Those same parents make compatibility concrete:

$$
\delta=\frac{c_1E}{N}+\frac{c_2D}{N}+c_3\bar W.
$$

Here $E$ and $D$ count excess and disjoint genes; $\bar W$ is the mean absolute weight difference across matching genes. With $E=2$, $D=1$, $\bar W=(.2+.4+.2)/3\approx.267$, coefficients $(c_1,c_2,c_3)=(1,1,.4)$ and the small-genome convention $N=1$, $\delta\approx3.107$. A compatibility threshold 3 separates these parents; threshold 3.2 groups them. In the general formula $N$ is the larger genome length, with the small-genome exception explicitly chosen here. Compatibility is compared with a species representative, rather than treating distance alone as a universal partition. [2](#ref-neat-followup)

The population view evaluates four fixed input pairs against $y^*=.8\tanh(x_1)-.2x_2$, with fitness $1/(1+\mathrm{MSE})$. A next-generation step retains the best genome, tries an add-node mutation, and creates matching-gene/weight variants before reevaluation. The target favors exactly the illustrated hidden nonlinearity, so solving it is deliberately easy. This deterministic toy demonstrates how task execution supplies selection; it omits species fitness sharing, noisy episodes, and the full original mutation schedule.

An experiment can compare the full method with fixed topology, removed historical alignment and removed speciation. Keep evaluation budgets comparable, record unsuccessful seeds, and count environment interactions as well as generations. Population size changes the cost of a generation, so generation count alone is not an efficiency metric.

## Where this route leads
The original evidence concerned particular control benchmarks, not modern language-model pretraining. Report wall-clock cost, parameter count, task success and robustness separately. For noisy environments, reevaluate candidates on fresh episode seeds so selection cannot simply exploit a lucky rollout.

Continue to [HyperNEAT](#/article/hyperneat) for an indirect encoding that generates connections from geometry. Compare with [backpropagation](#/article/backprop-biology) for derivative-based credit and [continual learning](#/article/continual-learning) for retention during sequential updates. Evolution across generations and adaptation within a deployed individual belong on different time axes.
