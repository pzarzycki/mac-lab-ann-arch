## Encode a pattern instead of listing every connection
HyperNEAT extends neuroevolution through an indirect encoding: a compact pattern-producing network generates the weights of another network. The generated network is the **substrate**; the generator is a **compositional pattern-producing network**, or CPPN. Keeping these two networks separate is essential. A small generator can specify many substrate connections without storing one independently evolved gene for every weight. [1](#ref-hyperneat)

This is representational compression, not a promise of low execution cost. Once generated, a large dense substrate can still require many operations and substantial memory. Ask separately how expensive it is to search for a generator, generate a substrate and execute that substrate.

## Coordinates become queries
Assign each substrate neuron a location. For a two-dimensional layout, a possible connection from source $(x_s,y_s)$ to target $(x_t,y_t)$ becomes a four-coordinate query:

$$
w_{st}=f_\phi(x_s,y_s,x_t,y_t).
$$

Here $f_\phi$ is the CPPN, and its output determines a candidate weight under the chosen scaling and connection-expression convention. Extra inputs such as distance or bias, and additional outputs for different connection sets, are implementation choices. The coordinates describe network geometry; they are not observations arriving from the task environment. [1](#ref-hyperneat)

The interactive diagram uses two rows of three neurons. Select a source and target to inspect the corresponding coordinate query. Switching the symmetry control changes the permitted generator terms. The figure composes a small explicit CPPN and evaluates a synthetic substrate task; its bounded coefficient search is distinct from full HyperNEAT, which also evolves CPPN topology.

## Why a CPPN can express regularity
CPPNs compose functions with different spatial biases. Symmetric and periodic functions can create repeated motifs, while other compositions introduce variation. Their developmental analogy concerns the genotype-to-phenotype mapping rather than a literal simulation of cell division. A coordinate-to-pattern function can describe a coherent large object through shared rules instead of independent values. [2](#ref-cppn)

The figure makes the composition explicit:

$$
g=\exp[-2(x_s-x_t)^2],\qquad p=\sin[\pi(x_s+x_t)/2],\qquad l=x_t,\qquad w=ag+bp+cl-.3.
$$

These nodes form a small hand-designed CPPN; only its coefficients are varied in the toy search. The Gaussian branch supplies locality and reflection symmetry; sine supplies a periodic branch and the linear branch depends on target location. With $b=c=0$, simultaneously reflecting both coordinates preserves every weight. With nonzero $b,c$, this invariance can break. The fixed y-coordinates identify source and target rows; this particular generator's active branches use only x. The inset displays every intermediate value for the selected pair.

Substrate edges preserve sign even when selected: solid means positive, dashed means negative, width reflects magnitude, and an orange halo marks the selected query. The synchronized matrix supplies exact signed values, including tiny weights. Neither color nor minimum stroke width alone is a quantitative weight scale.

A geometry-sensitive encoding can be a poor fit when nearby coordinates do not imply related roles. Choosing coordinates and permitted layer connections already introduces design knowledge. Compactness is useful only if its biases match useful solutions.

## Evolution acts on the generator
HyperNEAT uses NEAT to evolve CPPNs, including their weights and topology. Each candidate CPPN generates a substrate, whose task performance supplies the fitness signal. Thus an evolutionary change to one generator parameter may coherently alter many substrate connections. In direct NEAT, connection genes instead describe the task network more directly. [3](#ref-neat)

In the figure's toy fitness loop, four coefficient candidates generate a three-by-three weight matrix $W$. The substrate then executes $y_j=\sum_i x_iw_{ij}$ for input $x_i$ equal to source coordinate and desired output $y_j^*$ equal to target coordinate. Candidate fitness is $1/(1+\frac13\sum_j(y_j-y_j^*)^2)$. Retain the best generator and compare local coefficient variants in the next generation. The evaluation is deterministic and has a fixed three-by-three grid; it is not a controller rollout, does not demonstrate generalization, and does not evolve topology. The fitness panel exposes scores before selection rather than assigning an appealing-looking pattern a fitness by hand.

For the mechanism diagram, the relevant distinction is between a query-time computation that generates weights and the later execution of those weights on task inputs. Neither should be confused with gradient-based architecture optimization or online plasticity. A hybrid that learns after generation must state that extra update rule and include its cost.

## Scale the representation, then test the behavior
A coordinate function can be queried on a finer grid without increasing its own parameter count. The five-by-five resolution control holds the selected generator coefficients fixed and queries more coordinates; it does not select a new generator for the finer grid. The displayed substrate output and error are recomputed at that resolution, while evolutionary selection continues to use the declared three-by-three evaluation grid. This permits re-expression at another substrate resolution, but successful task transfer still depends on input/output semantics, scaling and evaluation. The original HyperNEAT work demonstrated particular visual-discrimination and food-gathering settings; that evidence does not establish automatic generalization to arbitrary larger networks. [1](#ref-hyperneat)

Compare direct and indirect encodings at matched environment-evaluation budgets. Record generator size, expressed connections, substrate runtime and variation across seeds. Deliberately scramble coordinates as an ablation to test whether geometry supplies the claimed advantage. Return to [NEAT](#/article/neat) for historical alignment and speciation, or compare [multimodal interfaces](#/article/multimodal-fusion) to see another setting where the choice of representation constrains what a learner can discover.
