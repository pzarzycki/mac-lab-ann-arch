## The representational shift
A discrete entity can enter a network as a one-hot vector $e_i\in\{0,1\}^{V}$, where $V$ is the number of identifiers. An embedding table $E\in\mathbb R^{d\times V}$ maps it to $v_i=Ee_i$. The category remains discrete; its internal representation is continuous and shared features can support generalization.

Hinton's 1986 work on distributed representations of concepts is the relevant historical thread for the family-relations example: learn internal representations from relational facts rather than hand-design one feature per concept. His author bibliography distinguishes this contribution from the related Rumelhart–Hinton–Williams backpropagation papers. These works should not be collapsed into one title. [1](#ref-hinton) [2](#ref-backprop)

## What makes the code distributed?
Several coordinates participate in one entity, and a coordinate participates in several entities. A one-hot identity has one dedicated active coordinate; a distributed internal code reuses coordinates. Continuity alone is insufficient: a continuous lookup can still memorize identifiers without useful shared structure.

The figure uses a **synthetic**, deliberately interpretable code, not Hinton's learned weights. Ada has $[1,1,0]$, Ben $[1,0.8,0.2]$, Cora $[0,1,0]$ and Dan $[0,0,1]$. Coordinates stand for older generation, branch A and branch B. Ada and Ben each use multiple coordinates; branch A is shared across Ada, Ben and Cora. Interpretability is imposed here to expose the calculation, whereas real features must be learned.

## A relational learning problem
Conceptually, a model learns a mapping

$$
(\text{person}_1,\text{relation})\longmapsto p(\text{person}_2),\qquad
\mathcal L=-\log p_\theta(\text{person}_2\mid\text{person}_1,\text{relation}).
$$

The notation is a modern abstraction of the task, not an exact reconstruction of its original circuitry. Similar hidden patterns may capture regularities shared across family members. This is evidence that learned continuous representations can support structured relational tasks; it does not establish a complete symbolic theorem prover.

## A fact changes another prediction
Hold the codes fixed and ask a two-candidate **has-child** query, with outputs Cora or Dan. Start with rows $w_C=[0,0.2,0]$ and $w_D=[0,0,0.2]$ in $W\in\mathbb R^{2\times3}$; logits are $z=Wv$ and probabilities are $\operatorname{softmax}(z)$. These are illustrative candidate scores, not assertions that the code identifies every actual family relationship.

For Ben, $z=[0.16,0.04]$ gives $p(Cora)=0.530$. Train one fact, Ada has-child Cora. With cross-entropy, step size $\eta=1$ and target $y=[1,0]$,

$$
\Delta W=\eta(y-p_{Ada})v_{Ada}^{\top},\qquad p_{Ada}=\operatorname{softmax}([0.2,0]).
$$

The new rows are approximately $[0.450,0.650,0]$ and $[-0.450,-0.450,0.2]$. Ben's new logits are $[0.970,-0.770]$, so $p(Cora)=0.851$ even though Ben was not the training query. The change transfers because $v_{Ada}^{\top}v_{Ben}=1.8$. In a separate classifier whose only inputs are isolated one-hot IDs, that update changes Ada's output column and leaves Ben's unchanged. Shared hidden computation could introduce transfer even with one-hot inputs; it is the learned internal reuse that matters. This one-step demonstration shows a possible transfer route, not evidence that its new prediction is correct.

## What the historical experiment observed
Hinton's concepts paper used two isomorphic family trees, 24 person IDs and 12 relationship IDs. Separate six-unit internal groups encoded the input person and relation; a query could have several correct relatives. The reported person codes captured generation and family branch, with corresponding English/Italian people represented similarly on many units. These are observed regularities in that small relational task, not universal interpretable coordinates or a broad generalization benchmark. The paper also notes that its relation set did not pressure the person code to represent sex. [3](#ref-hinton-concepts)

## From entities to word sequences
Bengio and collaborators' 2003 neural probabilistic language model jointly learned word representations and a probability function over sequences. Shared continuous features allow one observed sequence to inform predictions for related sequences. This is a direct conceptual ancestor of learned token embeddings and neural next-token modeling, although that model was not a Transformer. [4](#ref-bengio)

```python
context = embedding[token_ids]  # categorical lookup, differentiable parameters
features = nonlinear_network(context.flatten())
next_word_distribution = softmax(output(features))
```

## Influence, precisely stated
Distributed representation is a representational idea. Backpropagation is a credit-assignment method. Autoregressive language modeling is a probabilistic factorization. A Transformer is an architecture. Their later combination should not erase these distinctions or assign every component to one inventor.

## Reading exercise
Rebuild a small relation-completion task with a held-out set of facts. Compare training-fact memorization with generalization to unseen combinations. Control the split carefully: symmetry and inverse relations can leak answers across partitions. Then compare a learned lookup with fixed random embeddings at matched network size.

## Inspect the modern interface
For explicit lookup and unembedding shapes, weight tying, contextual geometry and a small interactive example, open [Token embeddings & embedding space](#/article/token-embeddings). For reading internal states rather than input vectors, explore [Transformer observability & interpretability](#/article/transformer-observability).
