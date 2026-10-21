## The representational shift
A discrete entity can enter a network as a one-hot vector $e_i\in\{0,1\}^{V}$, where $V$ is the number of identifiers. An embedding table $E\in\mathbb R^{d\times V}$ maps it to $v_i=Ee_i$. The category remains discrete; its internal representation is continuous and shared features can support generalization.

Hinton's 1986 work on distributed representations of concepts is the relevant historical thread for the family-relations example: learn internal representations from relational facts rather than hand-design one feature per concept. His author bibliography distinguishes this contribution from the related Rumelhart–Hinton–Williams backpropagation papers. These works should not be collapsed into one title. [1](#ref-hinton) [2](#ref-backprop)

## A relational learning problem
Conceptually, a model learns a mapping

$$
(\text{person}_1,\text{relation})\longmapsto p(\text{person}_2),\qquad
\mathcal L=-\log p_\theta(\text{person}_2\mid\text{person}_1,\text{relation}).
$$

The notation is a modern abstraction of the task, not an exact reconstruction of its original circuitry. Similar hidden patterns may capture regularities shared across family members. This is evidence that learned continuous representations can support structured relational tasks; it does not establish a complete symbolic theorem prover.

## From entities to word sequences
Bengio and collaborators' 2003 neural probabilistic language model jointly learned word representations and a probability function over sequences. Shared continuous features allow one observed sequence to inform predictions for related sequences. This is a direct conceptual ancestor of learned token embeddings and neural next-token modeling, although that model was not a Transformer. [3](#ref-bengio)

```python
context = embedding[token_ids]  # categorical lookup, differentiable parameters
features = nonlinear_network(context.flatten())
next_word_distribution = softmax(output(features))
```

## Influence, precisely stated
Distributed representation is a representational idea. Backpropagation is a credit-assignment method. Autoregressive language modeling is a probabilistic factorization. A Transformer is an architecture. Their later combination should not erase these distinctions or assign every component to one inventor.

## Reading exercise
Rebuild a small relation-completion task with a held-out set of facts. Compare training-fact memorization with generalization to unseen combinations. Control the split carefully: symmetry and inverse relations can leak answers across partitions. Then compare a learned lookup with fixed random embeddings at matched network size.
