## Four meanings of adaptation
| Mechanism | What changes? | What survives a reset? |
| --- | --- | --- |
| In-context conditioning | Activations and cache | Nothing unless context is saved |
| Retrieval memory | External records and index | Stored records |
| Adapter training | A selected parameter set | Saved adapter weights |
| Test-time training | Parameters updated using test-time objectives | Depends on reset policy |

These mechanisms can coexist, but they answer different questions. A system can remember facts without changing its language-model weights.

## Retrieve evidence
Retrieval-augmented generation conditions generation on retrieved documents. A simplified mixture is

$$
p(y\mid x)\approx\sum_{z\in\mathcal R(x)}p_\eta(z\mid x)\,p_\theta(y\mid x,z).
$$

$\mathcal R(x)$ is a retrieved set, $\eta$ retriever parameters and $\theta$ generator parameters. Sequence-level and token-level marginalization differ in the original RAG work. A production prompt that simply concatenates top documents is a related design rather than this exact probabilistic marginalization. [1](#ref-rag)

## Learn at test time
Test-time training updates parameters from an available self-supervised objective before predicting on a shifted input distribution. Sun and collaborators demonstrate the idea on visual tasks. Whether a label-free objective improves the target task depends on its relation to that task and the shift. [2](#ref-ttt)

```python
adapted = copy_parameters(base_model)
for _ in range(adaptation_steps):
    adapted = update(adapted, self_supervised_loss(unlabeled_input))
prediction = adapted(unlabeled_input)
```

A copied/reset model and a continuously updated model create different evaluation protocols. Explicitly state which is used. Low-rank adaptation can restrict the update space but does not solve objective selection. [3](#ref-lora)

## Evaluate persistence and interference
Test retained behavior after adaptation and after reset. Account for memory write policies, stale documents, retrieval failures and adaptation compute. For streaming updates, include forgetting diagnostics such as those motivated by replay work. [4](#ref-replay)

## Limits
Retrieval can provide useful evidence without guaranteeing faithful use. Test-time updates can amplify noisy or unrepresentative inputs. More context, more stored facts and better weights should not be reported as one undifferentiated “self-learning” capability.
