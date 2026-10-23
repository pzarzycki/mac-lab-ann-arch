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

## Specify the test-time objective and update boundary
In Sun et al.'s image example, an auxiliary head predicts which rotation ($0°,90°,180°,270°$) was applied to an unlabeled image. A shared feature extractor $\theta_e$ feeds both the rotation head $\theta_s$ and the target object-classification head $\theta_m$. Training jointly learns the main and auxiliary tasks. At test time, their standard formulation minimizes rotation loss over **shared features** $\theta_e$ while keeping the heads fixed, then classifies the image using $(\theta_e^{\mathrm{adapted}},\theta_m)$. The known rotation provides a self-supervised label; the object's unknown class does not. [2](#ref-ttt)

Lower auxiliary loss need not mean better classification. In the paper's airplane example, image margins can reveal orientation without useful object information, and the authors report no classification improvement for that class in the examined comparison. This is an objective-alignment failure, not a proof against test-time adaptation generally. [2](#ref-ttt)

A reproducible sequence is: copy the base shared features; adapt on the current unlabeled image's rotation task; classify with the main head; discard that copy before the next independent sample. An online protocol instead carries the adapted features into later samples. State this reset choice before reporting a result.

## Follow one information lifecycle
The figure tracks the synthetic fact “code = IRIS” as a state inventory, without claiming to simulate a model. Injecting it fills context, an external record, a loaded and saved adapter, and test-time parameter state. **Reset context** removes only transient context. **Reset parameters** restores the base parameter state, leaving stored records and the saved adapter file. **Retrieve** copies an existing record into context; **reload** restores the saved adapter, not the test-time update. These operations are deliberately separate so a session reset need not imply a parameter reset.

A flag says that a route contains the information. It cannot certify faithful retrieval use or successful parameter encoding. The lifecycle is an analogy for persistence; real test-time rotation training is not a general fact-writing operation.

## Evaluate persistence and interference
Test retained behavior after adaptation and after reset. Account for memory write policies, stale documents, retrieval failures and adaptation compute. For streaming updates, include forgetting diagnostics such as those motivated by replay work. [4](#ref-replay)

## Limits
Retrieval can provide useful evidence without guaranteeing faithful use. Test-time updates can amplify noisy or unrepresentative inputs. More context, more stored facts and better weights should not be reported as one undifferentiated “self-learning” capability.
