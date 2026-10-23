## Time is an information bottleneck
A video encoder sees sampled frames or clips, not the full continuous world. Audio frontends also discretize time. Frame rate, crop strategy and receptive field determine which events survive. A later language layer cannot recover motion that was never observed.

For a clip duration $T$ seconds sampled at $f$ frames per second with $p$ visual tokens per frame, an uncompressed visual sequence has approximately $n_v=Tfp$ tokens. With an illustrative $T=10,f=2,p=196$, this is 3,920 tokens before text and any compression. This arithmetic is not a claim about a particular released model.

## Temporal position and synchronization
Spatial patch indices and temporal indices serve different roles. Interleaving frames and text requires masks that preserve intended access. Audio/video synchronization can be represented by timestamps or aligned feature windows, but positional tags do not correct an inaccurate clock or dropped observations.

## A timestamp changes access, not observation
The figure uses a synthetic four-second clip and an event visible only on $[1.15,1.35)$ seconds. At 2 fps with phase zero, frames are sampled at $0,0.5,1.0,1.5,\ldots$: the event is missed. A half-interval phase moves the sample grid to $0.25,0.75,1.25,\ldots$, retaining the event at 1.25 s without increasing frame count. In either case, eight frames times 196 patches gives 1,568 visual tokens. Equal token budgets can therefore retain different evidence.

For audio, let a feature centered on physical time 1.25 s be stamped 1.55 s because its clock has offset $\Delta=+0.30$ s: $t_{reported}=t_{physical}+\Delta$. A video query at 1.25 s with the illustrative access rule $|t_{audio}-1.25|\leq0.26$ s excludes that feature. Subtracting the known offset admits it. This correction changes the alignment of retained evidence; it cannot reconstruct a video frame that was never sampled. The figure uses 0.20 s audio windows every 0.50 s for readability, not a released speech frontend.

For an online causal system, one could additionally require each window's **end** timestamp to be no later than the query time; an offline clip model may allow the whole observed clip. That access policy and any latency must be stated independently of positional encoding. See [modality interfaces](#/article/multimodal-fusion) for how retained features enter self-attention or external memory.

## Two concrete reference points
Whisper is an encoder–decoder speech model trained on large-scale weak supervision with task and language conditioning. It is a speech-system reference, not an example of every “native audio” LLM. Flamingo provides a visual-language architecture that can ingest images and videos through its visual interface. [1](#ref-whisper) [2](#ref-flamingo)

Whisper's concrete frontend resamples audio to 16 kHz and computes 80-channel log-Mel features from 25 ms windows with a 10 ms stride. At that sampling rate, each window spans 400 samples and successive starts are 160 samples apart, so adjacent windows overlap by 240 samples. These are frontend windows, not one-to-one language tokens; the model's convolutional stem further processes them before its Transformer encoder. Task/language tokens condition the decoder. This provides an operational audio example while preserving the distinction from the much coarser synthetic timeline. [1](#ref-whisper)

```python
frames, times = sample_video(video, frames_per_second=2)
features = vision_encoder(frames)
tokens = add_temporal_information(features, times)
```

The pseudocode leaves the temporal representation abstract because the choice is architecture-specific. ViT-style spatial patching alone does not define video dynamics. [3](#ref-vit)

## What to evaluate
Use tasks that distinguish event order, duration, state change and cross-modal correspondence. Shuffle frames or shift audio as controlled ablations. A strong clip caption can coexist with poor temporal localization. Report sampling density, input duration and whether the test answer is recoverable from one static frame.

## Bridge to action
For robotics, latency makes the observed state older than the current physical state. A language model's plausible description cannot substitute for a timestamped observation and a control loop. Continue to [closed-loop evaluation](#/article/robot-evaluation).
