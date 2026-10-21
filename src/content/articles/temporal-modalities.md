## Time is an information bottleneck
A video encoder sees sampled frames or clips, not the full continuous world. Audio frontends also discretize time. Frame rate, crop strategy and receptive field determine which events survive. A later language layer cannot recover motion that was never observed.

For a clip duration $T$ seconds sampled at $f$ frames per second with $p$ visual tokens per frame, an uncompressed visual sequence has approximately $n_v=Tfp$ tokens. With an illustrative $T=10,f=2,p=196$, this is 3,920 tokens before text and any compression. This arithmetic is not a claim about a particular released model.

## Temporal position and synchronization
Spatial patch indices and temporal indices serve different roles. Interleaving frames and text requires masks that preserve intended access. Audio/video synchronization can be represented by timestamps or aligned feature windows, but positional tags do not correct an inaccurate clock or dropped observations.

## Two concrete reference points
Whisper is an encoder–decoder speech model trained on large-scale weak supervision with task and language conditioning. It is a speech-system reference, not an example of every “native audio” LLM. Flamingo provides a visual-language architecture that can ingest images and videos through its visual interface. [1](#ref-whisper) [2](#ref-flamingo)

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
