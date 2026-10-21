## Why cache keys and values?
In a causal decoder, earlier positions cannot read later positions. With fixed weights and the same positional convention, their layerwise keys and values can therefore be reused when a new token arrives. Queries for the new position still read the retained context. A KV cache is intermediate inference state, not new model parameters. [1](#ref-mqa)

## The memory calculation
For a conventional full-context cache, assume equal key/value dimensions and one stored representation per KV head:

$$
M_{\mathrm{KV}}=2BLTH_{\mathrm{KV}}d_hs\quad\text{bytes}.
$$

$B$ is batch size, $L$ layer count, $T$ cached positions, $H_{\mathrm{KV}}$ KV heads, $d_h$ head dimension and $s$ bytes per scalar. The factor two counts keys and values. The calculator reports GiB using $2^{30}$ bytes. It excludes model weights, allocator overhead, temporary buffers, padding, quantization metadata and any architecture-specific extra states.

With $B=1,L=32,T=8192,H_{\mathrm{KV}}=8,d_h=128,s=2$, the result is **1 GiB**. Increasing KV heads to 32 makes it 4 GiB under exactly these assumptions.

## Share the KV representation
- **MHA:** separate KV heads for each query head.
- **MQA:** all query heads share one KV head.
- **GQA:** groups of query heads share an intermediate number of KV heads.

GQA changes the architecture's representation, rather than merely changing an allocator. The original work also studied converting multi-head checkpoints through additional training. [2](#ref-gqa)

## Orthogonal improvements
PagedAttention organizes cached tensors in blocks to reduce fragmentation and permit sharing; it does not replace GQA's mathematical projection. Multi-head latent attention (MLA), used in DeepSeek models, stores a compressed latent representation with an architecture-specific reconstruction and positional treatment. The simple formula above must not be reused as an exact MLA accounting model. [3](#ref-paged) [4](#ref-deepseek)

```python
cache_bytes = 2 * batch * layers * context * kv_heads * head_dim * bytes_per_scalar
cache_gib = cache_bytes / 2**30
```

## What a smaller cache does not prove
End-to-end latency also depends on memory bandwidth, kernel layout, scheduling, tensor parallelism and workload. Report time to first token separately from per-token decode latency and throughput. A cache estimate is not a GPU-memory fit guarantee.
