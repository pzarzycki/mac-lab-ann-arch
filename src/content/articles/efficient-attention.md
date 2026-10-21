## Compute versus materialization
Dense attention evaluates pairwise query–key scores. The $n\times n$ score matrix is expensive to store, but an implementation need not materialize the entire matrix in high-bandwidth memory. FlashAttention tiles the computation and uses online softmax bookkeeping to reduce memory traffic while computing the same dense attention function, up to floating-point differences. It is **exact attention**, not a sparse approximation. [1](#ref-flash)

## A sparse graph changes the function
Windowed attention restricts each token to a neighborhood. Additional global tokens or other structured connections can restore longer-range paths. Longformer is a primary example of combining local and global connectivity. A decoder adaptation must still enforce causality; the original pattern should not be copied without considering its prediction objective. [2](#ref-longformer)

For a causal window of width $w$, the number of allowed edges is

$$
E(n,w)=\sum_{i=0}^{n-1}\min(i+1,w).
$$

For dense causal attention it is $n(n+1)/2$. These counts omit head and feature dimensions. They are structural work proxies, not a measured FLOP count or runtime prediction.

## Inspect the mask
The lab shows deterministic connectivity. Narrow the window and look at the last row: older tokens disappear from a single layer's direct receptive field. Multiple layers may propagate information farther, but the reachable graph and effective retrieval performance are separate questions.

## Implementation questions
```python
allowed = (source_position <= destination_position)
if window is not None:
    allowed &= (destination_position - source_position < window)
```

A sparse mask passed to a dense kernel may preserve dense compute. Efficient sparsity requires a layout and kernel that exploit its structure. Conversely, FlashAttention can improve dense execution without changing the permitted edges.

## Evidence and limits
Published speedups are tied to hardware, sequence length, precision and baseline. This atlas does not translate one paper's GPU result into a promise for a different device. Compare kernel time, end-to-end time, memory and task quality together.
