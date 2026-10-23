## Compute versus materialization
Dense attention evaluates pairwise query–key scores. The $n\times n$ score matrix is expensive to store, but an implementation need not materialize the entire matrix in high-bandwidth memory. FlashAttention tiles the computation and uses online softmax bookkeeping to reduce memory traffic while computing the same dense attention function, up to floating-point differences. It is **exact attention**, not a sparse approximation. [1](#ref-flash)

For a query tile $Q_b\in\mathbb R^{B_r\times d_k}$, load key/value tiles $K_c\in\mathbb R^{B_c\times d_k}$ and $V_c\in\mathbb R^{B_c\times d_v}$ into on-chip SRAM. The temporary score tile is $B_r\times B_c$. Conventional materialization writes full score and probability arrays, each $n\times n$, to HBM. Tiling avoids these full arrays; output and per-row normalization state remain necessary. [1](#ref-flash)

## Why tile softmaxes can merge
For one query row, retain a running maximum $m$, normalizer $\ell$, and unnormalized value accumulator $u$. Given a tile of allowed scaled scores $s_j$ and values $v_j$, update

$$
m'=\max(m,\max_j s_j),\qquad a=\exp(m-m'),
$$
$$
\ell'=a\ell+\sum_j\exp(s_j-m'),\qquad
u'=au+\sum_j\exp(s_j-m')v_j,\qquad O=u'/\ell'.
$$

Initialize $m=-\infty,\ell=0,u=0$, treating the initial rescaling term as zero. The factor $a$ expresses old contributions in the new maximum's scale; adding independently normalized tile outputs without this correction would be wrong. The final ratio equals softmax over all permitted scores. [1](#ref-flash)

The synthetic panel uses scores $[0,1,2,-1]$, values $[1,0],[0,2],[2,1],[-1,1]$, and two tiles. After tile 1, $m=1$, $\ell=1+e^{-1}$ and $u=[e^{-1},2]$. Tile 2 raises the maximum to 2 and rescales that state by $e^{-1}$ before adding its terms. The final output is approximately $[1.343,1.150]$, matching the dense calculation. Changing the third score recomputes both routes. All four edges remain present: this is an execution change, unlike the sparse mask below. The example isolates the forward arithmetic and does not simulate GPU timing.

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
