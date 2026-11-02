# How the kernels work

This page explains the ideas behind each kernel in `kernels/`. Numbers live in the README; this page is about *why*.

## Memory-bound vs compute-bound

A GPU can do far more arithmetic than it can feed from memory. A kernel is **memory-bound** if its speed is set by how fast it moves bytes (vector add, reduction, transpose, softmax) and **compute-bound** if it is set by arithmetic (a good matrix multiply).

For memory-bound kernels `bench.py` reports *effective bandwidth*: bytes the kernel must read and write, divided by its time. The closer it gets to the GPU's peak bandwidth, the less there is left to gain.

## Coalescing (transpose)

Threads in a warp (32 threads) execute together. If they read 32 neighbouring floats, the hardware serves them with a few wide memory transactions. If they read addresses far apart, it needs many.

```
coalesced:      thread 0 1 2 3 ...      ->  addresses 0 1 2 3 ...      (few transactions)
uncoalesced:    thread 0 1 2 3 ...      ->  addresses 0 N 2N 3N ...    (one per thread)
```

A plain transpose reads rows (coalesced) but writes columns (uncoalesced). `transpose_tiled` reads a tile into shared memory with coalesced loads, then writes it out with coalesced stores.

## Shared memory and bank conflicts (transpose, matmul)

Shared memory is a small, fast, per-block scratchpad split into 32 banks. If several threads of a warp hit the same bank at the same time, the accesses are serialized. In the transpose tile, reading a column of a `[32][32]` array puts all 32 threads in one bank. Declaring it `[32][33]` shifts each row by one bank, which removes the conflict.

## Warp shuffles (reduction, softmax)

Threads in a warp can exchange register values directly with `__shfl_down_sync` / `__shfl_xor_sync`, with no shared memory and no `__syncthreads()`. `reduce_sum` and `softmax_warp` use them to combine 32 partial results in five steps.

`reduce_sum` also uses a **grid-stride loop**: a fixed number of blocks loop over the whole array, so the same launch works for any size and each thread accumulates many elements before the (comparatively expensive) cross-thread step.

## Softmax: one thread vs one warp per row

`softmax_naive` gives each row to one thread, which walks the row three times (max, sum, normalize). Neighbouring threads read addresses `cols` floats apart, so loads are uncoalesced.

`softmax_warp` gives each row to a full warp: the 32 lanes read consecutive elements together and combine their partial max and sum with shuffles. It subtracts the row maximum before `exp`, which is what keeps softmax numerically stable.

## Matmul: data reuse

In `matmul_naive`, every output element reads one row of A and one column of B from global memory, so each input value is fetched many times over. In `matmul_tiled`, a block loads a 32 x 32 tile of A and of B into shared memory once, and all 32 x 32 threads reuse those values for 32 multiply-adds each. Global-memory traffic drops by roughly the tile size.

Matmul is reported in GFLOP/s (2 x M x N x K floating-point operations) and compared against cuBLAS through `torch.matmul`, which is far more heavily optimized (register tiling, vectorized loads, tensor cores on supported GPUs).

## Where to go next

- Register tiling (each thread computes a small block of outputs) for matmul
- `float4` vectorized loads for vec_add and softmax
- Half precision and tensor cores (`wmma`)
- The NVIDIA CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
