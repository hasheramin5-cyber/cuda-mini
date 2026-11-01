# cuda-mini

A tiny repo: a few CUDA kernels plus one Python script that compiles, verifies and benchmarks them.

| File | Purpose |
|---|---|
| `kernels.cu` | vector add, sum reduction (warp shuffle), transpose (naive vs tiled) |
| `bench.py` | builds with `nvcc`, checks correctness against PyTorch, reports GB/s |

## Run

```bash
pip install torch        # CUDA build of PyTorch
python bench.py          # requires nvcc (CUDA toolkit)
```

The script prints the time and memory bandwidth (GB/s) of each kernel, next to the PyTorch equivalent where one exists.
Compare `transpose_naive` with `transpose_tiled` to see the effect of shared memory and coalesced access.

## Adding a new kernel

1. Write a `__global__` kernel in `kernels.cu`
2. Add a launcher function inside the `extern "C"` block
3. Call `lib.<name>(...)` from `bench.py` and compare the result with PyTorch
