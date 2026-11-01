# cuda-mini

A tiny repo: a few CUDA kernels plus one Python script that compiles, verifies and benchmarks them.

| File | Purpose |
|---|---|
| `kernels.cu` | vector add, sum reduction (warp shuffle), transpose (naive vs tiled) |
| `bench.py` | builds with `nvcc`, checks correctness against PyTorch, reports GB/s |

## Run

```bash
pip install torch        # CUDA build of PyTorch
python bench.py          # requires nvcc (CUDA toolkit) and an NVIDIA GPU
```

No local GPU? Use a free Google Colab notebook: set **Runtime → Change runtime type → T4 GPU**, upload `kernels.cu` and `bench.py`, then run `!python bench.py`.

The script prints the median time and memory bandwidth (GB/s) of each kernel, next to the PyTorch equivalent where one exists.

## Results

Tesla T4 (Google Colab), 2^26 floats for vector add and reduction, 8192 x 8192 floats for transpose:

| Kernel | Time (ms) | Bandwidth (GB/s) |
|---|---:|---:|
| `vec_add` | 3.068 | 262.5 |
| `torch.add` | 3.395 | 237.2 |
| `reduce_sum` | 1.127 | 238.2 |
| `torch.sum` | 1.009 | 266.0 |
| `transpose_naive` | 7.090 | 75.7 |
| `transpose_tiled` | 2.781 | 193.0 |

The peak memory bandwidth of a T4 is 320 GB/s. Numbers vary slightly between runs and GPUs.

### What to notice

- **Memory-bound kernels:** vector add and sum reduction do almost no math, so their speed is set by memory bandwidth. Both reach roughly 75-82% of the T4's peak.
- **Naive vs tiled transpose:** the tiled version is about 2.5x faster. Staging a tile in shared memory makes both the global reads and the global writes coalesced, and the `+1` padding avoids shared-memory bank conflicts.

## Adding a new kernel

1. Write a `__global__` kernel in `kernels.cu`
2. Add a launcher function inside the `extern "C"` block
3. Call `lib.<name>(...)` from `bench.py` and compare the result with PyTorch
