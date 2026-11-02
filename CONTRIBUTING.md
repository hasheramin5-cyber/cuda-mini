# Contributing

Thanks for helping! The repo is small on purpose, so a contribution is usually one new kernel.

## Add a kernel

1. Put the `__global__` kernel and an `extern "C"` launcher in a file under `kernels/`
   (new file or an existing one). Use plain `float*` arguments.
2. Add a `bench_<name>` function to `bench.py` that
   - checks the result against a PyTorch reference with `assert`,
   - returns `row(...)` entries (pass `nbytes=` for memory-bound kernels, `flops=` for compute-bound ones),
   - and is registered in the `GROUPS` dict.
3. Add a test to `tests/test_kernels.py` using **odd sizes** (not multiples of 32) to exercise bounds checks.
4. Describe the idea in `docs/how-it-works.md` if it teaches something new.

## Run things

```bash
python -m pytest -q tests      # correctness (needs a CUDA GPU)
python bench.py --save         # benchmark and write results/<gpu-name>.json
python tools/plot.py           # redraw docs/bandwidth_<gpu>.png
python tools/update_readme.py  # refresh the results table in README.md
```

## Share results from your GPU

Run `python bench.py --save` and open a pull request with the new `results/<gpu-name>.json`
(and the regenerated README table and plot). Results from different GPUs are very welcome.

## Style

- Python: `ruff check .` must pass.
- CUDA: 4-space indent, a short comment above each kernel saying what it demonstrates.
