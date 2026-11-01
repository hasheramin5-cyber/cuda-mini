"""bench.py - compile kernels.cu, verify against PyTorch, and benchmark.

Usage:  python bench.py
Requires: nvcc (CUDA toolkit) and a CUDA build of PyTorch.
"""
import ctypes
import os
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "kernels.cu")
LIB = os.path.join(HERE, "libkernels.so")


def build():
    """Compile kernels.cu into a shared library (skipped if already up to date)."""
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0]}{cap[1]}"
    cmd = ["nvcc", "-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC",
           SRC, "-o", LIB]
    print("Building:", " ".join(cmd))
    subprocess.check_call(cmd)


def bench(fn, iters=100, warmup=10):
    """Return the median time in milliseconds, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def ptr(t):
    """Raw device pointer of a CUDA tensor, for passing to the C launchers."""
    return ctypes.c_void_p(t.data_ptr())


def main():
    assert torch.cuda.is_available(), "No CUDA GPU found"
    build()
    lib = ctypes.CDLL(LIB)
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    # ---- vector add ----
    n = 1 << 26
    a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    run = lambda: lib.vec_add(ptr(a), ptr(b), ptr(c), ctypes.c_int(n))
    run()
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b), "vec_add result is wrong"
    t = bench(run)
    print(f"vec_add         {t:7.3f} ms  {3 * 4 * n / t / 1e6:7.1f} GB/s")
    t = bench(lambda: torch.add(a, b, out=c))
    print(f"  (torch.add)   {t:7.3f} ms  {3 * 4 * n / t / 1e6:7.1f} GB/s\n")

    # ---- sum reduction ----
    x = torch.randn(n, device="cuda")
    out = torch.zeros(1, device="cuda")
    run = lambda: lib.reduce_sum(ptr(x), ptr(out), ctypes.c_int(n))
    run()
    torch.cuda.synchronize()
    assert torch.allclose(out[0], x.sum(), rtol=1e-3, atol=1e-1), "reduce_sum result is wrong"
    t = bench(run)
    print(f"reduce_sum      {t:7.3f} ms  {4 * n / t / 1e6:7.1f} GB/s")
    t = bench(lambda: x.sum())
    print(f"  (torch.sum)   {t:7.3f} ms  {4 * n / t / 1e6:7.1f} GB/s\n")

    # ---- transpose: naive vs tiled ----
    rows, cols = 8192, 8192
    m = torch.randn(rows, cols, device="cuda")
    o = torch.empty(cols, rows, device="cuda")
    for name in ("transpose_naive", "transpose_tiled"):
        f = getattr(lib, name)
        run = lambda: f(ptr(m), ptr(o), ctypes.c_int(rows), ctypes.c_int(cols))
        o.zero_()
        run()
        torch.cuda.synchronize()
        assert torch.equal(o, m.t()), f"{name} result is wrong"
        t = bench(run)
        print(f"{name:15s} {t:7.3f} ms  {2 * 4 * rows * cols / t / 1e6:7.1f} GB/s")


if __name__ == "__main__":
    main()
