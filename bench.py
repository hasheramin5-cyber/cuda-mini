"""bench.py - build the CUDA kernels, verify them against PyTorch, and benchmark them.

Usage:
    python bench.py                    # run every benchmark group
    python bench.py --kernel matmul    # run one group
    python bench.py --save             # also write results/<gpu-name>.json

Requires: nvcc (CUDA toolkit) and a CUDA build of PyTorch.
"""
import argparse
import ctypes
import functools
import glob
import json
import os
import re
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "kernels")
LIB = os.path.join(HERE, "libkernels.so")
RESULTS_DIR = os.path.join(HERE, "results")

# Peak DRAM bandwidth (GB/s) of a few common GPUs; only used for the "% of peak" column.
PEAK_BANDWIDTH_GBS = {
    "Tesla T4": 320.0,
    "NVIDIA L4": 300.0,
    "Tesla V100-SXM2-16GB": 900.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
}

i32 = ctypes.c_int


def build():
    """Compile every kernels/*.cu into one shared library (skipped if up to date)."""
    sources = sorted(glob.glob(os.path.join(SRC_DIR, "*.cu")))
    newest = max(os.path.getmtime(s) for s in sources)
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= newest:
        return
    major, minor = torch.cuda.get_device_capability()
    cmd = ["nvcc", "-O3", f"-arch=sm_{major}{minor}", "-shared", "-Xcompiler", "-fPIC",
           *sources, "-o", LIB]
    print("Building:", " ".join(cmd))
    subprocess.check_call(cmd)


def load_lib():
    """Build (if needed) and load the kernel library."""
    build()
    return ctypes.CDLL(LIB)


def ptr(t):
    """Raw device pointer of a CUDA tensor, for passing to the C launchers."""
    return ctypes.c_void_p(t.data_ptr())


def time_kernel(fn, iters=100, warmup=10):
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


def row(name, ms, nbytes=None, flops=None):
    """One result row. Bandwidth is GB/s, throughput is GFLOP/s."""
    return {
        "kernel": name,
        "time_ms": round(ms, 3),
        "bandwidth_gbs": None if nbytes is None else round(nbytes / ms / 1e6, 1),
        "gflops": None if flops is None else round(flops / ms / 1e6, 1),
    }


# ---------------------------------------------------------------- benchmark groups
def bench_vec_add(lib, iters):
    n = 1 << 26
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    run = functools.partial(lib.vec_add, ptr(a), ptr(b), ptr(c), i32(n))
    run()
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b), "vec_add result is wrong"
    nbytes = 3 * 4 * n  # read a, read b, write c
    return [
        row("vec_add", time_kernel(run, iters), nbytes=nbytes),
        row("torch.add", time_kernel(lambda: torch.add(a, b, out=c), iters), nbytes=nbytes),
    ]


def bench_reduce(lib, iters):
    n = 1 << 26
    x = torch.randn(n, device="cuda")
    out = torch.zeros(1, device="cuda")
    run = functools.partial(lib.reduce_sum, ptr(x), ptr(out), i32(n))
    run()
    torch.cuda.synchronize()
    assert torch.allclose(out[0], x.sum(), rtol=1e-3, atol=1e-1), "reduce_sum result is wrong"
    nbytes = 4 * n
    return [
        row("reduce_sum", time_kernel(run, iters), nbytes=nbytes),
        row("torch.sum", time_kernel(lambda: x.sum(), iters), nbytes=nbytes),
    ]


def bench_transpose(lib, iters):
    rows, cols = 8192, 8192
    m = torch.randn(rows, cols, device="cuda")
    o = torch.empty(cols, rows, device="cuda")
    nbytes = 2 * 4 * rows * cols  # read once, write once
    results = []
    for name in ("transpose_naive", "transpose_tiled"):
        run = functools.partial(getattr(lib, name), ptr(m), ptr(o), i32(rows), i32(cols))
        o.zero_()
        run()
        torch.cuda.synchronize()
        assert torch.equal(o, m.t()), f"{name} result is wrong"
        results.append(row(name, time_kernel(run, iters), nbytes=nbytes))
    return results


def bench_softmax(lib, iters):
    rows, cols = 16384, 1024
    x = torch.randn(rows, cols, device="cuda")
    y = torch.empty_like(x)
    ref = torch.softmax(x, dim=1)
    nbytes = 2 * 4 * rows * cols  # ideal traffic: one read, one write
    results = []
    for name in ("softmax_naive", "softmax_warp"):
        run = functools.partial(getattr(lib, name), ptr(x), ptr(y), i32(rows), i32(cols))
        y.zero_()
        run()
        torch.cuda.synchronize()
        assert torch.allclose(y, ref, rtol=1e-4, atol=1e-6), f"{name} result is wrong"
        results.append(row(name, time_kernel(run, iters), nbytes=nbytes))
    results.append(row("torch.softmax",
                       time_kernel(lambda: torch.softmax(x, dim=1), iters), nbytes=nbytes))
    return results


def bench_matmul(lib, iters):
    torch.backends.cuda.matmul.allow_tf32 = False  # compare plain float32 with float32
    m = n = k = 2048
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    c = torch.empty(m, n, device="cuda")
    ref = a @ b
    flops = 2 * m * n * k
    iters = max(5, iters // 5)  # matmul is much slower per call
    results = []
    for name in ("matmul_naive", "matmul_tiled"):
        run = functools.partial(getattr(lib, name), ptr(a), ptr(b), ptr(c),
                                i32(m), i32(n), i32(k))
        c.zero_()
        run()
        torch.cuda.synchronize()
        assert torch.allclose(c, ref, rtol=1e-3, atol=1e-2), f"{name} result is wrong"
        results.append(row(name, time_kernel(run, iters), flops=flops))
    results.append(row("torch.matmul (cuBLAS)",
                       time_kernel(lambda: torch.matmul(a, b, out=c), iters), flops=flops))
    return results


GROUPS = {
    "vec_add": bench_vec_add,
    "reduce": bench_reduce,
    "transpose": bench_transpose,
    "softmax": bench_softmax,
    "matmul": bench_matmul,
}


# ---------------------------------------------------------------- output
def print_row(r):
    line = f"{r['kernel']:<24}{r['time_ms']:>9.3f} ms"
    if r["bandwidth_gbs"] is not None:
        line += f"{r['bandwidth_gbs']:>9.1f} GB/s"
    if r["gflops"] is not None:
        line += f"{r['gflops']:>10.1f} GFLOP/s"
    print(line)


def save_results(gpu, peak, results):
    """Write results/<gpu>.json, merging with any rows that are already there."""
    major, minor = torch.cuda.get_device_capability()
    slug = re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_")
    path = os.path.join(RESULTS_DIR, f"{slug}.json")
    merged = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            merged = {r["kernel"]: r for r in json.load(f).get("results", [])}
    for r in results:
        merged[r["kernel"]] = r
    data = {
        "gpu": gpu,
        "compute_capability": f"{major}.{minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "peak_bandwidth_gbs": peak,
        "results": list(merged.values()),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"\nSaved {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the cuda-mini kernels.")
    parser.add_argument("--kernel", choices=["all", *GROUPS], default="all",
                        help="benchmark group to run (default: all)")
    parser.add_argument("--iters", type=int, default=100, help="timed iterations per kernel")
    parser.add_argument("--save", action="store_true", help="write results/<gpu-name>.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. On Colab: Runtime > Change runtime type > GPU.")
    lib = load_lib()
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}")

    results = []
    for name in (GROUPS if args.kernel == "all" else [args.kernel]):
        print(f"\n== {name} ==")
        rows = GROUPS[name](lib, args.iters)
        for r in rows:
            print_row(r)
        results += rows

    if args.save:
        peak = next((v for k, v in PEAK_BANDWIDTH_GBS.items() if k in gpu), None)
        save_results(gpu, peak, results)


if __name__ == "__main__":
    main()
