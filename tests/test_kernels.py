"""Correctness tests. Odd sizes are used on purpose to exercise the bounds checks.

Run with:  python -m pytest -q tests      (needs a CUDA GPU; skipped otherwise)
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bench

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

i32 = bench.i32
ptr = bench.ptr


@pytest.fixture(scope="module")
def lib():
    return bench.load_lib()


def sync():
    torch.cuda.synchronize()


def test_vec_add(lib):
    n = 1_000_003
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    lib.vec_add(ptr(a), ptr(b), ptr(c), i32(n))
    sync()
    assert torch.allclose(c, a + b)


@pytest.mark.parametrize("n", [1, 31, 1000, 1_000_003])
def test_reduce_sum(lib, n):
    x = torch.randn(n, device="cuda")
    out = torch.zeros(1, device="cuda")
    lib.reduce_sum(ptr(x), ptr(out), i32(n))
    sync()
    assert torch.allclose(out[0], x.sum(), rtol=1e-3, atol=1e-1)


@pytest.mark.parametrize("name", ["transpose_naive", "transpose_tiled"])
@pytest.mark.parametrize("shape", [(1000, 777), (33, 65), (64, 64)])
def test_transpose(lib, name, shape):
    rows, cols = shape
    m = torch.randn(rows, cols, device="cuda")
    o = torch.zeros(cols, rows, device="cuda")
    getattr(lib, name)(ptr(m), ptr(o), i32(rows), i32(cols))
    sync()
    assert torch.equal(o, m.t())


@pytest.mark.parametrize("name", ["softmax_naive", "softmax_warp"])
@pytest.mark.parametrize("shape", [(37, 101), (64, 1024), (5, 7)])
def test_softmax(lib, name, shape):
    rows, cols = shape
    x = torch.randn(rows, cols, device="cuda")
    y = torch.zeros_like(x)
    getattr(lib, name)(ptr(x), ptr(y), i32(rows), i32(cols))
    sync()
    assert torch.allclose(y, torch.softmax(x, dim=1), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", ["matmul_naive", "matmul_tiled"])
@pytest.mark.parametrize("mnk", [(65, 33, 47), (128, 128, 128), (31, 70, 5)])
def test_matmul(lib, name, mnk):
    m, n, k = mnk
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    c = torch.zeros(m, n, device="cuda")
    getattr(lib, name)(ptr(a), ptr(b), ptr(c), i32(m), i32(n), i32(k))
    sync()
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.allclose(c, a @ b, rtol=1e-3, atol=1e-3)
