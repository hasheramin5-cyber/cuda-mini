// softmax.cu - row-wise softmax: a naive version and a warp-per-row version
#include <cuda_runtime.h>
#include <math.h>

// 1) Naive: one thread handles a whole row (three passes over global memory).
//    Neighbouring threads read addresses `cols` floats apart, so loads are
//    uncoalesced.
__global__ void softmax_naive_k(const float* in, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* x = in + (size_t)r * cols;
    float* y = out + (size_t)r * cols;

    float m = -INFINITY;
    for (int j = 0; j < cols; ++j) m = fmaxf(m, x[j]);
    float s = 0.f;
    for (int j = 0; j < cols; ++j) s += expf(x[j] - m);
    for (int j = 0; j < cols; ++j) y[j] = expf(x[j] - m) / s;
}

// 2) Warp per row: the 32 lanes of a warp walk the row together (coalesced
//    loads) and combine their partial max / sum with warp shuffles.
__global__ void softmax_warp_k(const float* in, float* out, int rows, int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;  // whole warps exit together, so shuffles stay safe
    const float* x = in + (size_t)row * cols;
    float* y = out + (size_t)row * cols;

    float m = -INFINITY;
    for (int j = lane; j < cols; j += 32) m = fmaxf(m, x[j]);
    for (int o = 16; o > 0; o >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));

    float s = 0.f;
    for (int j = lane; j < cols; j += 32) s += expf(x[j] - m);
    for (int o = 16; o > 0; o >>= 1)
        s += __shfl_xor_sync(0xffffffff, s, o);

    float inv = 1.f / s;
    for (int j = lane; j < cols; j += 32) y[j] = expf(x[j] - m) * inv;
}

// ---- C launchers ----
extern "C" {

void softmax_naive(const float* in, float* out, int rows, int cols) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    softmax_naive_k<<<blocks, threads>>>(in, out, rows, cols);
}

void softmax_warp(const float* in, float* out, int rows, int cols) {
    int threads = 256;                   // 8 warps = 8 rows per block
    int rows_per_block = threads / 32;
    int blocks = (rows + rows_per_block - 1) / rows_per_block;
    softmax_warp_k<<<blocks, threads>>>(in, out, rows, cols);
}

}  // extern "C"
