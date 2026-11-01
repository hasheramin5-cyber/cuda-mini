// kernels.cu - three small CUDA kernels for learning and benchmarking
// Build: nvcc -O3 -shared -Xcompiler -fPIC kernels.cu -o libkernels.so
#include <cuda_runtime.h>

#define TILE 32
#define ROWS_PER_BLOCK 8

// 1) Vector add: one thread per element
__global__ void vec_add_k(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// 2) Sum reduction: grid-stride loop + warp shuffle + atomicAdd
//    (block size must be a multiple of 32, e.g. 256)
__global__ void reduce_sum_k(const float* x, float* out, int n) {
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        v += x[i];

    // Reduce within each warp using register shuffles
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);

    // Collect one partial sum per warp in shared memory
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    // First warp reduces the partial sums, then adds to the global result
    if (warp == 0) {
        v = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffff, v, o);
        if (lane == 0) atomicAdd(out, v);
    }
}

// 3a) Transpose (naive): global writes are uncoalesced, so it is slow
__global__ void transpose_naive_k(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS_PER_BLOCK)
        if (x < cols && y + j < rows)
            out[x * rows + (y + j)] = in[(y + j) * cols + x];
}

// 3b) Transpose (tiled): shared memory + padding (+1) gives coalesced
//     reads/writes and avoids shared-memory bank conflicts
__global__ void transpose_tiled_k(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS_PER_BLOCK)
        if (x < cols && y + j < rows)
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS_PER_BLOCK)
        if (x < rows && y + j < cols)
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---- C launchers (called from Python through ctypes) ----
extern "C" {

void vec_add(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add_k<<<blocks, threads>>>(a, b, c, n);
}

void reduce_sum(const float* x, float* out, int n) {
    cudaMemset(out, 0, sizeof(float));
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 1024);
    reduce_sum_k<<<blocks, threads>>>(x, out, n);
}

void transpose_naive(const float* in, float* out, int rows, int cols) {
    dim3 block(TILE, ROWS_PER_BLOCK);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_naive_k<<<grid, block>>>(in, out, rows, cols);
}

void transpose_tiled(const float* in, float* out, int rows, int cols) {
    dim3 block(TILE, ROWS_PER_BLOCK);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_tiled_k<<<grid, block>>>(in, out, rows, cols);
}

}  // extern "C"
