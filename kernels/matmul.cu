// matmul.cu - C = A * B with A (M x K), B (K x N), C (M x N), row-major float32
#include <cuda_runtime.h>

#define TILE 32

// 1) Naive: one thread per output element, every operand read from global memory.
//    threadIdx.x runs along the columns, so reads of B and writes of C are coalesced.
__global__ void matmul_naive_k(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] = acc;
    }
}

// 2) Tiled: each block stages TILE x TILE tiles of A and B in shared memory,
//    so every value fetched from global memory is reused TILE times.
__global__ void matmul_tiled_k(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[(size_t)b_row * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// ---- C launchers ----
extern "C" {

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_naive_k<<<grid, block>>>(A, B, C, M, N, K);
}

void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_k<<<grid, block>>>(A, B, C, M, N, K);
}

}  // extern "C"
