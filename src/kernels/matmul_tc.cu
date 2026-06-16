#include <mma.h>
#include <cuda_fp16.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

extern "C" __global__ void __launch_bounds__(512) matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ __half As[TILE_M][TILE_K];
    __shared__ __half Bs[TILE_K][TILE_N];

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.y;
    int warp_m = warp_id / 4;
    int warp_n = warp_id % 4;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += TILE_K) {
        // Load As tile (64x32) using float2 -> __half2 vectorized loads
        #pragma unroll 2
        for (int i = 0; i < 2; i++) {
            int elem_idx = tid + i * 512;
            int r = elem_idx / 16;
            int c = (elem_idx % 16) * 2;
            float2 f2 = reinterpret_cast<const float2*>(A + (block_row + r) * K + (k + c))[0];
            *reinterpret_cast<__half2*>(&As[r][c]) = __float22half2_rn(f2);
        }

        #pragma unroll 2
        for (int i = 0; i < 2; i++) {
            int elem_idx = tid + i * 512;
            int r = elem_idx / 32;
            int c = (elem_idx % 32) * 2;
            float2 f2 = reinterpret_cast<const float2*>(B + (k + r) * N + (block_col + c))[0];
            *reinterpret_cast<__half2*>(&Bs[r][c]) = __float22half2_rn(f2);
        }

        __syncthreads();

        #pragma unroll 2
        for (int kk = 0; kk < TILE_K; kk += 16) {
            nvcuda::wmma::load_matrix_sync(a_frag, &As[warp_m * 16][kk], TILE_K);
            nvcuda::wmma::load_matrix_sync(b_frag, &Bs[kk][warp_n * 16], TILE_N);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    int c_row = block_row + warp_m * 16;
    int c_col = block_col + warp_n * 16;
    if (c_row < M && c_col < N) {
        nvcuda::wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, nvcuda::wmma::mem_row_major);
    }
}
