#include <mma.h>
#include <cuda_fp16.h>

#define TILE_M 128
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
    int warp_m = warp_id / 2;
    int warp_pair = warp_id % 2;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[2];
    nvcuda::wmma::fill_fragment(c_frag[0], 0.0f);
    nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);

    for (int k = 0; k < K; k += TILE_K) {
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 512;
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            if (r < TILE_M) {
                As[r][c] = __float2half(A[(block_row + r) * K + (k + c)]);
            }
        }

        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 512;
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            if (r < TILE_K) {
                Bs[r][c] = __float2half(B[(k + r) * N + (block_col + c)]);
            }
        }

        __syncthreads();

        for (int n_sub = 0; n_sub < 2; n_sub++) {
            int warp_n = 2 * warp_pair + n_sub;
            #pragma unroll 2
            for (int kk = 0; kk < TILE_K; kk += 16) {
                nvcuda::wmma::load_matrix_sync(a_frag, &As[warp_m * 16][kk], TILE_K);
                nvcuda::wmma::load_matrix_sync(b_frag, &Bs[kk][warp_n * 16], TILE_N);
                nvcuda::wmma::mma_sync(c_frag[n_sub], a_frag, b_frag, c_frag[n_sub]);
            }
        }

        __syncthreads();
    }

    for (int n_sub = 0; n_sub < 2; n_sub++) {
        int c_row = block_row + warp_m * 16;
        int c_col = block_col + (2 * warp_pair + n_sub) * 16;
        if (c_row < M && c_col < N) {
            nvcuda::wmma::store_matrix_sync(C + c_row * N + c_col, c_frag[n_sub], N, nvcuda::wmma::mem_row_major);
        }
    }
}
