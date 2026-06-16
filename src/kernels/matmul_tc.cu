#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

extern "C" __global__ void matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As_raw[TILE_M][TILE_K];
    __shared__ float Bs_raw[TILE_K][TILE_N];
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
        // Phase 1: Async copy A tile from global to As_raw (16 bytes per thread x 4 floats)
        {
            int r = tid / (TILE_K / 4);
            int c = (tid % (TILE_K / 4)) * 4;
            if (r < TILE_M) {
                __pipeline_memcpy_async(&As_raw[r][c], &A[(block_row + r) * K + k + c], 16);
            }
        }

        // Phase 2: Async copy B tile from global to Bs_raw
        {
            int r = tid / (TILE_N / 4);
            int c = (tid % (TILE_N / 4)) * 4;
            if (r < TILE_K) {
                __pipeline_memcpy_async(&Bs_raw[r][c], &B[(k + r) * N + block_col + c], 16);
            }
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // Phase 3: Convert float -> half in shared memory
        {
            int r = tid / (TILE_K / 4);
            int c = (tid % (TILE_K / 4)) * 4;
            if (r < TILE_M) {
                As[r][c] = __float2half(As_raw[r][c]);
                As[r][c + 1] = __float2half(As_raw[r][c + 1]);
                As[r][c + 2] = __float2half(As_raw[r][c + 2]);
                As[r][c + 3] = __float2half(As_raw[r][c + 3]);
            }
        }

        {
            int r = tid / (TILE_N / 4);
            int c = (tid % (TILE_N / 4)) * 4;
            if (r < TILE_K) {
                Bs[r][c] = __float2half(Bs_raw[r][c]);
                Bs[r][c + 1] = __float2half(Bs_raw[r][c + 1]);
                Bs[r][c + 2] = __float2half(Bs_raw[r][c + 2]);
                Bs[r][c + 3] = __float2half(Bs_raw[r][c + 3]);
            }
        }

        __syncthreads();

        // Phase 4: WMMA compute
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
