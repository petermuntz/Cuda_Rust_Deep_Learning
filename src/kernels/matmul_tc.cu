#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Single 16x16 tile: 32 threads cooperatively load 16x16 into shared memory,
    // then one warp does WMMA load/mma/store.
    __shared__ __half As[16][17];
    __shared__ __half Bs[16][17];

    // 32 threads load 8 elements each = 256 = 16x16
    for (int i = 0; i < 8; i++) {
        int pos = threadIdx.x + i * 32;
        int r = pos / 16;
        int c = pos % 16;
        As[r][c] = __float2half(A[r * K + c]);
        Bs[r][c] = __float2half(B[r * N + c]);
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    nvcuda::wmma::load_matrix_sync(a_frag, As[0], 17);
    nvcuda::wmma::load_matrix_sync(b_frag, Bs[0], 17);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    nvcuda::wmma::store_matrix_sync(C, c_frag, N, nvcuda::wmma::mem_row_major);
}
