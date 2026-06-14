#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // One block, one warp (32 threads), one 16x16 MMA tile from (0,0).
    // This is the absolute minimum WMMA smoke test.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    nvcuda::wmma::load_matrix_sync(a_frag, A, K);
    nvcuda::wmma::load_matrix_sync(b_frag, B, N);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    nvcuda::wmma::store_matrix_sync(C, c_frag, N, nvcuda::wmma::mem_row_major);
}
