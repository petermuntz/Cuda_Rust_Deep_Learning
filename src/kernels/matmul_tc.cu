#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Minimal WMMA test: fill accumulator and store. No load_matrix_sync at all.
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 1.0f);
    nvcuda::wmma::store_matrix_sync(C, c_frag, N, nvcuda::wmma::mem_row_major);
}
