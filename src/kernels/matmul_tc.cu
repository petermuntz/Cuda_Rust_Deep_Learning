#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Test: load A and B fragments from global memory, then copy one element
    // from each loaded fragment to C to prevent compiler from optimizing away
    // the load_matrix_sync calls. No mma_sync.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b_frag;

    nvcuda::wmma::load_matrix_sync(a_frag, reinterpret_cast<const __half*>(A), K);
    nvcuda::wmma::load_matrix_sync(b_frag, reinterpret_cast<const __half*>(B), N);

    // Use the loaded values to prevent optimization
    C[blockIdx.x] = __half2float(a_frag.x[0]);
    C[gridDim.x + blockIdx.x] = __half2float(b_frag.x[0]);
}
