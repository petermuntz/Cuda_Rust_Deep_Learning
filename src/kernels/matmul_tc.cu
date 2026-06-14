#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Test: load fragment from shared memory, read one element.
    // No mma_sync. Isolates shared-memory ldmatrix as the fault source.
    __shared__ __half buf[16][17];

    for (int i = 0; i < 8; i++) {
        int pos = threadIdx.x + i * 32;
        int r = pos / 16;
        int c = pos % 16;
        buf[r][c] = __float2half(A[r * K + c]);
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::load_matrix_sync(a_frag, buf[0], 17);

    // Prevent optimization
    C[threadIdx.x] = __half2float(a_frag.x[0]);
}
