#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Test shared memory load_matrix_sync with stride 16 (no padding).
    // Each row is 16*2=32 bytes apart = 16-byte aligned.
    __shared__ __half buf[16][16];

    for (int i = 0; i < 8; i++) {
        int pos = threadIdx.x + i * 32;
        int r = pos / 16;
        int c = pos % 16;
        buf[r][c] = __float2half(A[r * K + c]);
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::load_matrix_sync(a_frag, buf[0], 16);

    C[threadIdx.x] = __half2float(a_frag.x[0]);
}
