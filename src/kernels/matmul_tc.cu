extern "C" __global__ void matmul_tiled_tc(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Trivial kernel: copy A to C. Verifies launch infrastructure.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        C[idx] = A[idx];
    }
}
