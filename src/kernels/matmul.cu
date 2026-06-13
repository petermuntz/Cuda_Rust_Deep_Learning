// Tuned for 1024x1024 on T4 (sm_75)
#define TILE_M 128
#define TILE_N 32
#define TILE_K 16

// Each thread computes 8 outputs = TILE_N / (blockDim.y)
// Block dims: (TILE_M, TILE_N / 8) = (128, 4) = 512 threads
// Grid dims:  (N / TILE_N, M / TILE_M)

extern "C" __global__ void matmul_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" __global__ void matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int linear = ty * blockDim.x + tx;

    float accum[8] = {0.0f};

    for (int k = 0; k < K; k += TILE_K) {
        for (int i = 0; i < 4; i++) {
            int k_local = ty * 4 + i;
            int a_row = block_row + tx;
            int a_col = k + k_local;
            if (k_local < TILE_K && a_row < M && a_col < K) {
                As[tx][k_local] = A[a_row * K + a_col];
            }
        }

        int bk_row = linear / TILE_N;
        int bk_col = linear % TILE_N;
        int b_row = k + bk_row;
        int b_col = block_col + bk_col;
        if (bk_row < TILE_K && b_row < K && b_col < N) {
            Bs[bk_row][bk_col] = B[b_row * N + b_col];
        }

        __syncthreads();

        for (int ki = 0; ki < TILE_K; ki++) {
            float a_val = As[tx][ki];
            for (int oj = 0; oj < 8; oj++) {
                accum[oj] += a_val * Bs[ki][ty * 8 + oj];
            }
        }

        __syncthreads();
    }

    int row = block_row + tx;
    int base_col = block_col + ty * 8;
    for (int oj = 0; oj < 8; oj++) {
        if (row < M && base_col + oj < N) {
            C[row * N + base_col + oj] = accum[oj];
        }
    }
}
