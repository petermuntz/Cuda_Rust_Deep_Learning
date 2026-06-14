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

#define TILE_M 128
#define TILE_N 64
#define TILE_K 32

extern "C" __global__ void matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    int tx = threadIdx.x;  // 0..31: K-dim for A load, N-dim for B load
    int ty = threadIdx.y;  // 0..31: M-group

    float accum[8] = {0.0f};

    for (int k = 0; k < K; k += TILE_K) {
        for (int i = 0; i < 4; i++) {
            int m = ty * 4 + i;
            As[m][tx] = A[(block_row + m) * K + (k + tx)];
        }

        for (int j = 0; j < 2; j++) {
            int n = tx * 2 + j;
            Bs[ty][n] = B[(k + ty) * N + (block_col + n)];
        }

        __syncthreads();

        for (int kk = 0; kk < TILE_K; kk++) {
            for (int i = 0; i < 4; i++) {
                float a_val = As[ty * 4 + i][kk];
                for (int j = 0; j < 2; j++) {
                    accum[i * 2 + j] += a_val * Bs[kk][tx * 2 + j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            int row = block_row + ty * 4 + i;
            int col = block_col + tx * 2 + j;
            if (row < M && col < N) {
                C[row * N + col] = accum[i * 2 + j];
            }
        }
    }
}

#define TILE32 32

extern "C" __global__ void matmul_tiled_32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE32][TILE32];
    __shared__ float Bs[TILE32][TILE32];

    int row = blockIdx.y * TILE32 + threadIdx.y;
    int col = blockIdx.x * TILE32 + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < K; k += TILE32) {
        As[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        __syncthreads();

        for (int i = 0; i < TILE32; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
