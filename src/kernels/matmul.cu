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

#define TILE_M_W 128
#define TILE_N_W 128
#define TILE_K_W 32

extern "C" __global__ void matmul_tiled_128x128(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_M_W][TILE_K_W + 1];
    __shared__ float Bs[TILE_K_W][TILE_N_W + 1];

    int block_row = blockIdx.y * TILE_M_W;
    int block_col = blockIdx.x * TILE_N_W;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float accum[16] = {0.0f};

    for (int k = 0; k < K; k += TILE_K_W) {
        for (int i = 0; i < 4; i++) {
            int m = ty * 4 + i;
            As[m][tx] = A[(block_row + m) * K + (k + tx)];
        }

        for (int j = 0; j < 4; j++) {
            int n = tx * 4 + j;
            Bs[ty][n] = B[(k + ty) * N + (block_col + n)];
        }

        __syncthreads();

        for (int kk = 0; kk < TILE_K_W; kk++) {
            for (int i = 0; i < 4; i++) {
                float a_val = As[ty * 4 + i][kk];
                for (int j = 0; j < 4; j++) {
                    accum[i * 4 + j] += a_val * Bs[kk][tx * 4 + j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = block_row + ty * 4 + i;
            int col = block_col + tx * 4 + j;
            if (row < M && col < N) {
                C[row * N + col] = accum[i * 4 + j];
            }
        }
    }
}

#define TILE_M_DB 64
#define TILE_N_DB 64
#define TILE_K_DB 32

extern "C" __global__ void matmul_tiled_db(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[2][TILE_M_DB][TILE_K_DB + 1];
    __shared__ float Bs[2][TILE_K_DB][TILE_N_DB + 1];

    int block_row = blockIdx.y * TILE_M_DB;
    int block_col = blockIdx.x * TILE_N_DB;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float accum[4] = {0.0f};
    int bank = 0;
    int next_bank = 1;

    for (int i = 0; i < 2; i++) {
        int m = ty * 2 + i;
        As[bank][m][tx] = A[(block_row + m) * K + tx];
    }
    for (int j = 0; j < 2; j++) {
        int n = tx * 2 + j;
        Bs[bank][ty][n] = B[ty * N + (block_col + n)];
    }
    __syncthreads();

    int k;
    for (k = TILE_K_DB; k < K; k += TILE_K_DB) {
        for (int i = 0; i < 2; i++) {
            int m = ty * 2 + i;
            As[next_bank][m][tx] = A[(block_row + m) * K + (k + tx)];
        }
        for (int j = 0; j < 2; j++) {
            int n = tx * 2 + j;
            Bs[next_bank][ty][n] = B[(k + ty) * N + (block_col + n)];
        }

        for (int kk = 0; kk < TILE_K_DB; kk++) {
            for (int i = 0; i < 2; i++) {
                float a_val = As[bank][ty * 2 + i][kk];
                for (int j = 0; j < 2; j++) {
                    accum[i * 2 + j] += a_val * Bs[bank][kk][tx * 2 + j];
                }
            }
        }

        __syncthreads();
        bank ^= 1;
        next_bank ^= 1;
    }

    for (int kk = 0; kk < TILE_K_DB; kk++) {
        for (int i = 0; i < 2; i++) {
            float a_val = As[bank][ty * 2 + i][kk];
            for (int j = 0; j < 2; j++) {
                accum[i * 2 + j] += a_val * Bs[bank][kk][tx * 2 + j];
            }
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int row = block_row + ty * 2 + i;
            int col = block_col + tx * 2 + j;
            if (row < M && col < N) {
                C[row * N + col] = accum[i * 2 + j];
            }
        }
    }
}
