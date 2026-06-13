// src/kernels/vector_add.cu
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int num_elements) {
	// Grid-stride loop:
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;  // total threads in grid
	for (int i = idx; i < num_elements; i += stride) {
	    c[i] = a[i] + b[i];
	}
}
