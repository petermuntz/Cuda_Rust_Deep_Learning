// src/kernels/vector_add.cu
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int num_elements) {
    // Calculate the unique global thread ID for this specific parallel worker
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: make sure we don't read past the end of our arrays
    if (i < num_elements) {
        c[i] = a[i] + b[i];
    }
}
