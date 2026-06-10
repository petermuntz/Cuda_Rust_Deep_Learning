use cudarc::driver::{CudaDevice, LaunchConfig};
use std::sync::Arc;

// Include the compiled GPU assembly bytes dynamically from the Cargo build directory
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vector_add.ptx"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing CUDA Device...");

    // 1. Initialize the GPU device (0 is the T4 GPU on Colab)
    let dev = CudaDevice::new(0)?;

    // 2. Load our compiled assembly string into the GPU
    dev.load_ptx(PTX.into(), "vector_add_module", &["vector_add"])?;
    
    // Load our specific kernel function handle
    let vector_add_kernel = dev.get_func("vector_add_module", "vector_add").unwrap();

    // 3. Define the size of our vectors
    let num_elements = 10_000;
    
    // Create vector data on the host CPU
    let host_a = vec![1.0f32; num_elements];
    let host_b = vec![2.0f32; num_elements];

    println!("Allocating memory on the GPU and copying data over...");
    // 4. Transfer data from Host (CPU) to Device (GPU)
    let device_a = dev.htod_copy(host_a.clone())?;
    let device_b = dev.htod_copy(host_b.clone())?;
    
    // Allocate empty device memory for results
    let mut device_c = dev.alloc_zeros::<f32>(num_elements)?;

    // 5. Configure parallel grid execution layouts (Blocks and Threads)
    let threads_per_block = 256;
    let blocks_per_grid = (num_elements as u32 + threads_per_block - 1) / threads_per_block;
    
    let cfg = LaunchConfig {
        grid_dim: (blocks_per_grid, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Launching parallel CUDA kernel on the GPU grid...");
    // 6. Launch the kernel!
    unsafe {
        vector_add_kernel.launch(
            cfg,
            (&device_a, &device_b, &mut device_c, num_elements as i32),
        )?
    };

    println!("Copying results back to the CPU...");
    // 7. Download output calculations back to host memory
    let host_c = dev.dtoh_sync_copy(&device_c)?;

    // 8. Verify accuracy
    println!("Verifying math results:");
    println!("A[0] ({}) + B[0] ({}) = C[0] ({})", host_a[0], host_b[0], host_c[0]);
    println!("A[9999] ({}) + B[9999] ({}) = C[9999] ({})", host_a[9999], host_b[9999], host_c[9999]);

    println!("Success! Memory automatically cleared via Rust RAII dropping out of scope.");
    Ok(())
}
