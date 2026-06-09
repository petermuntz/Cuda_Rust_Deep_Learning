use cudarc::driver::{CudaContext, LaunchConfig};

// This magic macro includes the compiled GPU assembly bytes right into your binary at compile time.
// Your `build.rs` script outputs this file automatically.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vector_add.ptx"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing CUDA Device...");

    // 1. Initialize the GPU device (0 is your first graphics card, which maps to the Colab T4)
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // 2. Load our compiled assembly string into the GPU memory modules
    let module = ctx.load_module(PTX.to_string())?;
    // Load our specific kernel function by its C name string
    let vector_add_kernel = module.load_function("vector_add")?;

    // 3. Define the size of our vectors
    let num_elements = 10_000;
    
    // Create some sample data on the host (CPU)
    let host_a = vec![1.0f32; num_elements];
    let host_b = vec![2.0f32; num_elements];

    println!("Allocating memory on the GPU and copying data over...");
    // 4. Allocate memory on the GPU and securely copy host data to it using cudarc RAII
    let device_a = stream.clone_htod(&host_a)?; // Host to Device
    let device_b = stream.clone_htod(&host_b)?;
    
    // Allocate an empty space on the GPU to store our outputs
    let mut device_c = stream.alloc_zeros::<f32>(num_elements)?;

    // 5. Configure the Execution Grid (Blocks and Threads)
    // A single block can comfortably handle 256 parallel threads.
    let threads_per_block = 256;
    let blocks_per_grid = (num_elements as u32 + threads_per_block - 1) / threads_per_block;
    
    let cfg = LaunchConfig {
        grid_dim: (blocks_per_grid, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Launching parallel CUDA kernel on the GPU grid...");
    // 6. Launch the kernel safely using Rust's builder pattern!
    let mut builder = stream.launch_builder(&vector_add_kernel);
    builder
        .arg(&device_a)
        .arg(&device_b)
        .arg(&mut device_c)
        .arg(&(num_elements as i32));
    
    unsafe { builder.launch(cfg)? };

    println!("Copying results back to the CPU...");
    // 7. Download the results back from the GPU memory to the CPU host
    let host_c = stream.clone_dtoh(&device_c)?; // Device to Host

    // 8. Verify the GPU did the math correctly
    println!("Verifying math results:");
    println!("A[0] ({}) + B[0] ({}) = C[0] ({})", host_a[0], host_b[0], host_c[0]);
    println!("A[9999] ({}) + B[9999] ({}) = C[9999] ({})", host_a[9999], host_b[9999], host_c[9999]);

    println!("Success! Memory automatically cleared via Rust RAII dropping out of scope.");
    Ok(())
}
