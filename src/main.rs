use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync};
use std::sync::Arc;

mod matmul;
mod matmul_tc;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vector_add.ptx"));

fn run_vector_add(dev: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    dev.load_ptx(PTX.into(), "vector_add_module", &["vector_add"])?;
    let vector_add_kernel = dev.get_func("vector_add_module", "vector_add").unwrap();

    let num_elements = 10_000;
    let host_a = vec![1.0f32; num_elements];
    let host_b = vec![2.0f32; num_elements];

    println!("Allocating memory on the GPU and copying data over...");
    let device_a = dev.htod_copy(host_a.clone())?;
    let device_b = dev.htod_copy(host_b.clone())?;
    let mut device_c = dev.alloc_zeros::<f32>(num_elements)?;

    let threads_per_block = 256;
    let blocks_per_grid = (num_elements as u32 + threads_per_block - 1) / threads_per_block;
    
    let cfg = LaunchConfig {
        grid_dim: (blocks_per_grid, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Launching parallel CUDA kernel on the GPU grid...");
    unsafe {
        vector_add_kernel.launch(
            cfg,
            (&device_a, &device_b, &mut device_c, num_elements as i32),
        )?
    };

    println!("Copying results back to the CPU...");
    let host_c = dev.dtoh_sync_copy(&device_c)?;

    println!("Verifying math results:");
    println!("A[0] ({}) + B[0] ({}) = C[0] ({})", host_a[0], host_b[0], host_c[0]);
    println!("A[9999] ({}) + B[9999] ({}) = C[9999] ({})", host_a[9999], host_b[9999], host_c[9999]);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing CUDA Device...");
    let dev = CudaDevice::new(0)?;

    println!("\n═══ Vector Add ═══");
    run_vector_add(&dev)?;

    println!("\n═══ MatMul Benchmark ═══");
    matmul::run_matmul_benchmark(&dev)?;

    println!("\n═══ Tensor Core MatMul Benchmark ═══");
    matmul_tc::run_matmul_tc_benchmark(&dev)?;

    println!("Success! Memory automatically cleared via Rust RAII dropping out of scope.");
    Ok(())
}
