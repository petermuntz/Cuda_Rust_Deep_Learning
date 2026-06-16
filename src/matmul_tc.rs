use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync};
use std::sync::Arc;
use std::time::Instant;
use std::error::Error;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/matmul_tc.ptx"));

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn max_diff(cpu: &[f32], gpu: &[f32]) -> f32 {
    cpu.iter()
        .zip(gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

pub fn run_matmul_tc_benchmark(dev: &Arc<CudaDevice>) -> Result<(), Box<dyn Error>> {
    const M: usize = 1024;
    const N: usize = 1024;
    const K: usize = 1024;

    let host_a: Vec<f32> = (0..M * K).map(|i| (i % 127) as f32).collect();
    let host_b: Vec<f32> = (0..K * N).map(|i| ((i * 3) % 255) as f32).collect();
    let host_c_ref = cpu_matmul(&host_a, &host_b, M, N, K);

    println!("\nComputing reference on CPU... done.");

    println!("  Loading PTX...");
    dev.load_ptx(PTX.into(), "matmul_tc_module", &["matmul_tiled_tc", "convert_f32_to_f16"])?;
    let tc_func = dev.get_func("matmul_tc_module", "matmul_tiled_tc").unwrap();
    let convert_func = dev.get_func("matmul_tc_module", "convert_f32_to_f16").unwrap();

    println!("  Allocating device memory...");
    let d_a_f32 = dev.htod_copy(host_a)?;
    let d_b_f32 = dev.htod_copy(host_b)?;
    let mut d_a_half = dev.alloc_zeros::<u16>(M * K)?;
    let mut d_b_half = dev.alloc_zeros::<u16>(K * N)?;
    let mut d_c = dev.alloc_zeros::<f32>(M * N)?;
    println!("  Allocated device memory");

    println!("  Converting f32 -> f16 on GPU...");
    let n_elem = (M * K) as i32;
    let convert_cfg = LaunchConfig {
        grid_dim: (256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { convert_func.clone().launch(convert_cfg, (&d_a_f32, &mut d_a_half, n_elem))?; }
    unsafe { convert_func.clone().launch(convert_cfg, (&d_b_f32, &mut d_b_half, (K * N) as i32))?; }
    dev.synchronize()?;
    println!("  Conversion done");

    // Full 1024x1024 matmul: 64x64 tiles, half-precision inputs, block of 512 threads (16 warps)
    let cfg = LaunchConfig {
        grid_dim: (16, 16, 1),
        block_dim: (32, 16, 1),
        shared_mem_bytes: 0,
    };

    println!("  Launching warmup...");
    unsafe { tc_func.clone().launch(cfg, (&d_a_half, &d_b_half, &mut d_c, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;
    println!("  Warmup sync OK");

    println!("  Launching timed run...");
    let start = Instant::now();
    unsafe { tc_func.launch(cfg, (&d_a_half, &d_b_half, &mut d_c, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;
    println!("  Timed sync OK");
    let tc_time = start.elapsed();

    println!("  Copying back results...");
    let host_c_tc: Vec<f32> = dev.dtoh_sync_copy(&d_c)?;
    println!("  Copy OK");

    let tc_diff = max_diff(&host_c_ref, &host_c_tc);
    let tc_gflops = (2.0 * M as f64 * N as f64 * K as f64) / tc_time.as_secs_f64() / 1e9;

    println!("\n══════════ Tensor Core Matmul {}×{} × {}×{} ══════════", M, K, K, N);
    println!("{:<28} {:>10} {:>12} {:>12}", "Kernel", "Time", "MaxErr", "GFLOPS");
    println!("{:-<28} {:->10} {:->12} {:->12}", "", "", "", "");
    println!("{:<28} {:>8.3}ms  {:>11.2e}  {:>8.2}", "Tensor Core (WMMA)", tc_time.as_secs_f64() * 1e3, tc_diff, tc_gflops);
    println!("═══════════════════════════════════════════════\n");

    Ok(())
}
