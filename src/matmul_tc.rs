use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync};
use half::f16;
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

    let host_a_f32: Vec<f32> = (0..M * K).map(|i| (i % 127) as f32).collect();
    let host_b_f32: Vec<f32> = (0..K * N).map(|i| ((i * 3) % 255) as f32).collect();
    let host_c_ref = cpu_matmul(&host_a_f32, &host_b_f32, M, N, K);

    println!("\nComputing reference on CPU... done.");

    let host_a_half: Vec<f16> = host_a_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_b_half: Vec<f16> = host_b_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let d_a = dev.htod_copy(host_a_half)?;
    let d_b = dev.htod_copy(host_b_half)?;
    let mut d_c = dev.alloc_zeros::<f32>(M * N)?;

    dev.load_ptx(PTX.into(), "matmul_tc_module", &["matmul_tiled_tc"])?;
    let tc_func = dev.get_func("matmul_tc_module", "matmul_tiled_tc").unwrap();

    let grid = ((N as u32 + 63) / 64, (M as u32 + 63) / 64, 1);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (32, 16, 1),
        shared_mem_bytes: 0,
    };

    unsafe { tc_func.clone().launch(cfg, (&d_a, &d_b, &mut d_c, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;

    let start = Instant::now();
    unsafe { tc_func.launch(cfg, (&d_a, &d_b, &mut d_c, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;
    let tc_time = start.elapsed();

    let host_c_tc: Vec<f32> = dev.dtoh_sync_copy(&d_c)?;
    let tc_diff = max_diff(&host_c_ref, &host_c_tc);
    let tc_gflops = (2.0 * M as f64 * N as f64 * K as f64) / tc_time.as_secs_f64() / 1e9;

    println!("\n══════════ Tensor Core Matmul {}×{} × {}×{} ══════════", M, K, K, N);
    println!("{:<28} {:>10} {:>12} {:>12}", "Kernel", "Time", "MaxErr", "GFLOPS");
    println!("{:-<28} {:->10} {:->12} {:->12}", "", "", "", "");
    println!("{:<28} {:>8.3}ms  {:>11.2e}  {:>8.2}", "Tensor Core (WMMA)", tc_time.as_secs_f64() * 1e3, tc_diff, tc_gflops);
    println!("═══════════════════════════════════════════════\n");

    Ok(())
}
