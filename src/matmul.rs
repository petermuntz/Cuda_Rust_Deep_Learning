use cudarc::driver::{CudaDevice, LaunchConfig, LaunchAsync};
use std::sync::Arc;
use std::time::Instant;
use std::error::Error;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/matmul.ptx"));

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

pub fn run_matmul_benchmark(dev: &Arc<CudaDevice>) -> Result<(), Box<dyn Error>> {
    const M: usize = 1024;
    const N: usize = 1024;
    const K: usize = 1024;

    let host_a: Vec<f32> = (0..M * K).map(|i| (i % 127) as f32).collect();
    let host_b: Vec<f32> = (0..K * N).map(|i| ((i * 3) % 255) as f32).collect();
    let host_c_ref = cpu_matmul(&host_a, &host_b, M, N, K);

    println!("\nComputing reference on CPU... done.");

    let d_a = dev.htod_copy(host_a)?;
    let d_b = dev.htod_copy(host_b)?;
    let mut d_c_naive = dev.alloc_zeros::<f32>(M * N)?;
    let mut d_c_tiled = dev.alloc_zeros::<f32>(M * N)?;
    let mut d_c_tiled32 = dev.alloc_zeros::<f32>(M * N)?;
    let mut d_c_wide = dev.alloc_zeros::<f32>(M * N)?;
    let mut d_c_db = dev.alloc_zeros::<f32>(M * N)?;

    dev.load_ptx(PTX.into(), "matmul_module", &["matmul_naive", "matmul_tiled", "matmul_tiled_32", "matmul_tiled_128x128", "matmul_tiled_db"])?;

    // ── Naive matmul ──
    let naive_func = dev.get_func("matmul_module", "matmul_naive").unwrap();
    let grid_naive = ((N as u32 + 15) / 16, (M as u32 + 15) / 16, 1);
    let cfg_naive = LaunchConfig {
        grid_dim: grid_naive,
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    // warmup
    unsafe { naive_func.clone().launch(cfg_naive, (&d_a, &d_b, &mut d_c_naive, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;

    let start = Instant::now();
    unsafe { naive_func.launch(cfg_naive, (&d_a, &d_b, &mut d_c_naive, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;
    let naive_time = start.elapsed();

    let host_c_naive: Vec<f32> = dev.dtoh_sync_copy(&d_c_naive)?;
    let naive_diff = max_diff(&host_c_ref, &host_c_naive);
    let naive_gflops = (2.0 * M as f64 * N as f64 * K as f64) / naive_time.as_secs_f64() / 1e9;

    // ── Tiled matmul (8 outputs/thread) ──
    let tiled_func = dev.get_func("matmul_module", "matmul_tiled").unwrap();
    let grid_tiled = ((N as u32 + 63) / 64, (M as u32 + 127) / 128, 1);
    let cfg_tiled = LaunchConfig {
        grid_dim: grid_tiled,
        block_dim: (32, 32, 1),
        shared_mem_bytes: 0,
    };

    // warmup
    unsafe { tiled_func.clone().launch(cfg_tiled, (&d_a, &d_b, &mut d_c_tiled, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;

    let start = Instant::now();
    unsafe { tiled_func.launch(cfg_tiled, (&d_a, &d_b, &mut d_c_tiled, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;
    let tiled_time = start.elapsed();

    let host_c_tiled: Vec<f32> = dev.dtoh_sync_copy(&d_c_tiled)?;
    let tiled_diff = max_diff(&host_c_ref, &host_c_tiled);
    let tiled_gflops = (2.0 * M as f64 * N as f64 * K as f64) / tiled_time.as_secs_f64() / 1e9;

    // ── Tiled 32×32 matmul ──
    let tiled32_func = dev.get_func("matmul_module", "matmul_tiled_32").unwrap();
    let grid_tiled32 = ((N as u32 + 31) / 32, (M as u32 + 31) / 32, 1);
    let cfg_tiled32 = LaunchConfig {
        grid_dim: grid_tiled32,
        block_dim: (32, 32, 1),
        shared_mem_bytes: 0,
    };

    // warmup
    unsafe { tiled32_func.clone().launch(cfg_tiled32, (&d_a, &d_b, &mut d_c_tiled32, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;

    let start = Instant::now();
    unsafe { tiled32_func.launch(cfg_tiled32, (&d_a, &d_b, &mut d_c_tiled32, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;
    let tiled32_time = start.elapsed();

    let host_c_tiled32: Vec<f32> = dev.dtoh_sync_copy(&d_c_tiled32)?;
    let tiled32_diff = max_diff(&host_c_ref, &host_c_tiled32);
    let tiled32_gflops = (2.0 * M as f64 * N as f64 * K as f64) / tiled32_time.as_secs_f64() / 1e9;

    // ── Tiled 128×128 wide matmul (16 outputs/thread) ──
    let wide_func = dev.get_func("matmul_module", "matmul_tiled_128x128").unwrap();
    let grid_wide = ((N as u32 + 127) / 128, (M as u32 + 127) / 128, 1);
    let cfg_wide = LaunchConfig {
        grid_dim: grid_wide,
        block_dim: (32, 32, 1),
        shared_mem_bytes: 0,
    };

    unsafe { wide_func.clone().launch(cfg_wide, (&d_a, &d_b, &mut d_c_wide, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;

    let start = Instant::now();
    unsafe { wide_func.launch(cfg_wide, (&d_a, &d_b, &mut d_c_wide, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;
    let wide_time = start.elapsed();

    let host_c_wide: Vec<f32> = dev.dtoh_sync_copy(&d_c_wide)?;
    let wide_diff = max_diff(&host_c_ref, &host_c_wide);
    let wide_gflops = (2.0 * M as f64 * N as f64 * K as f64) / wide_time.as_secs_f64() / 1e9;

    // ── Tiled double-buffered matmul (4 outputs/thread) ──
    let db_func = dev.get_func("matmul_module", "matmul_tiled_db").unwrap();
    let grid_db = ((N as u32 + 63) / 64, (M as u32 + 63) / 64, 1);
    let cfg_db = LaunchConfig {
        grid_dim: grid_db,
        block_dim: (32, 32, 1),
        shared_mem_bytes: 0,
    };

    unsafe { db_func.clone().launch(cfg_db, (&d_a, &d_b, &mut d_c_db, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;

    let start = Instant::now();
    unsafe { db_func.launch(cfg_db, (&d_a, &d_b, &mut d_c_db, M as i32, N as i32, K as i32))?; }
    dev.synchronize()?;
    let db_time = start.elapsed();

    let host_c_db: Vec<f32> = dev.dtoh_sync_copy(&d_c_db)?;
    let db_diff = max_diff(&host_c_ref, &host_c_db);
    let db_gflops = (2.0 * M as f64 * N as f64 * K as f64) / db_time.as_secs_f64() / 1e9;

    // ── Results ──
    println!("\n══════════ Matmul {}×{} × {}×{} ══════════", M, K, K, N);
    println!("{:<28} {:>10} {:>10} {:>12} {:>12}", "Kernel", "Time", "Speedup", "MaxErr", "GFLOPS");
    println!("{:-<28} {:->10} {:->10} {:->12} {:->12}", "", "", "", "", "");
    println!("{:<28} {:>8.3}ms {:>8.1}×  {:>11.2e}  {:>8.2}", "Naive (1:1)", naive_time.as_secs_f64() * 1e3, 1.0, naive_diff, naive_gflops);
    println!("{:<28} {:>8.3}ms {:>8.1}×  {:>11.2e}  {:>8.2}", "Tiled (coalesced)", tiled_time.as_secs_f64() * 1e3, naive_time.as_secs_f64() / tiled_time.as_secs_f64(), tiled_diff, tiled_gflops);
    println!("{:<28} {:>8.3}ms {:>8.1}×  {:>11.2e}  {:>8.2}", "Tiled 32×32", tiled32_time.as_secs_f64() * 1e3, naive_time.as_secs_f64() / tiled32_time.as_secs_f64(), tiled32_diff, tiled32_gflops);
    println!("{:<28} {:>8.3}ms {:>8.1}×  {:>11.2e}  {:>8.2}", "Tiled 128×128 (16/thread)", wide_time.as_secs_f64() * 1e3, naive_time.as_secs_f64() / wide_time.as_secs_f64(), wide_diff, wide_gflops);
    println!("{:<28} {:>8.3}ms {:>8.1}×  {:>11.2e}  {:>8.2}", "Tiled double-buf (4/thr)", db_time.as_secs_f64() * 1e3, naive_time.as_secs_f64() / db_time.as_secs_f64(), db_diff, db_gflops);
    println!("═══════════════════════════════════════════════\n");

    Ok(())
}
