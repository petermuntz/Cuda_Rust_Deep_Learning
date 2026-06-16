# Google Sloplab system installs & exec pipeline
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y ; source /root/.cargo/env ; rm -rf Cuda_Rust_Deep_Learning ; git clone https://github.com/petermuntz/Cuda_Rust_Deep_Learning.git ; cd Cuda_Rust_Deep_Learning ; cargo run --release



# Operations:
[x] Vector Add  
[x] Matmul 1024x1024 naive 1:1 thread allocation
[x] Matmul 1024x1024 naive tiling, 32x32 tile per block
[x] Matmul 1024x1024 coalesced memory access tiling, 128x64 tile per block 
[x] Matmul 1024x1024 coalesced memory access tiling, 128x128 tile per block
[x] Tensor Core 1024x1024 optimal matmul

══════════ Matmul 1024×1024 × 1024×1024 ══════════
Kernel                             Time    Speedup       MaxErr       GFLOPS
---------------------------- ---------- ---------- ------------ ------------
Naive (1:1)                     9.227ms      1.0×       0.00e0    232.73
Tiled (coalesced)               2.531ms      3.6×       0.00e0    848.41
Tiled 32×32                     5.256ms      1.8×       0.00e0    408.60
Tiled 128×128 (16/thread)       2.386ms      3.9×       0.00e0    900.18
Tiled double-buf (4/thr)        3.566ms      2.6×       0.00e0    602.25
═══════════════════════════════════════════════

══════════ Tensor Core Matmul 1024×1024 × 1024×1024 ══════════
Kernel                             Time       MaxErr       GFLOPS
---------------------------- ---------- ------------ ------------
Tensor Core (WMMA)              1.459ms       0.00e0   1471.41
═══════════════════════════════════════════════


# To do:
[] Realistic Size and Shape Tensor multiplicaiton
[] Residual conneciton addition

[] Normalization Layer

[] Activation function
[] Hadamard product / SWIGLU 
