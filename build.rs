use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/vector_add.cu");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let status = Command::new("nvcc")
        .args(&[
            "-ptx",
            "src/kernels/vector_add.cu",
            "-o",
            out_dir.join("vector_add.ptx").to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run nvcc");

    assert!(status.success(), "nvcc compilation failed");
}
