use std::process::Command;
use std::path::PathBuf;

fn compile_cu(name: &str, out_dir: &PathBuf) {
    let src = format!("src/kernels/{}.cu", name);
    let out = out_dir.join(format!("{}.ptx", name));
    println!("cargo:rerun-if-changed={}", src);
    let status = Command::new("nvcc")
        .args(&["-ptx", &src, "-o", out.to_str().unwrap()])
        .status()
        .unwrap_or_else(|_| panic!("Failed to run nvcc for {}", src));
    assert!(status.success(), "nvcc compilation failed for {}", src);
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    compile_cu("vector_add", &out_dir);
    compile_cu("matmul", &out_dir);
}
