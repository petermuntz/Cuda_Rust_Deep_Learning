use std::path::PathBuf;

fn main() {
    // Tell Cargo to re-run this script if our CUDA file changes
    println!("cargo:rerun-if-changed=src/kernels/vector_add.cu");

    // We'll output the compiled assembly to Cargo's build directory
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Use the `cc` crate to invoke nvcc under the hood
    cc::Build::new()
        .cuda(true)
        .file("src/kernels/vector_add.cu")
        .compile("vector_add"); 

    // Move the generated PTX to where our Rust code can find it
    std::fs::copy(
        out_dir.join("src/kernels/vector_add.ptx"),
        out_dir.join("vector_add.ptx"),
    ).unwrap();
}
