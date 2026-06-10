# Google Sloplab system installs & exec pipeline

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

source /root/.cargo/env

rm -rf Cuda_Rust_Deep_Learning

git clone https://github.com/petermuntz/Cuda_Rust_Deep_Learning.git

cd Cuda_Rust_Deep_Learning

cargo run --release
