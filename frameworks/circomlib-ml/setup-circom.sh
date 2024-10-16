curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

git clone https://github.com/iden3/circom.git

# checkout to 2eaaa6dface934356972b34cab64b25d382e59de (version 2.1.9)
git checkout 2eaaa6dface934356972b34cab64b25d382e59de

cd circom

cargo build --release
cargo install --path circom

npm install -g snarkjs@0.4.19
