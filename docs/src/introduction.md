# yao-rs

A Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl) focused on quantum circuit description and tensor network export.

## Features

- **Gate enum** with named qubit gates (X, Y, Z, H, S, T, SWAP, Rx, Ry, Rz) and custom qudit gates
- **Qudit support** with per-site dimensions
- **Circuit validation** with controlled gates (qubit-only controls)
- **Tensor network export** via [omeco](https://crates.io/crates/omeco) for contraction order optimization
- **Diagonal gate optimization** in tensor networks (shared legs vs input/output legs)
- **Generic apply** for correctness verification
