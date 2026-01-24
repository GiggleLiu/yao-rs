# Introduction

yao-rs is a Rust library for describing quantum circuits and exporting them as tensor networks. It is a port of the Julia [Yao.jl](https://github.com/QuantumBFS/Yao.jl) framework, focused on the circuit description and tensor network layers, bringing compile-time safety and performance to quantum circuit modeling.

## Key capabilities

- Named qubit gates (X, Y, Z, H, S, T, SWAP) and parameterized rotations (Rx, Ry, Rz, Phase)
- Custom qudit gates with arbitrary matrices
- Circuit construction with `put` and `control` builder API
- Per-site qudit dimensions (not limited to qubits)
- Circuit validation with detailed error types
- Tensor network export via `circuit_to_einsum`
- Diagonal gate optimization (shared legs in tensor networks)
- Contraction order optimization via omeco
- Direct state-vector simulation via `apply`

## Next steps

See [Getting Started](./getting-started.md) to install yao-rs and build your first circuit.
