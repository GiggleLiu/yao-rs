# Comparison with Yao.jl

yao-rs is a focused port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl), covering the circuit description and tensor network layers. Yao.jl is a full-featured quantum computing framework with additional capabilities beyond circuit representation.

## Feature Comparison

| Feature | Yao.jl | yao-rs | Notes |
|---------|--------|--------|-------|
| **Gates** | | | |
| Pauli gates (X, Y, Z) | Yes | Yes | |
| Hadamard (H) | Yes | Yes | |
| Phase gates (S, T) | Yes | Yes | |
| SWAP | Yes | Yes | |
| Rotation gates (Rx, Ry, Rz) | Yes | Yes | |
| Shift/Phase gate | Yes | Yes | `shift(θ)` in Yao.jl, `Phase(θ)` in yao-rs |
| Custom matrix gates | Yes | Yes | |
| Diagonal gate flag | No (inferred) | Yes | Explicit `is_diagonal` in yao-rs |
| General qudit gates | Yes | Yes | Arbitrary local dimension |
| Multi-parameter gates | Yes | No | e.g., U3 gate |
| Gate algebra (composition) | Yes | No | `chain`, `+`, `*` operators |
| Dagger/inverse gates | Yes | No | |
| **Circuits** | | | |
| `put` builder | Yes | Yes | Place gate on sites |
| `control` builder | Yes | Yes | Controlled gates |
| Active-high controls | Yes | Yes | Control triggers on \|1⟩ |
| Active-low controls | Yes | No | Control triggers on \|0⟩ |
| Nested circuits (`chain`) | Yes | No | Flat gate list in yao-rs |
| `subroutine` (subcircuits) | Yes | No | |
| `kron` (parallel gates) | Yes | No | Use multiple `put` calls instead |
| `repeat` (repeated gates) | Yes | No | |
| Circuit validation | Yes | Yes | Detailed error types in yao-rs |
| Qudit support (per-site dims) | Yes | Yes | |
| **Simulation** | | | |
| State-vector simulation | Yes | Yes | Full-matrix `apply` in yao-rs |
| Efficient apply (in-place) | Yes | No | yao-rs builds full matrices |
| Density matrix simulation | Yes | No | |
| GPU simulation | Yes | No | |
| Measurement/sampling | Yes | No | |
| Expectation values | Yes | No | |
| Noisy simulation | Yes | No | |
| **Tensor Networks** | | | |
| Tensor network export | Partial | Yes | Core feature of yao-rs |
| Diagonal gate optimization | No | Yes | Reduced tensor rank |
| Contraction order optimization | No | Yes | Via omeco |
| EinCode/einsum representation | No | Yes | |
| **State Representation** | | | |
| Zero state | Yes | Yes | |
| Product state | Yes | Yes | |
| Arbitrary state vectors | Yes | No | Only basis states as constructors |
| Row-major indexing | Yes | Yes | |
| Qudit states | Yes | Yes | |
| **Differentiation** | | | |
| Automatic differentiation | Yes | No | |
| Parameter shift rules | Yes | No | |
| **Visualization** | | | |
| Circuit drawing | Yes | No | |
| State visualization | Yes | No | |
| **Interoperability** | | | |
| OpenQASM export | Yes | No | |
| Other framework interop | Yes | No | |

## Summary

**yao-rs strengths:**
- Tensor network export with diagonal gate optimization
- Contraction order optimization via omeco
- Compile-time type safety (Rust)
- Detailed circuit validation error types
- Explicit diagonal gate annotation

**Yao.jl strengths:**
- Full quantum computing framework
- Efficient state-vector simulation (in-place operations)
- Automatic differentiation for variational algorithms
- Circuit algebra and composition operators
- Measurement, sampling, and expectation values
- GPU support
- Circuit visualization
- Broad ecosystem integration
