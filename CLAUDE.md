# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

yao-rs is a Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl) focused on quantum circuit description and tensor network export. It supports qudit (not just qubit) circuits with per-site dimensions.

## Build and Test Commands

```bash
make check-all     # Run fmt-check, clippy, and test (use before commits)
make test          # Run test suite with all features
make clippy        # Run clippy with -D warnings
make fmt           # Format code
make fmt-check     # Check formatting without modifying

# Run a single test
cargo test --all-features test_name

# Documentation
make doc           # Build mdBook documentation
make rustdoc       # Build and open Rust API docs
```

## Architecture

### Core Data Flow

```
Gate → PositionedGate → Circuit → TensorNetwork/State
```

- **Gate** (`gate.rs`): Enum of quantum gates (X, Y, Z, H, S, T, SWAP, Rx, Ry, Rz, etc.) with matrix representations. Named gates require d=2 (qubits); `Gate::Custom` supports arbitrary qudit matrices.

- **PositionedGate** (`circuit.rs`): Gate + placement info (`target_locs`, `control_locs`, `control_configs`). Built via `put()` for uncontrolled gates or `control()` for controlled gates.

- **Circuit** (`circuit.rs`): Validated sequence of positioned gates on a register with specified dimensions (`dims: Vec<usize>`). Constructor validates: control sites must be qubits, no overlapping control/target locs, matrix size matches target dimensions.

- **State** (`state.rs`): Dense state vector as `ArrayD<Complex64>`.

### Two Execution Paths

1. **Direct application** (`apply.rs`, `instruct.rs`): State vector simulation via `apply(&circuit, &state)`. Uses optimized routines (`u1rows`, `udrows`, `mulrow`) for single-qubit, diagonal, and controlled gates.

2. **Tensor network export** (`einsum.rs`, `tensors.rs`): Convert circuit to einsum via `circuit_to_einsum()`. Returns `TensorNetwork` with `EinCode<usize>` (from omeco crate), tensors, and size dictionary. Diagonal gates without controls share input/output legs (optimization).

### Key Design Patterns

- **Diagonal gate optimization**: `Gate::is_diagonal()` enables tensor leg reuse in einsum and specialized application paths.
- **Control configurations**: `control_configs: Vec<bool>` allows active-low controls (trigger on |0⟩), though the `control()` helper defaults to active-high.
- **Qudit generality**: Circuit dimensions are `Vec<usize>`, not assumed 2. Named gates assert d=2; custom gates work with any dimension.

### Module Overview

- `easybuild.rs`: Pre-built circuits (QFT, variational ansatz, phase estimation, Google supremacy)
- `measure.rs`: Measurement operations (`probs`, `measure`, `measure_and_collapse`)
- `operator.rs`: Pauli operators and polynomials for expectation values
- `json.rs`: Circuit serialization
- `typst.rs` (feature-gated): PDF circuit rendering via embedded Typst

## Feature Flags

- `parallel`: Enable rayon for parallel operations
- `torch`: PyTorch tensor contraction via tch
- `typst`: PDF circuit diagram generation
