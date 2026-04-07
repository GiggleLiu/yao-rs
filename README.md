# yao-rs

[![CI](https://github.com/GiggleLiu/yao-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/GiggleLiu/yao-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GiggleLiu/yao-rs/graph/badge.svg?token=UwHXVMpsP3)](https://codecov.io/gh/GiggleLiu/yao-rs)
[![Docs](https://github.com/GiggleLiu/yao-rs/actions/workflows/docs.yml/badge.svg)](https://giggleliu.github.io/yao-rs/)

A Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl) focused on quantum circuit description and tensor network export.

## Features

- **Gate enum** with named qubit gates (X, Y, Z, H, S, T, SWAP, Rx, Ry, Rz) and custom qudit gates
- **Qudit support** with per-site dimensions
- **Circuit validation** with controlled gates (qubit-only controls)
- **Tensor network export** via [omeco](https://crates.io/crates/omeco) for contraction order optimization
- **Diagonal gate optimization** in tensor networks (shared legs vs input/output legs)
- **Generic apply** for correctness verification
- **CLI tool** (`yao`) for simulation, measurement, and tensor export from the command line

## Quick Start

### Library

```rust
use yao_rs::{Gate, Circuit, ArrayReg, put, control, apply, circuit_to_einsum};

// Build a Bell circuit: H on qubit 0, then CNOT(0→1)
let circuit = Circuit::new(vec![2, 2], vec![
    put(vec![0], Gate::H),
    control(vec![0], vec![1], Gate::X),
]).unwrap();

// Apply to |00⟩
let reg = ArrayReg::zero_state(2);
let result = apply(&circuit, &reg);

// Or export as tensor network
let tn = circuit_to_einsum(&circuit);
```

### CLI

```bash
# Install
cargo install --path yao-cli

# Simulate a Bell circuit and measure
yao run bell.json --shots 1024

# Compute expectation value
yao run bell.json --op "Z(0)Z(1)"

# Pipeline: simulate then get probabilities
yao simulate bell.json | yao probs -

# Export as tensor network
yao toeinsum bell.json --output tn.json
```

## Documentation

See the [mdBook documentation](https://giggleliu.github.io/yao-rs/) for detailed guides.
