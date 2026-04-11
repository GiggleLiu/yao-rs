# yao-rs

[![CI](https://github.com/GiggleLiu/yao-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/GiggleLiu/yao-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GiggleLiu/yao-rs/graph/badge.svg?token=UwHXVMpsP3)](https://codecov.io/gh/GiggleLiu/yao-rs)
[![Docs](https://github.com/GiggleLiu/yao-rs/actions/workflows/docs.yml/badge.svg)](https://giggleliu.github.io/yao-rs/)

A Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl) focused on quantum circuit description, qubit simulation, tensor network export, and built-in SVG circuit visualization.

## Features

- **Gate enum** with named qubit gates (X, Y, Z, H, S, T, SWAP, Rx, Ry, Rz) and custom qudit gates
- **Qudit support** with per-site dimensions
- **Circuit validation** with controlled gates (qubit-only controls)
- **Qubit simulation** via `ArrayReg` and direct circuit application
- **Tensor network export** via [omeco](https://crates.io/crates/omeco) for contraction order optimization
- **Diagonal gate optimization** in tensor networks (shared legs vs input/output legs)
- **SVG circuit rendering** via `Circuit::to_svg()` and `yao visualize`
- **CLI tool** (`yao`) for simulation, measurement, tensor export, and visualization from the command line

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

// Render as SVG
let svg = circuit.to_svg();

// Or export as tensor network
let tn = circuit_to_einsum(&circuit);
```

### CLI

```bash
# Install
cargo install --path yao-cli

# Generate an example circuit
yao example bell > bell.json

# Simulate a Bell circuit and measure
yao run bell.json --shots 1024

# Compute expectation value
yao run bell.json --op "Z(0)Z(1)"

# Render a circuit diagram
yao visualize bell.json --output bell.svg

# Pipeline: simulate then get probabilities
yao simulate bell.json | yao probs -

# Export as tensor network
yao toeinsum bell.json --output tn.json
```

## Documentation

See the [mdBook documentation](https://giggleliu.github.io/yao-rs/) for detailed guides, including the [CLI guide](https://giggleliu.github.io/yao-rs/cli.html), the [Getting Started guide](https://giggleliu.github.io/yao-rs/getting-started.html), and the [QFT walkthrough](https://giggleliu.github.io/yao-rs/examples/qft.html).
