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

## Overview

`yao-rs` is split into two main surfaces:

- The `yao-rs` library for circuit construction, simulation, tensor network export, and SVG rendering
- The `yao` CLI for running the same workflows from the terminal without writing Rust

For library usage, examples, and API details, see the documentation links at the end of this README.

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
```

The CLI also includes a tensor-network workflow by default:

```bash
# Export a circuit as a tensor network
yao toeinsum bell.json --output bell-tn.json

# Optimize contraction order
yao optimize bell-tn.json --output bell-tn-opt.json

# Contract the optimized tensor network
yao contract bell-tn-opt.json

# Full pipeline with no intermediate files
yao toeinsum bell.json --mode state | yao optimize - | yao contract -

# Overlap / expectation-style workflows
yao toeinsum bell.json --mode overlap | yao optimize - | yao contract -
yao toeinsum bell.json --op "Z(0)Z(1)" | yao optimize - | yao contract -
```

Other CLI capabilities include:

- `yao inspect` for circuit structure inspection
- `yao fromqasm` / `yao toqasm` for OpenQASM 2.0 conversion
- `yao fetch qasmbench ...` for benchmark circuit downloads

## Documentation

See the [mdBook documentation](https://giggleliu.github.io/yao-rs/) for detailed guides, including the [CLI guide](https://giggleliu.github.io/yao-rs/cli.html), the [Getting Started guide](https://giggleliu.github.io/yao-rs/getting-started.html), and the [QFT walkthrough](https://giggleliu.github.io/yao-rs/examples/qft.html).
