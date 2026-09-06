# yao-rs

[![CI](https://github.com/GiggleLiu/yao-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/GiggleLiu/yao-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GiggleLiu/yao-rs/graph/badge.svg?token=UwHXVMpsP3)](https://codecov.io/gh/GiggleLiu/yao-rs)
[![Docs](https://github.com/GiggleLiu/yao-rs/actions/workflows/docs.yml/badge.svg)](https://giggleliu.github.io/yao-rs/)

A Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl) focused on quantum circuit description, qubit simulation, tensor network export, and built-in SVG circuit visualization.

## Features

- **Gate enum** with named qubit gates (X, Y, Z, H, S, T, SWAP, Rx, Ry, Rz) and custom qudit gates
- **Qudit support** with per-site dimensions for tensor-network export; state simulation is qubit-only
- **Circuit validation** with controlled gates (qubit-only controls)
- **Qubit simulation** via `ArrayReg`, plus density-matrix simulation of noise channels
- **Differentiable circuits** with expectation gradients, parameter/input-state VJPs and JVPs, shared physical parameters, and tenferro custom losses
- **Tensor network export** via [omeco](https://crates.io/crates/omeco) for contraction order optimization
- **Diagonal gate optimization** in tensor networks (shared legs vs input/output legs)
- **SVG circuit rendering** via `Circuit::to_svg()` and `yao visualize`
- **CLI tool** (`yao`) for simulation, measurement, tensor export, and visualization from the command line

## Overview

`yao-rs` is split into two main surfaces:

- The `yao-rs` library for circuit construction, simulation, tensor network export, and SVG rendering
- The `yao` CLI for running the same workflows from the terminal without writing Rust

## Rust library

```toml
[dependencies]
yao-rs = "0.1"
```

```rust
use yao_rs::{ArrayReg, Circuit, Gate, apply, control, probs, put};

let bell = Circuit::qubits(2, vec![
    put(vec![0], Gate::H),
    control(vec![0], vec![1], Gate::X),
]).unwrap();
let state = apply(&bell, &ArrayReg::zero_state(2));
let p = probs(&state, None);
assert!((p[0] - 0.5).abs() < 1e-12);
assert!((p[3] - 0.5).abs() < 1e-12);
```

Qubit 0 is the most significant bit of a state-vector index. Library features
are opt-in: `qasm` enables OpenQASM 2.0, `omeinsum` enables native tensor
contraction, and `parallel` enables Rayon operations. The CLI enables `qasm`
and `omeinsum` by default. The optional `tenferro` feature adds complex128 CPU
contraction with explicit threads and reusable plans (Rust 1.96 or newer).
Install from this checkout with `cargo install --path yao-cli --features tenferro --locked`,
then select `yao contract tn.json --backend tenferro --threads 4`.
See [tensor-network execution](docs/src/tensor-networks.md#tenferro-cpu-execution).

Enable `tenferro-ad` for custom-loss circuit differentiation. The optional
`optimizer-example` feature runs an argmin L-BFGS state-fitting example:
`cargo run --release --example custom_loss --features optimizer-example`.
See [differentiation](docs/src/differentiation.md) for cotangent conventions,
parameter sharing, and CPU execution/retention boundaries.

Pauli Hamiltonians, Ising/Heisenberg models, product-formula evolution and shared
physical parameter gradients are available in the [Hamiltonian guide](docs/src/hamiltonian.md).

## CLI

Install the published CLI below. To build the development version instead,
clone this repository and run `cargo install --path yao-cli --locked`.
The project uses Rust edition 2024; use a current stable Rust toolchain.


```bash
# Install
cargo install yao-cli --locked

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

[Full Rust API reference](https://docs.rs/yao-rs) ·
[Differentiable simulation](https://giggleliu.github.io/yao-rs/differentiation.html)

See the [mdBook documentation](https://giggleliu.github.io/yao-rs/) for detailed guides, including the [CLI guide](https://giggleliu.github.io/yao-rs/cli.html), the [Getting Started guide](https://giggleliu.github.io/yao-rs/getting-started.html), and the [QFT walkthrough](https://giggleliu.github.io/yao-rs/examples/qft.html).

## Development

Run `make check-all` for workspace formatting, Clippy, and all-feature tests.
Run `make doc` to build the book and full API docs (requires mdBook 0.5.2).
See [CONTRIBUTING.md](CONTRIBUTING.md) for feature checks and the
[release guide](RELEASING.md) for packaging and publishing.

Licensed under the [MIT license](LICENSE).
