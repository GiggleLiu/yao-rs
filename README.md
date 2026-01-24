# yao-rs

[![CI](https://github.com/GiggleLiu/yao-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/GiggleLiu/yao-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GiggleLiu/yao-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/yao-rs)
[![Docs](https://github.com/GiggleLiu/yao-rs/actions/workflows/docs.yml/badge.svg)](https://giggleliu.github.io/yao-rs/)

A Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl) focused on quantum circuit description and tensor network export.

## Features

- **Gate enum** with named qubit gates (X, Y, Z, H, S, T, SWAP, Rx, Ry, Rz) and custom qudit gates
- **Qudit support** with per-site dimensions
- **Circuit validation** with controlled gates (qubit-only controls)
- **Tensor network export** via [omeco](https://crates.io/crates/omeco) for contraction order optimization
- **Diagonal gate optimization** in tensor networks (shared legs vs input/output legs)
- **Generic apply** for correctness verification

## Quick Start

```rust
use yao_rs::{Gate, PositionedGate, Circuit, State, apply, circuit_to_einsum};

// Build a Bell circuit: H on qubit 0, then CNOT(0→1)
let gates = vec![
    PositionedGate::new(Gate::H, vec![0], vec![], vec![]),
    PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]),
];
let circuit = Circuit::new(vec![2, 2], gates).unwrap();

// Apply to |00⟩
let state = State::zero_state(&[2, 2]);
let result = apply(&circuit, &state);

// Or export as tensor network
let tn = circuit_to_einsum(&circuit);
```

## Documentation

See the [mdBook documentation](https://giggleliu.github.io/yao-rs/) for detailed guides.
