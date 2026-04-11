# Getting Started

This guide walks you through installing yao-rs and building your first quantum circuit.

> **Prefer the command line?** The [`yao` CLI tool](./cli.md) lets you simulate circuits and compute measurements without writing Rust code.

## Installation

Add yao-rs to your project's `Cargo.toml`:

```toml
[dependencies]
yao-rs = { git = "https://github.com/QuantumBFS/yao-rs" }
```

The crate uses Rust edition 2024 and depends on:

- `num-complex` for complex number arithmetic
- `ndarray` for multi-dimensional arrays
- `omeco` for tensor network contraction

## Your First Circuit: Bell State

Let's build a Bell circuit that entangles two qubits. The circuit applies a Hadamard gate on qubit 0, followed by a CNOT gate with qubit 0 as control and qubit 1 as target:

```rust
use yao_rs::{Gate, Circuit, ArrayReg, put, control, apply};

fn main() {
    // Build a Bell circuit: H on qubit 0, then CNOT
    let gates = vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();

    // Apply to |00⟩
    let reg = ArrayReg::zero_state(2);
    let result = apply(&circuit, &reg);

    // Print amplitudes
    for (i, amp) in result.state_vec().iter().enumerate() {
        if amp.norm() > 1e-10 {
            println!("|{:02b}⟩: {:.4} + {:.4}i", i, amp.re, amp.im);
        }
    }
}
```

This produces the Bell state (|00> + |11>)/sqrt(2), one of the four maximally entangled two-qubit states. You should see non-zero amplitudes only for the |00> and |11> basis states, each with magnitude 1/sqrt(2).

## Rendering as SVG

Once you have a `Circuit`, you can render it directly with the built-in SVG backend:

```rust
let svg = circuit.to_svg();
std::fs::write("bell.svg", svg).unwrap();
```

If you want a complete end-to-end example with annotations, see [`examples/circuit_svg.rs`](https://github.com/GiggleLiu/yao-rs/blob/main/examples/circuit_svg.rs).

From the CLI, the same workflow is:

```bash
yao example bell > bell.json
yao visualize bell.json --output bell.svg
```

## Exporting as a Tensor Network

yao-rs can export a circuit as a tensor network using Einstein summation notation. This is useful for analyzing circuit structure or contracting the network with custom strategies:

```rust
use yao_rs::circuit_to_einsum;

let tn = circuit_to_einsum(&circuit);
println!("Tensors: {}", tn.tensors.len());
println!("Labels: {:?}", tn.size_dict);
```

The returned tensor network contains the gate tensors and their index labels, along with a size dictionary mapping each index to its dimension.

## Running the QFT Example

The repository includes a Quantum Fourier Transform example that you can run directly:

```bash
cargo run --example qft
```

This builds a 4-qubit QFT circuit and applies it to various input states, demonstrating how the QFT maps computational basis states to uniform superpositions with structured phases.
