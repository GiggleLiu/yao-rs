# yao-rs

Quantum circuit description and tensor network export in Rust.

## What is yao-rs?

yao-rs is a library for describing quantum circuits and exporting them as tensor networks. It provides a type-safe circuit construction API with validation, and converts circuits into einsum representations suitable for contraction order optimization via [omeco](https://crates.io/crates/omeco).

Ported from the Julia library [Yao.jl](https://github.com/QuantumBFS/Yao.jl), focused on the circuit description and tensor network layers.

## Module Architecture

<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>

<div id="module-graph"></div>
<div id="mg-controls">
  <div id="mg-legend">
    <span class="swatch" style="background:#c8f0c8;"></span>Core
    <span class="swatch" style="background:#c8c8f0;"></span>Simulation
    <span class="swatch" style="background:#f0f0a0;"></span>Tensor Export
    <span class="swatch" style="background:#e0e0e0;"></span>Utilities
    <span class="swatch" style="background:#e0c8f0;"></span>Higher-level
    <span class="swatch" style="background:#f0c8e0;"></span>Visualization
  </div>
</div>
<div id="mg-help">
  Click a module to expand/collapse its public items.
  Double-click to open rustdoc.
</div>
<div id="mg-tooltip"></div>

## Why Tensor Network Export?

Tensor networks provide an alternative to full state-vector simulation. Instead of tracking the entire 2^n-dimensional state vector, a circuit is decomposed into a network of small tensors. The contraction order determines computational cost — and can make an exponential difference:

| Approach | Memory | Scaling |
|----------|--------|---------|
| State vector | O(2^n) | Exponential in qubits |
| Tensor network | Depends on order | Can be much better for structured circuits |

yao-rs further optimizes by recognizing diagonal gates (Z, S, T, Phase, Rz), which reduce tensor rank in the network.

## Key Features

- **Circuit Description**: `put`/`control` builder API with qudit support
- **Tensor Network Export**: `circuit_to_einsum` with diagonal gate optimization
- **Contraction Optimization**: Integration with [omeco](https://crates.io/crates/omeco)
- **State-Vector Simulation**: Direct `apply` for verification
- **CLI Tool**: `yao` command for simulation, measurement, and tensor export without writing Rust code

## Example

```rust
use yao_rs::{Gate, Circuit, ArrayReg, put, control, apply, circuit_to_einsum};

// Build a Bell circuit
let circuit = Circuit::new(vec![2, 2], vec![
    put(vec![0], Gate::H),
    control(vec![0], vec![1], Gate::X),
]).unwrap();

// Simulate
let reg = ArrayReg::zero_state(2);
let result = apply(&circuit, &reg);

// Export as tensor network
let tn = circuit_to_einsum(&circuit);
println!("Tensors: {}, Labels: {}", tn.tensors.len(), tn.size_dict.len());
```

## Next Steps

- [Getting Started](./getting-started.md) - Install yao-rs and build your first circuit
- [CLI Tool](./cli.md) - Use the `yao` command-line tool
- [Gates](./gates.md) - All gate variants and their properties
- [Tensor Networks](./tensor-networks.md) - Understand the einsum export
