# Tensor networks

A tensor network breaks an experiment into small arrays connected by shared
indices. Contracting the network combines those arrays to compute a result.
Choose the result you need first, then optimize the order of computation.

## Export, optimize, contract

Starting with a circuit file, compute its output state from the all-zero input:

```bash
yao toeinsum circuit.json --mode state | yao optimize - | yao contract - --json
```

Each command handles one step:

1. `toeinsum` builds the network and attaches the requested boundaries.
2. `optimize` chooses a contraction order.
3. `contract` evaluates the network with that order.

You can save any intermediate JSON file with `--output` to inspect or reuse it.
Native contraction requires `omeinsum` (included in the default CLI) or the
optional `tenferro` feature.

## Choose the result

| Desired result | Export options | Boundary conditions |
|---|---|---|
| Full output state | `--mode state` | All-zero input, open outputs |
| All-zero return amplitude | `--mode overlap` | All-zero input and output |
| Expectation value | `--op "Z(0)Z(1)"` | All-zero input, observable or sum of observables at the output |
| Gate tensors for further processing | `--mode pure` (default) | No explicit state boundary tensors |

An overlap is a complex amplitude. Its squared magnitude is the probability
of returning to the all-zero state. Prefer an overlap or expectation value
when you need a scalar; requesting the full state still requires storing
all output amplitudes.

## Use from Rust

Build a network with explicit state boundaries:

```rust
use yao_rs::{Circuit, Gate, circuit_to_einsum_with_boundary, put};

let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::H)]).unwrap();
let network = circuit_to_einsum_with_boundary(&circuit, &[]);
```

The empty slice leaves every output open. Listing output sites pins those
sites to state 0 instead. `circuit_to_overlap` pins all outputs;
`circuit_to_expectation` inserts an observable.

With the [`omeinsum` feature](installation.md#optional-features), contract directly:

```rust
use yao_rs::{Circuit, Gate, circuit_to_einsum_with_boundary, contract_tn, put};

let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::H)]).unwrap();
let network = circuit_to_einsum_with_boundary(&circuit, &[]);
let state = contract_tn(&network);
assert_eq!(state.len(), 2);
```

`contract_tn` chooses a greedy contraction order automatically. A network also
exposes its tensors, index pattern (`code`), and index dimensions (`size_dict`)
for custom processing. See the [tensor network API](api/yao_rs/einsum/index.html)
for all export functions.

## Control computation cost

Contraction cost depends on the circuit's connectivity and the chosen order.
The default greedy optimizer is a quick starting point. To spend more time
searching for a better order:

```bash
yao toeinsum circuit.json --mode state --output network.json
yao optimize network.json --method treesa --ntrials 20 --output optimized.json
yao contract optimized.json
```

Diagonal gates share their input and output indices, reducing tensor rank.
This applies to controlled diagonal gates too. The exporter performs this
simplification automatically.

## Noise and density matrices

Circuits containing [noise channels](conventions.md#gate-elements) need a
density-matrix network:

```bash
yao toeinsum noisy.json --mode dm | yao optimize - | yao contract - --json
```

From Rust, use `circuit_to_einsum_dm` or `circuit_to_expectation_dm` and
`contract_dm`. These paths represent both sides of the density matrix.

Tensor export also supports qudit circuits built with custom gates. The
[format reference](conventions.md#tensor-network-json) documents the CLI's
serialized network representation.

## Tenferro CPU execution

Enable `yao-rs`'s optional `tenferro` feature with Rust 1.96 or newer. The adapter
uses published tenferro 0.4.0 and its faer CPU provider. It supports the same
complex128 tensor exports as the existing contractor, including mixed qudit
dimensions and density matrices with negative bra labels.

```rust
use yao_rs::{Circuit, Gate, put, circuit_to_einsum_with_boundary};
use yao_rs::tenferro::CpuContractor;

let circuit = Circuit::qubits(2, vec![put(vec![1], Gate::Ry(0.3))]).unwrap();
let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
let cpu = CpuContractor::new(4).unwrap();
let plan = cpu.prepare(&tn.code, &tn.size_dict, None).unwrap();
let state = cpu.execute(&plan, &tn.tensors).unwrap();
// Reuse `plan` for new values with exactly the same tensor shapes.
let state_again = cpu.execute(&plan, &tn.tensors).unwrap();
assert_eq!(state, state_again);
```

`prepare` chooses a deterministic omeco greedy tree when passed `None`. Pass
`Some(&tree)` to preserve an existing omeco contraction tree. Every node's
intermediate outputs and grouping are retained; an n-ary node contracts its
children left-to-right. Repeated input labels implement traces/diagonals.
An empty network returns scalar one. Invalid trees, missing/zero dimensions,
overflowing complex128 storage sizes, and changed input shapes return errors.

The context owns its CPU runtime and thread configuration; each plan owns its
compiled graph, independently of tensor values. Keep both alive for repeated
work and drop unneeded plans. No global cache is maintained by yao-rs. Avoid
calling the adapter inside another tenferro backend session or oversubscribing
CPU threads with an outer parallel loop.

Contiguous C/Fortran arrays and axis permutations are borrowed with their actual
strides. Negative-stride and noncontiguous inputs are materialized once at the
adapter boundary; backend kernels may also pack inputs. Returned arrays own
their data, with logical axes in `code.iy` order. Array iteration follows those
logical axes: qubit 0 remains the most significant site regardless of storage
layout. Tenferro tensor/runtime types stay behind the adapter.

This is CPU tensor contraction. Native state-vector application and existing
expectation gradients continue to use their specialized kernels. For device-resident GPU execution, see [CUDA simulation](cuda.md). [Circuit differentiation](differentiation.md)
and [memory controls with multiple observables](tensor-memory.md) are available.
The [CPU measurements](https://github.com/GiggleLiu/yao-rs/tree/main/benchmarks/results)
separate conversion/planning costs from prepared execution; tenferro is not
universally faster and does not replace the default provider in this release.
