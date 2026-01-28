# Tensor Networks

## What is a Tensor Network?

A tensor network represents a quantum circuit as a collection of tensors connected by shared indices. Each gate becomes a tensor, and wires between gates become shared indices (labels). The circuit's output state is obtained by contracting (summing over) all internal indices -- this is equivalent to an einsum operation.

This representation is useful because:

- It separates the circuit's structure (the contraction pattern) from the gate data (the tensor values).
- Contraction order optimization can dramatically reduce computational cost.
- Diagonal gates naturally simplify to lower-rank tensors.

## The `circuit_to_einsum` Function

The entry point for tensor network export is `circuit_to_einsum`, which converts a `Circuit` into a `TensorNetwork`:

```rust
use yao_rs::{Circuit, Gate, put, control, circuit_to_einsum};

let circuit = Circuit::new(vec![2, 2], vec![
    put(vec![0], Gate::H),
    control(vec![0], vec![1], Gate::X),
]).unwrap();

let tn = circuit_to_einsum(&circuit);
```

## TensorNetwork Struct

The returned `TensorNetwork` has three fields:

- **`code: EinCode<usize>`** -- The einsum contraction code from omeco. Contains input index lists (one per tensor) and the output indices.
- **`tensors: Vec<ArrayD<Complex64>>`** -- The tensor data for each gate, stored as n-dimensional complex arrays.
- **`size_dict: HashMap<usize, usize>`** -- Maps each label to its dimension (e.g., 2 for qubits).

## Label Assignment Algorithm

The algorithm walks through the circuit gates in order, assigning integer labels to tensor legs:

1. **Initial labels** `0..n-1` represent the input state indices for each site (one label per site).

2. **For each gate in order:**
   - **Diagonal gate (no controls):** The tensor uses the current labels of its target sites. No new labels are allocated. Current labels are unchanged.
   - **Non-diagonal gate (or gate with controls):** New output labels are allocated for all involved sites (controls and targets). The tensor's indices are `[new_output_labels..., current_input_labels...]`. Current labels are updated to the new output labels.

3. **After all gates:** The final current labels become the output indices of the einsum.

Labels that appear in multiple tensors' index lists are internal (contracted) indices. Labels that only appear once and in the output are external (open) indices.

## Diagonal vs Non-Diagonal Gates

The `Leg` enum describes how each axis of a gate tensor is interpreted:

```rust
pub enum Leg {
    Out(usize),  // Output leg for site at given index
    In(usize),   // Input leg for site at given index
    Diag(usize), // Shared (diagonal) leg for site at given index
}
```

### Non-diagonal gates

Shape: `(d_0_out, d_1_out, ..., d_0_in, d_1_in, ...)`

The tensor has separate input and output legs for each involved site. This represents a general linear map. The legs are ordered as `[Out(0), Out(1), ..., In(0), In(1), ...]`.

### Diagonal gates (no controls)

Shape: `(d_0, d_1, ...)`

The tensor has one shared leg per target site (the diagonal elements only). Since the gate is diagonal, the input and output share the same index -- no new label is needed. The legs are all `[Diag(0), Diag(1), ...]`.

**Why this matters:** Fewer legs means simpler contraction and potentially better contraction orders. A single-qubit diagonal gate is a rank-1 tensor (a vector) instead of a rank-2 tensor (a matrix).

## Example: Bell Circuit

Tracing through a Bell circuit (H followed by CNOT on 2 qubits):

```
Sites: 0, 1        (both dimension 2)
Initial labels: [0, 1]

Gate 1: H on site 0 (non-diagonal)
  → Allocate new label 2 for site 0
  → Tensor indices: [2, 0] (out, in)
  → Current labels: [2, 1]

Gate 2: CNOT = control(0, 1, X) (non-diagonal, has controls)
  → All locs = [0, 1] (control 0, target 1)
  → Allocate new labels 3, 4 for sites 0, 1
  → Tensor indices: [3, 4, 2, 1] (out_0, out_1, in_0, in_1)
  → Current labels: [3, 4]

Output labels: [3, 4]
EinCode: [[2, 0], [3, 4, 2, 1]] → [3, 4]
```

The internal indices (2) connect H's output to CNOT's input on site 0. Index 1 connects the initial state of site 1 to CNOT's input on site 1. The output indices [3, 4] are the open legs of the final state.

## Using omeco for Contraction

The `EinCode` from omeco can be used to find optimal contraction orders:

```rust
use yao_rs::{Circuit, Gate, put, control, circuit_to_einsum};

let circuit = Circuit::new(vec![2, 2], vec![
    put(vec![0], Gate::H),
    control(vec![0], vec![1], Gate::X),
]).unwrap();

// The TensorNetwork's code field is an EinCode
let tn = circuit_to_einsum(&circuit);
println!("EinCode: {:?}", tn.code);
println!("Size dict: {:?}", tn.size_dict);
```

The `EinCode` encodes the full contraction pattern. You can use omeco's optimization methods to find efficient contraction orders before performing the actual tensor contraction.
