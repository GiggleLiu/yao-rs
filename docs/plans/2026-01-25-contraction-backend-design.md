# Contraction Backend & PyTorch Support Design (Issue #2)

VectorMode tensor network contraction with boundary conditions and libtorch-based contractor.

## Boundary Conditions

Extend tensor network export to support fixed initial/final states:

```rust
pub fn circuit_to_einsum_with_boundary(
    circuit: &Circuit,
    final_state: &[usize],  // qubit indices pinned to |0⟩ in output
) -> TensorNetwork
```

### Semantics

- **Initial state**: always |0...0⟩. Each qubit gets a rank-1 tensor `[1, 0]` (or `[1, 0, ..., 0]` for dimension d) attached to its input leg.
- **Final state**: qubits listed in `final_state` get a rank-1 tensor `[1, 0]` on their output leg. Other qubits remain as open legs.
- All pinned → scalar result (amplitude `⟨0|C|0⟩`)
- None pinned → full output state tensor

### Implementation

1. For each qubit i, add a tensor `[1, 0, ..., 0]` (length = dims[i]) with a single leg matching the input label of qubit i.
2. For each qubit in `final_state`, add a tensor `[1, 0, ..., 0]` with a single leg matching the output label of qubit i.
3. Unpinned output labels become the open indices of the EinCode.

## Libtorch Contractor

Feature-gated behind `torch` feature flag.

### Cargo.toml

```toml
[features]
torch = ["tch"]

[dependencies]
tch = { version = "0.17", optional = true }
```

### API

```rust
#[cfg(feature = "torch")]
pub fn contract(tn: &TensorNetwork, device: tch::Device) -> tch::Tensor
```

User passes `tch::Device::Cpu` or `tch::Device::Cuda(0)` explicitly.

### Implementation

1. Convert each `ArrayD<Complex64>` tensor to `tch::Tensor` (complex64) on the specified device
2. Walk the optimized contraction tree from omeco's `EinCode`
3. Execute each pairwise contraction using `tch::Tensor::einsum`
4. Return the final result tensor

The contraction tree from omeco provides the binary execution order. Each step is a pairwise einsum of two tensors.

## Testing

Reproduce YaoToEinsum test patterns:

1. **Basic gates**: For each gate type (X, Y, Z, H, CNOT, controlled-Y, SWAP, custom matrix), build a circuit, contract with all-zero boundary, compare amplitude against `apply()` result
2. **QFT circuit**: `circuit_to_einsum_with_boundary` on QFT with all qubits pinned, verify `⟨0|QFT|0⟩` amplitude
3. **Open legs**: partial pinning — pin some final qubits, leave others open, verify output tensor matches `apply()` projected onto the pinned subspace
4. **EasyBuild circuits**: variational circuit, Google53 (small instance), verify contraction matches `apply()`
5. **Torch contractor** (`#[cfg(feature = "torch")]`): same tests via `contract()` on CPU, verify results match ndarray-based contraction

## mdBook Chapter (`docs/src/torch-interop.md`)

Contents:
- How to install libtorch and enable the `torch` feature
- Example: build a circuit, export to tensor network, contract on CPU/GPU
- Performance comparison: small circuit on CPU vs GPU
- How tch-rs tensors can be used with PyTorch models (shared memory, no-copy)
