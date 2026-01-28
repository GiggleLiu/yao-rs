# Efficient State-Vector Simulation Design

## Goal

Replace the naive O(4^n) full-matrix `apply` function with an efficient O(2^n) in-place implementation, porting Yao.jl's `instruct!` approach to Rust with full qudit support.

## Background

### Current Implementation (src/apply.rs)

```rust
pub fn apply(circuit: &Circuit, state: &State) -> State {
    let full_matrix = circuit.full_matrix();  // O(4^n) - builds 2^n × 2^n matrix
    let new_data = full_matrix.dot(&state.data);
    State::new(state.dims.clone(), new_data)
}
```

**Problems:**
- Memory: O(4^n) for the full matrix
- Time: O(4^n) for matrix construction + O(8^n) for matrix-vector multiply
- Practical limit: ~12 qubits before memory exhaustion

### Yao.jl's Approach (instruct.jl)

Yao.jl applies gates by directly manipulating amplitudes:
- Single-qubit gate on qubit k: iterate over 2^(n-1) pairs of amplitudes
- Each pair differs only in bit k position
- Apply 2×2 unitary to each pair in-place

**Advantages:**
- Memory: O(2^n) - only the state vector
- Time: O(2^n) per gate
- Practical limit: 25+ qubits with 32GB RAM

## Design

### Core Types

```rust
// In src/instruct.rs (new file)

/// Apply a gate in-place to a state vector
pub fn instruct(
    state: &mut State,
    gate: &Gate,
    locs: &[usize],
    ctrl_locs: &[usize],
    ctrl_configs: &[usize],
) {
    // Dispatch based on gate properties
}
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Public API                               │
│  apply(circuit, state) -> State                             │
│  apply_inplace(circuit, state)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   instruct() dispatcher                      │
│  - Checks gate type (diagonal vs general)                   │
│  - Checks if controlled                                      │
│  - Dispatches to specialized implementation                  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ instruct_single  │ │instruct_diagonal │ │instruct_control  │
│                  │ │                  │ │                  │
│ General d×d gate │ │ Phase, Z, S, T   │ │ CNOT, Toffoli    │
│ on single site   │ │ Only diagonal    │ │ Controlled gates │
└──────────────────┘ └──────────────────┘ └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Index Iteration                            │
│  iter_basis_except(dims, locs) - iterate excluding locs     │
│  mixed_radix_index(indices, dims) - compute flat index      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Primitive Operations                       │
│  u1rows!(state, i, j, gate) - apply 2×2 to pair            │
│  udrows!(state, indices, gate) - apply d×d to d amplitudes │
│  mulrow!(state, i, factor) - multiply by scalar             │
└─────────────────────────────────────────────────────────────┘
```

### Qudit Indexing

For mixed-dimension systems (e.g., dims = [2, 3, 2] for qubit-qutrit-qubit):

```rust
/// Convert site indices to flat state vector index
/// Example: indices=[1,2,0], dims=[2,3,2] -> 1*6 + 2*2 + 0 = 10
fn mixed_radix_index(indices: &[usize], dims: &[usize]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for i in (0..dims.len()).rev() {
        idx += indices[i] * stride;
        stride *= dims[i];
    }
    idx
}

/// Iterate over all basis states, yielding (flat_index, site_indices)
fn iter_basis(dims: &[usize]) -> impl Iterator<Item = (usize, Vec<usize>)> {
    // ... mixed-radix counter
}

/// Iterate over basis states with fixed values at certain sites
fn iter_basis_fixed(
    dims: &[usize],
    fixed_locs: &[usize],
    fixed_vals: &[usize],
) -> impl Iterator<Item = usize> {
    // ... for controlled gates
}
```

### Single-Site Gate Application

For a d-dimensional site at location `loc`:

```rust
fn instruct_single(
    state: &mut State,
    gate: &Array2<Complex64>,  // d×d unitary
    loc: usize,
) {
    let d = state.dims[loc];
    let total_dim: usize = state.dims.iter().product();

    // For each basis state of the OTHER sites
    for other_basis in iter_basis_except(&state.dims, &[loc]) {
        // Gather d amplitudes (one for each value at loc)
        let mut indices = Vec::with_capacity(d);
        let mut amps = Vec::with_capacity(d);

        for val in 0..d {
            let idx = insert_index(other_basis, loc, val, &state.dims);
            indices.push(idx);
            amps.push(state.data[idx]);
        }

        // Apply d×d gate
        let new_amps = gate.dot(&Array1::from(amps));

        // Scatter back
        for (i, &idx) in indices.iter().enumerate() {
            state.data[idx] = new_amps[i];
        }
    }
}
```

### Diagonal Gate Optimization

Diagonal gates (Z, S, T, Phase, Rz) only multiply amplitudes by phase factors:

```rust
fn instruct_diagonal(
    state: &mut State,
    phases: &[Complex64],  // d phases for d-dimensional site
    loc: usize,
) {
    let d = state.dims[loc];

    // For each basis state
    for (flat_idx, site_indices) in iter_basis(&state.dims) {
        let site_val = site_indices[loc];
        state.data[flat_idx] *= phases[site_val];
    }
}
```

### Controlled Gate Application

For controlled gates, only apply when control sites have specified values:

```rust
fn instruct_controlled(
    state: &mut State,
    gate: &Array2<Complex64>,
    ctrl_locs: &[usize],
    ctrl_configs: &[usize],  // required values at control sites
    tgt_locs: &[usize],
) {
    // Only iterate over basis states where controls match
    for basis_idx in iter_basis_fixed(&state.dims, ctrl_locs, ctrl_configs) {
        // Apply gate to target sites
        apply_to_targets(state, gate, tgt_locs, basis_idx);
    }
}
```

### Parallelism with Rayon

Optional parallel iteration for large states:

```rust
#[cfg(feature = "parallel")]
fn instruct_single_parallel(
    state: &mut State,
    gate: &Array2<Complex64>,
    loc: usize,
) {
    use rayon::prelude::*;

    let d = state.dims[loc];
    let chunks = partition_basis_except(&state.dims, &[loc]);

    // Process chunks in parallel
    state.data.par_chunks_mut(chunk_size)
        .zip(chunks.par_iter())
        .for_each(|(chunk, basis_range)| {
            // Apply gate to this chunk
        });
}
```

Threshold for parallelism: ~14 qubits (16K amplitudes) based on Rayon overhead.

## API Changes

### Before

```rust
// Allocates new state
pub fn apply(circuit: &Circuit, state: &State) -> State;
```

### After

```rust
// In-place mutation (primary)
pub fn apply_inplace(circuit: &Circuit, state: &mut State);

// Convenience wrapper (clones then mutates)
pub fn apply(circuit: &Circuit, state: &State) -> State {
    let mut result = state.clone();
    apply_inplace(circuit, &mut result);
    result
}
```

## File Structure

```
src/
├── lib.rs          # Add: pub mod instruct;
├── instruct.rs     # NEW: Core instruct functions
├── index.rs        # NEW: Mixed-radix indexing utilities
├── apply.rs        # MODIFY: Use instruct internally
└── ...
```

## Testing Strategy

1. **Correctness tests**: Compare new `apply` output with old full-matrix approach for small circuits (n ≤ 8)
2. **Gate-specific tests**: Test each gate type (single, diagonal, controlled) individually
3. **Qudit tests**: Mixed-dimension systems (qubit + qutrit)
4. **Property tests**: Unitarity preservation, normalization

## Performance Expectations

| Qubits | Old (full matrix) | New (in-place) | Speedup |
|--------|-------------------|----------------|---------|
| 10     | ~1s               | ~1ms           | 1000x   |
| 15     | OOM               | ~30ms          | ∞       |
| 20     | OOM               | ~1s            | ∞       |
| 25     | OOM               | ~30s           | ∞       |

## Implementation Tasks

1. **Index utilities** - `mixed_radix_index`, `iter_basis`, `iter_basis_fixed`
2. **Primitive operations** - `udrows!` for d-dimensional gates
3. **Single-site instruct** - General and diagonal variants
4. **Controlled instruct** - With arbitrary control configurations
5. **Apply rewrite** - Use instruct internally
6. **Parallel support** - Rayon feature flag
7. **Benchmarks** - Compare old vs new, single vs parallel
