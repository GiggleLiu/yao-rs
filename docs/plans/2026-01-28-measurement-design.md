# Quantum Measurement Design

## Goal

Add quantum measurement functionality to yao-rs with a simplified API compared to Yao.jl.

## API Design

### Core Functions

```rust
/// Sample measurement outcomes without collapsing state
/// Returns Vec of measurement results, each result is Vec<usize> of qudit values
pub fn measure(
    state: &State,
    locs: Option<&[usize]>,  // None = all qudits
    nshots: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<usize>>

/// Measure and collapse state to the measured outcome
/// Returns the measurement result
pub fn measure_and_collapse(
    state: &mut State,
    locs: Option<&[usize]>,
    rng: &mut impl Rng,
) -> Vec<usize>

/// Collapse state to a specific outcome (post-selection)
pub fn collapse_to(
    state: &mut State,
    locs: &[usize],
    values: &[usize],
)

/// Get probability distribution over computational basis
pub fn probs(state: &State, locs: Option<&[usize]>) -> Vec<f64>
```

### Comparison with Yao.jl

| Yao.jl | yao-rs | Notes |
|--------|--------|-------|
| `measure(reg; nshots)` | `measure(state, None, nshots, rng)` | Explicit RNG |
| `measure(reg, locs; nshots)` | `measure(state, Some(&locs), nshots, rng)` | |
| `measure!(reg)` | `measure_and_collapse(state, None, rng)` | |
| `measure!(RemoveMeasured(), reg)` | Not supported | Can manually resize |
| `measure!(ResetTo(val), reg)` | `measure_and_collapse` + `collapse_to` | Two steps |
| `probs(reg)` | `probs(state, None)` | |
| `select(reg, bits)` | Not needed | Direct indexing |
| `collapseto!(reg, val)` | `collapse_to(state, locs, values)` | |

### Removed Complexity

1. **No PostProcess enum** - separate functions instead
2. **No batch registers** - yao-rs doesn't have them
3. **No density matrix** - not implemented yet
4. **No focus/relax** - direct loc indexing
5. **No operator measurement** - tracked in issue #13

## Implementation

### File: `src/measure.rs`

```rust
use ndarray::Array1;
use num_complex::Complex64;
use rand::Rng;
use crate::state::State;
use crate::index::{mixed_radix_index, linear_to_indices};

/// Compute probability distribution over computational basis
pub fn probs(state: &State, locs: Option<&[usize]>) -> Vec<f64> {
    match locs {
        None => {
            // Full state probabilities
            state.data.iter().map(|c| c.norm_sqr()).collect()
        }
        Some(locs) => {
            // Marginal probabilities for specified qudits
            marginal_probs(state, locs)
        }
    }
}

/// Sample from probability distribution
fn sample_from_probs(probs: &[f64], rng: &mut impl Rng) -> usize {
    let r: f64 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

/// Measure without collapse
pub fn measure(
    state: &State,
    locs: Option<&[usize]>,
    nshots: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<usize>> {
    let p = probs(state, locs);
    let dims: Vec<usize> = match locs {
        None => state.dims.clone(),
        Some(locs) => locs.iter().map(|&i| state.dims[i]).collect(),
    };

    (0..nshots)
        .map(|_| {
            let idx = sample_from_probs(&p, rng);
            linear_to_indices(idx, &dims)
        })
        .collect()
}

/// Measure and collapse
pub fn measure_and_collapse(
    state: &mut State,
    locs: Option<&[usize]>,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let result = measure(state, locs, 1, rng).pop().unwrap();

    let locs_vec: Vec<usize> = match locs {
        None => (0..state.dims.len()).collect(),
        Some(l) => l.to_vec(),
    };

    collapse_to(state, &locs_vec, &result);
    result
}

/// Collapse to specific outcome
pub fn collapse_to(state: &mut State, locs: &[usize], values: &[usize]) {
    // Zero out amplitudes that don't match
    // Renormalize remaining amplitudes
    let total_dim: usize = state.dims.iter().product();
    let mut norm_sq = 0.0;

    for flat_idx in 0..total_dim {
        let indices = linear_to_indices(flat_idx, &state.dims);
        let matches = locs.iter().zip(values.iter())
            .all(|(&loc, &val)| indices[loc] == val);

        if matches {
            norm_sq += state.data[flat_idx].norm_sqr();
        } else {
            state.data[flat_idx] = Complex64::new(0.0, 0.0);
        }
    }

    // Renormalize
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        for amp in state.data.iter_mut() {
            *amp /= norm;
        }
    }
}
```

### Marginal Probabilities

For measuring a subset of qudits, we need marginal probabilities:

```rust
fn marginal_probs(state: &State, locs: &[usize]) -> Vec<f64> {
    let marginal_dims: Vec<usize> = locs.iter().map(|&i| state.dims[i]).collect();
    let marginal_size: usize = marginal_dims.iter().product();
    let mut probs = vec![0.0; marginal_size];

    let total_dim: usize = state.dims.iter().product();
    for flat_idx in 0..total_dim {
        let indices = linear_to_indices(flat_idx, &state.dims);
        let marginal_indices: Vec<usize> = locs.iter().map(|&i| indices[i]).collect();
        let marginal_idx = mixed_radix_index(&marginal_indices, &marginal_dims);
        probs[marginal_idx] += state.data[flat_idx].norm_sqr();
    }

    probs
}
```

## Testing

1. **probs tests**: Verify probabilities sum to 1, match expected values
2. **measure tests**: Statistical distribution matches probabilities
3. **measure_and_collapse tests**: State correctly collapsed
4. **collapse_to tests**: Post-selection works correctly
5. **Qudit tests**: Works for d > 2

## Exports

Add to `src/lib.rs`:
```rust
pub mod measure;
pub use measure::{probs, measure, measure_and_collapse, collapse_to};
```
