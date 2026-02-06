//! Quantum measurement operations.
//!
//! This module provides functions for measuring quantum states in the computational basis.
//!
//! # Overview
//!
//! - [`probs`] - Get probability distribution over computational basis
//! - [`measure`] - Sample measurement outcomes without collapsing state
//! - [`measure_and_collapse`] - Measure and collapse state to the outcome
//! - [`collapse_to`] - Collapse state to a specific outcome (post-selection)

use num_complex::Complex64;
use rand::Rng;

use crate::index::{linear_to_indices, mixed_radix_index};
use crate::state::State;

/// Compute probability distribution over computational basis.
///
/// If `locs` is `None`, returns probabilities for all qudits.
/// If `locs` is `Some(&[...])`, returns marginal probabilities for specified qudits.
///
/// # Example
/// ```
/// use yao_rs::{State, measure::probs};
///
/// // |+⟩ state has 50% probability for |0⟩ and |1⟩
/// let state = State::new(
///     vec![2],
///     ndarray::array![
///         num_complex::Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
///         num_complex::Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
///     ],
/// );
/// let p = probs(&state, None);
/// assert!((p[0] - 0.5).abs() < 1e-10);
/// assert!((p[1] - 0.5).abs() < 1e-10);
/// ```
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

/// Compute marginal probabilities for a subset of qudits.
fn marginal_probs(state: &State, locs: &[usize]) -> Vec<f64> {
    let marginal_dims: Vec<usize> = locs.iter().map(|&i| state.dims[i]).collect();
    let marginal_size: usize = marginal_dims.iter().product();
    let mut prob_vec = vec![0.0; marginal_size];

    let total_dim: usize = state.dims.iter().product();
    for flat_idx in 0..total_dim {
        let indices = linear_to_indices(flat_idx, &state.dims);
        let marginal_indices: Vec<usize> = locs.iter().map(|&i| indices[i]).collect();
        let marginal_idx = mixed_radix_index(&marginal_indices, &marginal_dims);
        prob_vec[marginal_idx] += state.data[flat_idx].norm_sqr();
    }

    prob_vec
}

/// Sample an index from a probability distribution.
fn sample_from_probs(probs: &[f64], rng: &mut impl Rng) -> usize {
    let r: f64 = rng.r#gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

/// Sample measurement outcomes without collapsing state.
///
/// Returns a vector of measurement results. Each result is a vector of qudit values.
///
/// # Arguments
/// * `state` - The quantum state to measure
/// * `locs` - Which qudits to measure (`None` for all)
/// * `nshots` - Number of measurement samples
/// * `rng` - Random number generator
///
/// # Example
/// ```
/// use yao_rs::{State, measure::measure};
/// use rand::SeedableRng;
///
/// let state = State::zero_state(&[2, 2]); // |00⟩
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let results = measure(&state, None, 10, &mut rng);
///
/// // |00⟩ always measures to [0, 0]
/// for result in results {
///     assert_eq!(result, vec![0, 0]);
/// }
/// ```
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

/// Measure and collapse state to the measured outcome.
///
/// This modifies the state in-place, collapsing it to the measured computational
/// basis state (with renormalization).
///
/// # Arguments
/// * `state` - The quantum state to measure (modified in-place)
/// * `locs` - Which qudits to measure (`None` for all)
/// * `rng` - Random number generator
///
/// # Returns
/// The measurement result as a vector of qudit values.
///
/// # Example
/// ```
/// use yao_rs::{State, Gate, Circuit, put, apply, measure::measure_and_collapse};
/// use rand::SeedableRng;
///
/// // Create |+⟩ state
/// let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
/// let mut state = apply(&circuit, &State::zero_state(&[2]));
///
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let result = measure_and_collapse(&mut state, None, &mut rng);
///
/// // State is now collapsed to |0⟩ or |1⟩
/// assert!(result == vec![0] || result == vec![1]);
/// ```
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

/// Collapse state to a specific outcome (post-selection).
///
/// This sets all amplitudes that don't match the specified outcome to zero,
/// then renormalizes the remaining amplitudes.
///
/// # Arguments
/// * `state` - The quantum state to collapse (modified in-place)
/// * `locs` - Which qudits to fix
/// * `values` - The values to fix them to
///
/// # Panics
/// Panics if `locs` and `values` have different lengths.
///
/// # Example
/// ```
/// use yao_rs::{State, Gate, Circuit, put, apply, measure::{collapse_to, probs}};
///
/// // Create Bell state (|00⟩ + |11⟩)/√2
/// let circuit = Circuit::new(
///     vec![2, 2],
///     vec![put(vec![0], Gate::H), yao_rs::control(vec![0], vec![1], Gate::X)],
/// ).unwrap();
/// let mut state = apply(&circuit, &State::zero_state(&[2, 2]));
///
/// // Collapse first qubit to |0⟩
/// collapse_to(&mut state, &[0], &[0]);
///
/// // Now state is |00⟩
/// let p = probs(&state, None);
/// assert!((p[0] - 1.0).abs() < 1e-10); // |00⟩ has probability 1
/// ```
pub fn collapse_to(state: &mut State, locs: &[usize], values: &[usize]) {
    assert_eq!(
        locs.len(),
        values.len(),
        "locs and values must have the same length"
    );

    let total_dim: usize = state.dims.iter().product();
    let mut norm_sq = 0.0;

    // Zero out non-matching amplitudes and compute norm
    for flat_idx in 0..total_dim {
        let indices = linear_to_indices(flat_idx, &state.dims);
        let matches = locs
            .iter()
            .zip(values.iter())
            .all(|(&loc, &val)| indices[loc] == val);

        if matches {
            norm_sq += state.data[flat_idx].norm_sqr();
        } else {
            state.data[flat_idx] = Complex64::new(0.0, 0.0);
        }
    }

    // Renormalize
    let norm = norm_sq.sqrt();
    if norm > 1e-15 {
        for amp in state.data.iter_mut() {
            *amp /= norm;
        }
    }
}
