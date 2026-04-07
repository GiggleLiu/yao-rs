//! Quantum measurement operations.
//!
//! This module provides functions for measuring quantum states in the computational basis.
//!
//! # Overview
//!
//! - [`probs`] - Get probability distribution over computational basis
//! - [`measure`] - Sample measurement outcomes without collapsing state
//! - [`measure_and_collapse`] - Measure and collapse state to the outcome
//! - [`measure_with_postprocess`] - Measure an [`ArrayReg`](crate::register::ArrayReg)
//!   with post-processing
//! - [`collapse_to`] - Collapse state to a specific outcome (post-selection)

use num_complex::Complex64;
use rand::Rng;

use crate::density_matrix::DensityMatrix;
use crate::index::{linear_to_indices, mixed_radix_index};
use crate::register::{ArrayReg, Register};
use crate::state::State;

/// Post-processing behavior for qubit-register measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PostProcess {
    NoPostProcess,
    ResetTo(usize),
    RemoveMeasured,
}

/// Result of measuring an [`ArrayReg`](crate::register::ArrayReg).
#[derive(Clone, Debug)]
pub enum MeasureResult {
    Value(Vec<usize>),
    Removed(Vec<usize>, ArrayReg),
}

#[doc(hidden)]
pub trait ProbabilitySource {
    fn full_probs(&self) -> Vec<f64>;
    fn marginal_probs(&self, locs: &[usize]) -> Vec<f64>;
}

impl ProbabilitySource for State {
    fn full_probs(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.norm_sqr()).collect()
    }

    fn marginal_probs(&self, locs: &[usize]) -> Vec<f64> {
        marginal_probs_state(self, locs)
    }
}

impl ProbabilitySource for ArrayReg {
    fn full_probs(&self) -> Vec<f64> {
        self.state_vec().iter().map(|c| c.norm_sqr()).collect()
    }

    fn marginal_probs(&self, locs: &[usize]) -> Vec<f64> {
        validate_measure_locs(self.nqubits(), locs);
        marginal_probs_qubits(self.state_vec(), locs)
    }
}

impl ProbabilitySource for DensityMatrix {
    fn full_probs(&self) -> Vec<f64> {
        let dim = 1usize << self.nbits();
        (0..dim)
            .map(|basis| self.state_data()[basis * dim + basis].re.max(0.0))
            .collect()
    }

    fn marginal_probs(&self, locs: &[usize]) -> Vec<f64> {
        validate_measure_locs(self.nbits(), locs);
        marginal_probs_density_matrix(self, locs)
    }
}

/// Compute probability distribution over computational basis.
///
/// If `locs` is `None`, returns probabilities for all sites.
/// If `locs` is `Some(&[...])`, returns marginal probabilities for specified sites.
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
pub fn probs<T: ProbabilitySource + ?Sized>(state: &T, locs: Option<&[usize]>) -> Vec<f64> {
    match locs {
        None => state.full_probs(),
        Some(locs) => state.marginal_probs(locs),
    }
}

/// Compute marginal probabilities for a subset of qudits.
fn marginal_probs_state(state: &State, locs: &[usize]) -> Vec<f64> {
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

fn marginal_probs_qubits(state: &[Complex64], locs: &[usize]) -> Vec<f64> {
    let mut prob_vec = vec![0.0; 1usize << locs.len()];

    for (basis, amp) in state.iter().enumerate() {
        let mut marginal_idx = 0usize;
        for (idx, &loc) in locs.iter().enumerate() {
            marginal_idx |= ((basis >> loc) & 1) << idx;
        }
        prob_vec[marginal_idx] += amp.norm_sqr();
    }

    prob_vec
}

fn marginal_probs_density_matrix(dm: &DensityMatrix, locs: &[usize]) -> Vec<f64> {
    let dim = 1usize << dm.nbits();
    let mut prob_vec = vec![0.0; 1usize << locs.len()];

    for basis in 0..dim {
        let mut marginal_idx = 0usize;
        for (idx, &loc) in locs.iter().enumerate() {
            marginal_idx |= ((basis >> loc) & 1) << idx;
        }
        prob_vec[marginal_idx] += dm.state_data()[basis * dim + basis].re.max(0.0);
    }

    prob_vec
}

fn validate_measure_locs(nbits: usize, locs: &[usize]) {
    let mut seen = std::collections::BTreeSet::new();
    for &loc in locs {
        assert!(loc < nbits, "measurement location {loc} is out of range for {nbits} qubits");
        assert!(seen.insert(loc), "duplicate measurement location {loc}");
    }
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

fn decode_outcome_bits(outcome_idx: usize, nbits: usize) -> Vec<usize> {
    (0..nbits).map(|idx| (outcome_idx >> idx) & 1).collect()
}

fn collapse_qubits_to(state: &mut [Complex64], locs: &[usize], values: &[usize]) {
    assert_eq!(
        locs.len(),
        values.len(),
        "locs and values must have the same length"
    );

    let mut norm_sq = 0.0;
    for (basis, amp) in state.iter_mut().enumerate() {
        let matches = locs
            .iter()
            .zip(values.iter())
            .all(|(&loc, &value)| ((basis >> loc) & 1) == value);
        if matches {
            norm_sq += amp.norm_sqr();
        } else {
            *amp = Complex64::new(0.0, 0.0);
        }
    }

    let norm = norm_sq.sqrt();
    if norm > 1e-15 {
        for amp in state.iter_mut() {
            *amp /= norm;
        }
    }
}

fn reset_qubits_to(state: &mut [Complex64], locs: &[usize], from: &[usize], reset_val: usize) {
    let to_bits = decode_outcome_bits(reset_val, locs.len());
    if from == &to_bits[..] {
        return;
    }

    let mut swap_mask = 0usize;
    for (idx, &loc) in locs.iter().enumerate() {
        if from[idx] != to_bits[idx] {
            swap_mask |= 1usize << loc;
        }
    }

    for basis in 0..state.len() {
        let matches_from = locs
            .iter()
            .zip(from.iter())
            .all(|(&loc, &value)| ((basis >> loc) & 1) == value);
        if !matches_from {
            continue;
        }

        let target_basis = basis ^ swap_mask;
        let amp = state[basis];
        state[target_basis] = amp;
        state[basis] = Complex64::new(0.0, 0.0);
    }
}

fn remove_measured_qubits(reg: &ArrayReg, locs: &[usize]) -> ArrayReg {
    let kept_locs: Vec<usize> = (0..reg.nqubits()).filter(|loc| !locs.contains(loc)).collect();
    let mut new_state = vec![Complex64::new(0.0, 0.0); 1usize << kept_locs.len()];

    for (basis, amp) in reg.state_vec().iter().enumerate() {
        if amp.norm_sqr() < 1e-30 {
            continue;
        }

        let mut new_basis = 0usize;
        for (idx, &loc) in kept_locs.iter().enumerate() {
            new_basis |= ((basis >> loc) & 1) << idx;
        }
        new_state[new_basis] += *amp;
    }

    let mut new_reg = ArrayReg::from_vec(kept_locs.len(), new_state);
    new_reg.normalize();
    new_reg
}

/// Measure an [`ArrayReg`](crate::register::ArrayReg) with optional post-processing.
pub fn measure_with_postprocess(
    reg: &mut ArrayReg,
    locs: &[usize],
    post: PostProcess,
    rng: &mut impl Rng,
) -> MeasureResult {
    validate_measure_locs(reg.nqubits(), locs);

    let probs = marginal_probs_qubits(reg.state_vec(), locs);
    let outcome_idx = sample_from_probs(&probs, rng);
    let outcome = decode_outcome_bits(outcome_idx, locs.len());

    match post {
        PostProcess::NoPostProcess => MeasureResult::Value(outcome),
        PostProcess::ResetTo(reset_val) => {
            assert!(
                reset_val < (1usize << locs.len()),
                "reset value {} does not fit in {} measured qubits",
                reset_val,
                locs.len()
            );
            collapse_qubits_to(reg.state_vec_mut(), locs, &outcome);
            reset_qubits_to(reg.state_vec_mut(), locs, &outcome, reset_val);
            MeasureResult::Value(outcome)
        }
        PostProcess::RemoveMeasured => {
            collapse_qubits_to(reg.state_vec_mut(), locs, &outcome);
            let new_reg = remove_measured_qubits(reg, locs);
            MeasureResult::Removed(outcome, new_reg)
        }
    }
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

/// Measure specified qudits and reset them to a given value.
///
/// After measuring, collapses the state and then swaps amplitudes so the
/// measured qudits are in the `reset_val` basis state.
///
/// # Safety invariant
/// The swap loop is safe because after `measure_and_collapse`, only amplitudes
/// matching the measurement result are nonzero. Since `result != reset_val` on
/// at least one loc (early return handles the equal case), the source set
/// (indices matching result) and target set (indices matching reset_val) are disjoint.
///
/// Julia: `measure!(YaoAPI.ResetTo(val), reg, locs)`
pub fn measure_reset(
    state: &mut State,
    locs: &[usize],
    reset_val: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let result = measure_and_collapse(state, Some(locs), rng);

    // If already in target state, nothing to do
    if result.iter().all(|&v| v == reset_val) {
        return result;
    }

    // Swap amplitudes from measured values to reset values
    let total_dim: usize = state.dims.iter().product();
    for flat_idx in 0..total_dim {
        let indices = linear_to_indices(flat_idx, &state.dims);

        // Check if this index has the measured values at locs
        let matches_measured = locs
            .iter()
            .zip(result.iter())
            .all(|(&l, &v)| indices[l] == v);
        if matches_measured {
            let mut target_indices = indices.clone();
            for &l in locs {
                target_indices[l] = reset_val;
            }
            let target_flat = mixed_radix_index(&target_indices, &state.dims);
            state.data[target_flat] = state.data[flat_idx];
            state.data[flat_idx] = Complex64::new(0.0, 0.0);
        }
    }

    result
}

/// Measure specified qudits and remove them from the state.
///
/// Returns (measurement_result, new_smaller_state).
///
/// Julia: `measure!(YaoAPI.RemoveMeasured(), reg, locs)`
pub fn measure_remove(state: &State, locs: &[usize], rng: &mut impl Rng) -> (Vec<usize>, State) {
    let result = measure(state, Some(locs), 1, rng).pop().unwrap();

    // Build new dims without the measured qudits
    let remaining_locs: Vec<usize> = (0..state.dims.len())
        .filter(|i| !locs.contains(i))
        .collect();
    let new_dims: Vec<usize> = remaining_locs.iter().map(|&i| state.dims[i]).collect();
    let new_total: usize = new_dims.iter().product();

    let mut new_data = ndarray::Array1::zeros(new_total);

    let total_dim: usize = state.dims.iter().product();
    let mut norm_sq = 0.0;

    for flat_idx in 0..total_dim {
        let indices = linear_to_indices(flat_idx, &state.dims);

        // Check if measured qudits match the result
        let matches = locs
            .iter()
            .zip(result.iter())
            .all(|(&l, &v)| indices[l] == v);
        if !matches {
            continue;
        }

        // Compute new index without measured qudits
        let new_indices: Vec<usize> = remaining_locs.iter().map(|&i| indices[i]).collect();
        let new_flat = mixed_radix_index(&new_indices, &new_dims);
        new_data[new_flat] = state.data[flat_idx];
        norm_sq += state.data[flat_idx].norm_sqr();
    }

    // Normalize
    let norm = norm_sq.sqrt();
    if norm > 1e-15 {
        new_data.mapv_inplace(|v| v / norm);
    }

    (result, State::new(new_dims, new_data))
}
