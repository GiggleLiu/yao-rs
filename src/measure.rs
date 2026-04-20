//! Quantum measurement operations.
//!
//! This module provides functions for measuring quantum states in the computational basis.
//!
//! # Overview
//!
//! - [`probs`] - Get probability distribution over computational basis
//! - [`measure_with_postprocess`] - Measure an [`ArrayReg`]
//!   with post-processing

use num_complex::Complex64;
use rand::{Rng, RngExt};

use crate::density_matrix::DensityMatrix;
use crate::register::{ArrayReg, Register};

/// Post-processing behavior for qubit-register measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PostProcess {
    NoPostProcess,
    ResetTo(usize),
    RemoveMeasured,
}

/// Result of measuring an [`ArrayReg`].
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
/// use yao_rs::{ArrayReg, measure::probs};
///
/// // |+⟩ state has 50% probability for |0⟩ and |1⟩
/// let s = 1.0 / 2.0_f64.sqrt();
/// let reg = ArrayReg::from_vec(1, vec![
///     num_complex::Complex64::new(s, 0.0),
///     num_complex::Complex64::new(s, 0.0),
/// ]);
/// let p = probs(&reg, None);
/// assert!((p[0] - 0.5).abs() < 1e-10);
/// assert!((p[1] - 0.5).abs() < 1e-10);
/// ```
pub fn probs<T: ProbabilitySource + ?Sized>(state: &T, locs: Option<&[usize]>) -> Vec<f64> {
    match locs {
        None => state.full_probs(),
        Some(locs) => state.marginal_probs(locs),
    }
}

fn marginal_probs_qubits(state: &[Complex64], locs: &[usize]) -> Vec<f64> {
    let nbits = state.len().trailing_zeros() as usize;
    let mut prob_vec = vec![0.0; 1usize << locs.len()];

    for (basis, amp) in state.iter().enumerate() {
        let mut marginal_idx = 0usize;
        for (idx, &loc) in locs.iter().enumerate() {
            let bit = logical_loc_to_bit(nbits, loc);
            let out_bit = locs.len() - 1 - idx;
            marginal_idx |= ((basis >> bit) & 1) << out_bit;
        }
        prob_vec[marginal_idx] += amp.norm_sqr();
    }

    prob_vec
}

fn marginal_probs_density_matrix(dm: &DensityMatrix, locs: &[usize]) -> Vec<f64> {
    let nbits = dm.nbits();
    let dim = 1usize << nbits;
    let mut prob_vec = vec![0.0; 1usize << locs.len()];

    for basis in 0..dim {
        let mut marginal_idx = 0usize;
        for (idx, &loc) in locs.iter().enumerate() {
            let bit = logical_loc_to_bit(nbits, loc);
            let out_bit = locs.len() - 1 - idx;
            marginal_idx |= ((basis >> bit) & 1) << out_bit;
        }
        prob_vec[marginal_idx] += dm.state_data()[basis * dim + basis].re.max(0.0);
    }

    prob_vec
}

fn validate_measure_locs(nbits: usize, locs: &[usize]) {
    let mut seen = std::collections::BTreeSet::new();
    for &loc in locs {
        assert!(
            loc < nbits,
            "measurement location {loc} is out of range for {nbits} qubits"
        );
        assert!(seen.insert(loc), "duplicate measurement location {loc}");
    }
}

fn logical_loc_to_bit(nbits: usize, loc: usize) -> usize {
    nbits - 1 - loc
}

/// Sample an index from a probability distribution.
fn sample_from_probs(probs: &[f64], rng: &mut impl Rng) -> usize {
    let r: f64 = rng.random();
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
    (0..nbits)
        .map(|idx| (outcome_idx >> (nbits - 1 - idx)) & 1)
        .collect()
}

fn collapse_qubits_to(state: &mut [Complex64], locs: &[usize], values: &[usize]) {
    assert_eq!(
        locs.len(),
        values.len(),
        "locs and values must have the same length"
    );

    let nbits = state.len().trailing_zeros() as usize;
    let mut norm_sq = 0.0;
    for (basis, amp) in state.iter_mut().enumerate() {
        let matches = locs
            .iter()
            .zip(values.iter())
            .all(|(&loc, &value)| ((basis >> logical_loc_to_bit(nbits, loc)) & 1) == value);
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

    let nbits = state.len().trailing_zeros() as usize;
    let mut swap_mask = 0usize;
    for (idx, &loc) in locs.iter().enumerate() {
        if from[idx] != to_bits[idx] {
            swap_mask |= 1usize << logical_loc_to_bit(nbits, loc);
        }
    }

    for basis in 0..state.len() {
        let matches_from = locs
            .iter()
            .zip(from.iter())
            .all(|(&loc, &value)| ((basis >> logical_loc_to_bit(nbits, loc)) & 1) == value);
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
    let kept_locs: Vec<usize> = (0..reg.nqubits())
        .filter(|loc| !locs.contains(loc))
        .collect();
    let mut new_state = vec![Complex64::new(0.0, 0.0); 1usize << kept_locs.len()];

    for (basis, amp) in reg.state_vec().iter().enumerate() {
        if amp.norm_sqr() < 1e-30 {
            continue;
        }

        let mut new_basis = 0usize;
        for (idx, &loc) in kept_locs.iter().enumerate() {
            let bit = logical_loc_to_bit(reg.nqubits(), loc);
            let out_bit = kept_locs.len() - 1 - idx;
            new_basis |= ((basis >> bit) & 1) << out_bit;
        }
        new_state[new_basis] += *amp;
    }

    let mut new_reg = ArrayReg::from_vec(kept_locs.len(), new_state);
    new_reg.normalize();
    new_reg
}

/// Measure an [`ArrayReg`] with optional post-processing.
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
