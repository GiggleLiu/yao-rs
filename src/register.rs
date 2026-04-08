use num_complex::Complex64;
use rand::{Rng, RngExt};

use crate::circuit::Circuit;

/// Core register interface shared by pure and mixed-state registers.
pub trait Register {
    fn nbits(&self) -> usize;
    fn apply(&mut self, circuit: &Circuit);
    fn state_data(&self) -> &[Complex64];
}

/// Non-batched qubit register backed by a dense state vector.
#[derive(Clone, Debug)]
pub struct ArrayReg {
    nbits: usize,
    pub state: Vec<Complex64>,
}

impl ArrayReg {
    pub fn zero_state(nbits: usize) -> Self {
        let dim = 1usize << nbits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        Self { nbits, state }
    }

    pub fn product_state<const N: usize>(bitstr: bitbasis::BitStr<N>) -> Self {
        Self {
            nbits: N,
            state: bitstr.onehot(),
        }
    }

    pub fn rand_state(nbits: usize, rng: &mut impl Rng) -> Self {
        let dim = 1usize << nbits;
        let mut state: Vec<Complex64> = (0..dim)
            .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
            .collect();
        let norm = state.iter().map(|amp| amp.norm_sqr()).sum::<f64>().sqrt();
        for amp in &mut state {
            *amp /= norm;
        }
        Self { nbits, state }
    }

    pub fn uniform_state(nbits: usize) -> Self {
        let dim = 1usize << nbits;
        let amp = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
        Self {
            nbits,
            state: vec![amp; dim],
        }
    }

    pub fn ghz_state(nbits: usize) -> Self {
        let dim = 1usize << nbits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        let amp = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        state[0] = amp;
        state[dim - 1] = amp;
        Self { nbits, state }
    }

    pub fn from_vec(nbits: usize, state: Vec<Complex64>) -> Self {
        assert_eq!(state.len(), 1usize << nbits);
        Self { nbits, state }
    }

    /// Deterministic benchmark state with amplitudes derived from the basis index.
    pub fn deterministic_state(nbits: usize) -> Self {
        let dim = 1usize << nbits;
        let mut state: Vec<Complex64> = (0..dim)
            .map(|k| {
                let kf = k as f64;
                Complex64::new((0.1 * kf).cos(), (0.2 * kf).sin())
            })
            .collect();
        let norm = state.iter().map(|amp| amp.norm_sqr()).sum::<f64>().sqrt();
        for amp in &mut state {
            *amp /= norm;
        }
        Self { nbits, state }
    }

    pub fn nqubits(&self) -> usize {
        self.nbits
    }

    pub fn state_vec(&self) -> &[Complex64] {
        &self.state
    }

    pub fn state_vec_mut(&mut self) -> &mut [Complex64] {
        &mut self.state
    }

    pub fn norm(&self) -> f64 {
        self.state
            .iter()
            .map(|amp| amp.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            for amp in &mut self.state {
                *amp /= norm;
            }
        }
    }

    pub fn fidelity(&self, other: &ArrayReg) -> f64 {
        assert_eq!(self.nbits, other.nbits);
        let inner: Complex64 = self
            .state
            .iter()
            .zip(other.state.iter())
            .map(|(lhs, rhs)| lhs.conj() * rhs)
            .sum();
        inner.norm_sqr()
    }
}

impl Register for ArrayReg {
    fn nbits(&self) -> usize {
        self.nbits
    }

    fn apply(&mut self, circuit: &Circuit) {
        crate::apply::apply_inplace(circuit, self);
    }

    fn state_data(&self) -> &[Complex64] {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_zero_state() {
        let reg = ArrayReg::zero_state(3);
        assert_eq!(reg.nqubits(), 3);
        assert_eq!(reg.state.len(), 8);
        assert_abs_diff_eq!(reg.state[0].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(reg.norm(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_uniform_state() {
        let reg = ArrayReg::uniform_state(2);
        for amp in reg.state_vec() {
            assert_abs_diff_eq!(amp.re, 0.5, epsilon = 1e-12);
        }
        assert_abs_diff_eq!(reg.norm(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ghz_state() {
        let reg = ArrayReg::ghz_state(3);
        let amp = 1.0 / 2.0_f64.sqrt();
        assert_abs_diff_eq!(reg.state[0].re, amp, epsilon = 1e-12);
        assert_abs_diff_eq!(reg.state[7].re, amp, epsilon = 1e-12);
        assert_abs_diff_eq!(reg.state[1].norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fidelity_same_state() {
        let reg = ArrayReg::zero_state(3);
        assert_abs_diff_eq!(reg.fidelity(&reg), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fidelity_orthogonal() {
        let r0 = ArrayReg::from_vec(1, vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let r1 = ArrayReg::from_vec(1, vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
        assert_abs_diff_eq!(r0.fidelity(&r1), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_deterministic_state() {
        let reg = ArrayReg::deterministic_state(4);
        assert_eq!(reg.nqubits(), 4);
        assert_eq!(reg.state.len(), 16);
        assert_abs_diff_eq!(reg.norm(), 1.0, epsilon = 1e-12);
        assert!((reg.state[0] - reg.state[1]).norm() > 1e-6);
    }
}
