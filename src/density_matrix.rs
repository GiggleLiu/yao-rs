use ndarray::Array2;
use num_complex::Complex64;

use crate::circuit::Circuit;
use crate::register::{ArrayReg, Register};

/// Density matrix for a qubit register, stored row-major.
#[derive(Clone, Debug)]
pub struct DensityMatrix {
    nbits: usize,
    pub state: Vec<Complex64>,
}

impl DensityMatrix {
    fn dim(&self) -> usize {
        1usize << self.nbits
    }

    pub fn zero_state(nbits: usize) -> Self {
        Self::from_reg(&ArrayReg::zero_state(nbits))
    }

    pub fn from_reg(reg: &ArrayReg) -> Self {
        let dim = 1usize << reg.nqubits();
        let mut state = vec![Complex64::new(0.0, 0.0); dim * dim];
        for row in 0..dim {
            for col in 0..dim {
                state[row * dim + col] = reg.state_vec()[row] * reg.state_vec()[col].conj();
            }
        }
        Self {
            nbits: reg.nqubits(),
            state,
        }
    }

    pub fn mixed(weights: &[f64], regs: &[ArrayReg]) -> Self {
        assert!(!regs.is_empty());
        assert_eq!(weights.len(), regs.len());

        let nbits = regs[0].nqubits();
        let dim = 1usize << nbits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim * dim];

        for (weight, reg) in weights.iter().zip(regs.iter()) {
            assert_eq!(reg.nqubits(), nbits);
            for row in 0..dim {
                for col in 0..dim {
                    state[row * dim + col] += Complex64::new(*weight, 0.0)
                        * reg.state_vec()[row]
                        * reg.state_vec()[col].conj();
                }
            }
        }

        Self { nbits, state }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.dim() + col
    }

    pub fn trace(&self) -> Complex64 {
        (0..self.dim())
            .map(|idx| self.state[self.idx(idx, idx)])
            .sum()
    }

    pub fn purity(&self) -> f64 {
        let dim = self.dim();
        let mut acc = Complex64::new(0.0, 0.0);
        for row in 0..dim {
            for inner in 0..dim {
                acc += self.state[self.idx(row, inner)] * self.state[self.idx(inner, row)];
            }
        }
        acc.re
    }

    pub fn partial_tr(&self, traced_locs: &[usize]) -> DensityMatrix {
        let kept_locs: Vec<usize> = (0..self.nbits)
            .filter(|loc| !traced_locs.contains(loc))
            .collect();
        let dim_full = self.dim();
        let dim_reduced = 1usize << kept_locs.len();
        let mut reduced = vec![Complex64::new(0.0, 0.0); dim_reduced * dim_reduced];

        for row in 0..dim_full {
            for col in 0..dim_full {
                if traced_locs
                    .iter()
                    .all(|&loc| ((row >> loc) & 1) == ((col >> loc) & 1))
                {
                    let reduced_row = kept_locs
                        .iter()
                        .enumerate()
                        .fold(0usize, |acc, (idx, &loc)| acc | (((row >> loc) & 1) << idx));
                    let reduced_col = kept_locs
                        .iter()
                        .enumerate()
                        .fold(0usize, |acc, (idx, &loc)| acc | (((col >> loc) & 1) << idx));
                    reduced[reduced_row * dim_reduced + reduced_col] +=
                        self.state[self.idx(row, col)];
                }
            }
        }

        DensityMatrix {
            nbits: kept_locs.len(),
            state: reduced,
        }
    }

    pub fn von_neumann_entropy(&self) -> f64 {
        let dim = self.dim();
        let matrix = Array2::from_shape_fn((dim, dim), |(row, col)| self.state[self.idx(row, col)]);
        let eigenvalues = hermitian_eigenvalues(&matrix);
        -eigenvalues
            .into_iter()
            .filter(|&value| value > 1e-15)
            .map(|value| value * value.ln())
            .sum::<f64>()
    }

    fn is_diagonal_matrix(matrix: &Array2<Complex64>) -> bool {
        (0..matrix.nrows()).all(|row| {
            (0..matrix.ncols()).all(|col| row == col || matrix[[row, col]].norm() < 1e-15)
        })
    }

    fn conjugated_gate(gate: &crate::gate::Gate) -> crate::gate::Gate {
        let matrix = gate.matrix().mapv(|value| value.conj());
        crate::gate::Gate::Custom {
            matrix,
            is_diagonal: gate.is_diagonal(),
            label: "conj".to_string(),
        }
    }

    /// Shift all qubit locations in a gate by `offset` (for vectorized DM evolution).
    fn shift_gate(
        pg: &crate::circuit::PositionedGate,
        offset: usize,
    ) -> crate::circuit::PositionedGate {
        crate::circuit::PositionedGate::new(
            pg.gate.clone(),
            pg.target_locs.iter().map(|&l| l + offset).collect(),
            pg.control_locs.iter().map(|&l| l + offset).collect(),
            pg.control_configs.clone(),
        )
    }

    /// Apply a single gate as ρ → U ρ U† using vectorized evolution.
    ///
    /// Treats the flattened DM as a 2^(2n)-element state vector where:
    /// - high n bits = row index (left/ket space)
    /// - low n bits = column index (right/bra space)
    ///
    /// Then ρ → U ρ U† = apply U at qubit `loc`, then conj(U) at qubit `loc + n`.
    /// Uses the same dispatch_arrayreg_gate as ArrayReg — 2 calls instead of 2*dim.
    fn apply_gate(&mut self, pg: &crate::circuit::PositionedGate) {
        let n = self.nbits;
        let total_bits = 2 * n;

        // Left multiply: U on row-index qubits (positions 0..n-1, no shift needed)
        crate::apply::dispatch_arrayreg_gate(total_bits, &mut self.state, pg);

        // Right multiply: conj(U) on column-index qubits (shift by n)
        let conj_gate = Self::conjugated_gate(&pg.gate);
        let conj_pg = crate::circuit::PositionedGate::new(
            conj_gate,
            pg.target_locs.clone(),
            pg.control_locs.clone(),
            pg.control_configs.clone(),
        );
        let shifted = Self::shift_gate(&conj_pg, n);
        crate::apply::dispatch_arrayreg_gate(total_bits, &mut self.state, &shifted);
    }

    /// Apply a noise channel as ρ → Σ_i K_i ρ K_i† using vectorized evolution.
    fn apply_channel(&mut self, channel: &crate::noise::NoiseChannel, locs: &[usize]) {
        let n = self.nbits;
        let total_bits = 2 * n;
        let original = self.state.clone();
        let mut accumulated = vec![Complex64::new(0.0, 0.0); original.len()];
        let mut branch_state = vec![Complex64::new(0.0, 0.0); original.len()];

        for kraus_op in channel.kraus_operators() {
            branch_state.copy_from_slice(&original);

            // K_i on row-index qubits (no shift)
            let pg = crate::circuit::PositionedGate::new(
                crate::gate::Gate::Custom {
                    matrix: kraus_op.clone(),
                    is_diagonal: Self::is_diagonal_matrix(&kraus_op),
                    label: "kraus".to_string(),
                },
                locs.to_vec(),
                vec![],
                vec![],
            );
            crate::apply::dispatch_arrayreg_gate(total_bits, &mut branch_state, &pg);

            // conj(K_i) on column-index qubits (shift by n)
            let conj_pg = crate::circuit::PositionedGate::new(
                crate::gate::Gate::Custom {
                    matrix: kraus_op.mapv(|v| v.conj()),
                    is_diagonal: Self::is_diagonal_matrix(&kraus_op),
                    label: "kraus_conj".to_string(),
                },
                locs.to_vec(),
                vec![],
                vec![],
            );
            let shifted = Self::shift_gate(&conj_pg, n);
            crate::apply::dispatch_arrayreg_gate(total_bits, &mut branch_state, &shifted);

            for (dst, src) in accumulated.iter_mut().zip(branch_state.iter()) {
                *dst += *src;
            }
        }

        self.state = accumulated;
    }
}

fn hermitian_eigenvalues(matrix: &Array2<Complex64>) -> Vec<f64> {
    let n = matrix.nrows();
    let mut m = faer::Mat::<faer::c64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = faer::c64::new(matrix[[i, j]].re, matrix[[i, j]].im);
        }
    }

    m.as_ref()
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .expect("eigendecomposition failed")
        .into_iter()
        .map(|v| v.max(0.0))
        .collect()
}

impl Register for DensityMatrix {
    fn nbits(&self) -> usize {
        self.nbits
    }

    fn apply(&mut self, circuit: &Circuit) {
        use crate::circuit::CircuitElement;

        for element in &circuit.elements {
            match element {
                CircuitElement::Gate(pg) => self.apply_gate(pg),
                CircuitElement::Channel(pc) => self.apply_channel(&pc.channel, &pc.locs),
                CircuitElement::Annotation(_) => {}
            }
        }
    }

    fn state_data(&self) -> &[Complex64] {
        &self.state
    }
}

pub fn density_matrix_from_reg(reg: &ArrayReg, locs: &[usize]) -> DensityMatrix {
    let full = DensityMatrix::from_reg(reg);
    let traced_locs: Vec<usize> = (0..reg.nqubits())
        .filter(|loc| !locs.contains(loc))
        .collect();
    full.partial_tr(&traced_locs)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_pure_state_trace() {
        let reg = ArrayReg::zero_state(2);
        let dm = DensityMatrix::from_reg(&reg);
        assert_abs_diff_eq!(dm.trace().re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_zero_state_density_matrix() {
        let dm = DensityMatrix::zero_state(2);
        assert_eq!(dm.nbits(), 2);
        assert_eq!(dm.state.len(), 16);
        assert_abs_diff_eq!(dm.trace().re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dm.purity(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dm.state[0].re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_pure_state_purity() {
        let reg = ArrayReg::zero_state(2);
        let dm = DensityMatrix::from_reg(&reg);
        assert_abs_diff_eq!(dm.purity(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_mixed_state_purity() {
        let r0 = ArrayReg::from_vec(1, vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let r1 = ArrayReg::from_vec(1, vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
        let dm = DensityMatrix::mixed(&[0.5, 0.5], &[r0, r1]);
        assert_abs_diff_eq!(dm.purity(), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_partial_trace_bell_state() {
        let reg = ArrayReg::ghz_state(2);
        let dm = DensityMatrix::from_reg(&reg);
        let reduced = dm.partial_tr(&[1]);
        assert_abs_diff_eq!(reduced.purity(), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(reduced.trace().re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_von_neumann_entropy_maximally_mixed_qubit() {
        let r0 = ArrayReg::from_vec(1, vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let r1 = ArrayReg::from_vec(1, vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
        let dm = DensityMatrix::mixed(&[0.5, 0.5], &[r0, r1]);
        assert_abs_diff_eq!(
            dm.von_neumann_entropy(),
            std::f64::consts::LN_2,
            epsilon = 1e-8
        );
    }

    #[test]
    fn test_dm_apply_with_noise_channel() {
        use crate::circuit::{Circuit, channel, put};
        use crate::gate::Gate;
        use crate::noise::NoiseChannel;

        let circ = Circuit::qubits(
            1,
            vec![
                put(vec![0], Gate::H),
                channel(vec![0], NoiseChannel::BitFlip { p: 0.1 }),
            ],
        )
        .unwrap();

        let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(1));
        dm.apply(&circ);
        assert_abs_diff_eq!(dm.trace().re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dm.purity(), 1.0, epsilon = 1e-10);

        let circ2 =
            Circuit::qubits(1, vec![channel(vec![0], NoiseChannel::BitFlip { p: 0.5 })]).unwrap();
        let mut dm2 = DensityMatrix::from_reg(&ArrayReg::zero_state(1));
        dm2.apply(&circ2);
        assert_abs_diff_eq!(dm2.trace().re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dm2.purity(), 0.5, epsilon = 1e-10);
    }
}
