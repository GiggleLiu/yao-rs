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
        (0..matrix.nrows())
            .all(|row| (0..matrix.ncols()).all(|col| row == col || matrix[[row, col]].norm() < 1e-15))
    }

    fn conjugated_gate(gate: &crate::gate::Gate) -> crate::gate::Gate {
        let matrix = gate.matrix().mapv(|value| value.conj());
        crate::gate::Gate::Custom {
            matrix,
            is_diagonal: gate.is_diagonal(),
            label: "conj".to_string(),
        }
    }

    fn apply_left_map(&mut self, circuit: &crate::circuit::Circuit) {
        let dim = self.dim();
        for col in 0..dim {
            let column: Vec<Complex64> = (0..dim).map(|row| self.state[self.idx(row, col)]).collect();
            let mut reg = ArrayReg::from_vec(self.nbits, column);
            crate::apply::apply_inplace(circuit, &mut reg);
            for row in 0..dim {
                let idx = row * dim + col;
                self.state[idx] = reg.state[row];
            }
        }
    }

    fn apply_right_map(&mut self, circuit: &crate::circuit::Circuit) {
        let dim = self.dim();
        for row in 0..dim {
            let row_state: Vec<Complex64> = (0..dim).map(|col| self.state[self.idx(row, col)]).collect();
            let mut reg = ArrayReg::from_vec(self.nbits, row_state);
            crate::apply::apply_inplace(circuit, &mut reg);
            for col in 0..dim {
                let idx = row * dim + col;
                self.state[idx] = reg.state[col];
            }
        }
    }

    /// Apply a single gate as rho -> U rho U†.
    fn apply_gate(&mut self, pg: &crate::circuit::PositionedGate) {
        let single_circuit = crate::circuit::Circuit::qubits(
            self.nbits,
            vec![crate::circuit::CircuitElement::Gate(pg.clone())],
        )
        .unwrap();
        let conjugated_gate = crate::circuit::PositionedGate::new(
            Self::conjugated_gate(&pg.gate),
            pg.target_locs.clone(),
            pg.control_locs.clone(),
            pg.control_configs.clone(),
        );
        let conjugated_circuit = crate::circuit::Circuit::qubits(
            self.nbits,
            vec![crate::circuit::CircuitElement::Gate(conjugated_gate)],
        )
        .unwrap();

        self.apply_left_map(&single_circuit);
        self.apply_right_map(&conjugated_circuit);
    }

    /// Apply a noise channel as rho -> sum_i K_i rho K_i†.
    fn apply_channel(&mut self, channel: &crate::noise::NoiseChannel, locs: &[usize]) {
        let original = self.state.clone();
        let mut accumulated = vec![Complex64::new(0.0, 0.0); original.len()];

        for kraus_op in channel.kraus_operators() {
            let gate = crate::gate::Gate::Custom {
                matrix: kraus_op.clone(),
                is_diagonal: Self::is_diagonal_matrix(&kraus_op),
                label: "kraus".to_string(),
            };
            let conjugated = crate::gate::Gate::Custom {
                matrix: kraus_op.mapv(|value| value.conj()),
                is_diagonal: Self::is_diagonal_matrix(&kraus_op),
                label: "kraus_conj".to_string(),
            };
            let left_circuit =
                crate::circuit::Circuit::qubits(self.nbits, vec![crate::circuit::put(locs.to_vec(), gate)])
                    .unwrap();
            let right_circuit = crate::circuit::Circuit::qubits(
                self.nbits,
                vec![crate::circuit::put(locs.to_vec(), conjugated)],
            )
            .unwrap();

            let mut branch = Self {
                nbits: self.nbits,
                state: original.clone(),
            };
            branch.apply_left_map(&left_circuit);
            branch.apply_right_map(&right_circuit);

            for (dst, src) in accumulated.iter_mut().zip(branch.state.into_iter()) {
                *dst += src;
            }
        }

        self.state = accumulated;
    }
}

fn hermitian_eigenvalues(matrix: &Array2<Complex64>) -> Vec<f64> {
    let mut current = matrix.clone();
    let n = current.nrows();
    let max_iter = 256usize;
    let tolerance = 1e-12;

    for _ in 0..max_iter {
        let (q, r) = qr_decompose(&current);
        current = r.dot(&q);

        let mut off_diag_sum = 0.0;
        for row in 0..n {
            for col in 0..n {
                if row != col {
                    off_diag_sum += current[[row, col]].norm_sqr();
                }
            }
        }
        let off_diag = off_diag_sum.sqrt();
        if off_diag < tolerance {
            break;
        }
    }

    (0..n).map(|idx| current[[idx, idx]].re.max(0.0)).collect()
}

fn qr_decompose(matrix: &Array2<Complex64>) -> (Array2<Complex64>, Array2<Complex64>) {
    let n = matrix.nrows();
    let mut q = Array2::<Complex64>::zeros((n, n));
    let mut r = Array2::<Complex64>::zeros((n, n));

    for col in 0..n {
        let mut v: Vec<Complex64> = (0..n).map(|row| matrix[[row, col]]).collect();

        for prev in 0..col {
            let coeff: Complex64 = (0..n).map(|row| q[[row, prev]].conj() * v[row]).sum();
            r[[prev, col]] = coeff;
            for row in 0..n {
                v[row] -= coeff * q[[row, prev]];
            }
        }

        let norm = v.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
        if norm <= 1e-15 {
            continue;
        }

        r[[col, col]] = Complex64::new(norm, 0.0);
        for row in 0..n {
            q[[row, col]] = v[row] / norm;
        }
    }

    (q, r)
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
