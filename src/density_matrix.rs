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
                    state[row * dim + col] +=
                        Complex64::new(*weight, 0.0) * reg.state_vec()[row] * reg.state_vec()[col].conj();
                }
            }
        }

        Self { nbits, state }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.dim() + col
    }

    pub fn trace(&self) -> Complex64 {
        (0..self.dim()).map(|idx| self.state[self.idx(idx, idx)]).sum()
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
                    let reduced_row = kept_locs.iter().enumerate().fold(0usize, |acc, (idx, &loc)| {
                        acc | (((row >> loc) & 1) << idx)
                    });
                    let reduced_col = kept_locs.iter().enumerate().fold(0usize, |acc, (idx, &loc)| {
                        acc | (((col >> loc) & 1) << idx)
                    });
                    reduced[reduced_row * dim_reduced + reduced_col] += self.state[self.idx(row, col)];
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
        let dim = self.dim();
        let mut columns = Vec::with_capacity(dim);

        for basis_state in 0..dim {
            let mut state = vec![Complex64::new(0.0, 0.0); dim];
            state[basis_state] = Complex64::new(1.0, 0.0);
            let mut reg = ArrayReg::from_vec(self.nbits, state);
            crate::apply::apply_inplace(circuit, &mut reg);
            columns.push(reg.state);
        }

        let mut transformed = vec![Complex64::new(0.0, 0.0); dim * dim];
        for row in 0..dim {
            for col in 0..dim {
                let mut acc = Complex64::new(0.0, 0.0);
                for left in 0..dim {
                    for right in 0..dim {
                        acc += columns[left][row] * self.state[left * dim + right] * columns[right][col].conj();
                    }
                }
                transformed[row * dim + col] = acc;
            }
        }

        self.state = transformed;
    }

    fn state_data(&self) -> &[Complex64] {
        &self.state
    }
}

pub fn density_matrix_from_reg(reg: &ArrayReg, locs: &[usize]) -> DensityMatrix {
    let full = DensityMatrix::from_reg(reg);
    let traced_locs: Vec<usize> = (0..reg.nqubits()).filter(|loc| !locs.contains(loc)).collect();
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
        let r0 = ArrayReg::from_vec(
            1,
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        );
        let r1 = ArrayReg::from_vec(
            1,
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        );
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
        let r0 = ArrayReg::from_vec(
            1,
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        );
        let r1 = ArrayReg::from_vec(
            1,
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        );
        let dm = DensityMatrix::mixed(&[0.5, 0.5], &[r0, r1]);
        assert_abs_diff_eq!(dm.von_neumann_entropy(), std::f64::consts::LN_2, epsilon = 1e-8);
    }
}
