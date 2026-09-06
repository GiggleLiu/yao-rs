use num_complex::Complex64;

use crate::density_matrix::DensityMatrix;
use crate::operator::{Op, OperatorPolynomial, OperatorString, op_matrix};
use crate::register::{ArrayReg, Register};

pub fn expect_arrayreg(reg: &ArrayReg, op: &OperatorPolynomial) -> Complex64 {
    let mut scratch = vec![Complex64::new(0., 0.); reg.state_vec().len()];
    expect_arrayreg_with_scratch(reg, op, &mut scratch)
}

/// Reuse a caller-owned state buffer for streaming trajectory observables.
pub(crate) fn expect_arrayreg_with_scratch(
    reg: &ArrayReg,
    op: &OperatorPolynomial,
    scratch: &mut [Complex64],
) -> Complex64 {
    op.iter()
        .map(|(coeff, word)| {
            scratch.copy_from_slice(reg.state_vec());
            for &(loc, op) in word.ops() {
                apply_single_op(scratch, loc, &op);
            }
            let value: Complex64 = reg
                .state_vec()
                .iter()
                .zip(scratch.iter())
                .map(|(lhs, rhs)| lhs.conj() * rhs)
                .sum();
            *coeff * value
        })
        .sum()
}

fn apply_single_op(state: &mut [Complex64], loc: usize, op: &Op) {
    let matrix = op_matrix(op);
    crate::instruct_qubit::instruct_1q(
        state,
        loc,
        matrix[[0, 0]],
        matrix[[0, 1]],
        matrix[[1, 0]],
        matrix[[1, 1]],
    );
}

pub fn expect_dm(dm: &DensityMatrix, op: &OperatorPolynomial) -> Complex64 {
    op.iter()
        .map(|(coeff, opstring)| *coeff * expect_opstring_dm(dm, opstring))
        .sum()
}

fn expect_opstring_dm(dm: &DensityMatrix, opstring: &OperatorString) -> Complex64 {
    let dim = 1usize << dm.nbits();
    let mut state = dm.state_data().to_vec();
    // Row-major vectorization places the row qubits first (0 = MSB).
    // Left multiplication by O followed by the trace gives Tr(O rho).
    for &(loc, op) in opstring.ops() {
        assert!(loc < dm.nbits(), "operator site out of range");
        apply_single_op(&mut state, loc, &op);
    }
    (0..dim).map(|i| state[i * dim + i]).sum()
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::operator::{Op, OperatorString};

    #[test]
    fn test_expect_z_zero_state() {
        let reg = ArrayReg::zero_state(1);
        let op = OperatorPolynomial::single(0, Op::Z, Complex64::new(1.0, 0.0));
        let result = expect_arrayreg(&reg, &op);
        assert_abs_diff_eq!(result.re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_expect_x_zero_state() {
        let reg = ArrayReg::zero_state(1);
        let op = OperatorPolynomial::single(0, Op::X, Complex64::new(1.0, 0.0));
        let result = expect_arrayreg(&reg, &op);
        assert_abs_diff_eq!(result.re, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_expect_z_plus_state() {
        let reg = ArrayReg::uniform_state(1);
        let op = OperatorPolynomial::single(0, Op::Z, Complex64::new(1.0, 0.0));
        let result = expect_arrayreg(&reg, &op);
        assert_abs_diff_eq!(result.re, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_expect_dm_matches_pure() {
        let reg = ArrayReg::uniform_state(2);
        let dm = DensityMatrix::from_reg(&reg);
        let op = OperatorPolynomial::new(
            vec![Complex64::new(1.0, 0.0)],
            vec![OperatorString::new(vec![(0, Op::Z), (1, Op::Z)])],
        );

        let pure = expect_arrayreg(&reg, &op);
        let mixed = expect_dm(&dm, &op);
        assert_abs_diff_eq!(pure.re, mixed.re, epsilon = 1e-10);
        assert_abs_diff_eq!(pure.im, mixed.im, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod ordering_tests {
    use super::*;

    #[test]
    fn density_expectations_match_pure_on_asymmetric_complex_states() {
        let reg = ArrayReg::deterministic_state(3);
        let dm = DensityMatrix::from_reg(&reg);
        for site in 0..3 {
            for op in [Op::X, Op::Y, Op::Z, Op::P0, Op::P1, Op::Pu, Op::Pd] {
                let operator = OperatorPolynomial::single(site, op, Complex64::new(0.7, -0.3));
                let pure = expect_arrayreg(&reg, &operator);
                let mixed = expect_dm(&dm, &operator);
                assert!(
                    (pure - mixed).norm() < 1e-12,
                    "site={site} op={op:?}: {pure} != {mixed}"
                );
            }
        }
    }
}
