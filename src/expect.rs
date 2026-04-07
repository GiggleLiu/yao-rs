use num_complex::Complex64;

use crate::density_matrix::DensityMatrix;
use crate::operator::{Op, OperatorPolynomial, OperatorString, op_matrix};
use crate::register::{ArrayReg, Register};

pub fn expect_arrayreg(reg: &ArrayReg, op: &OperatorPolynomial) -> Complex64 {
    op.iter()
        .map(|(coeff, opstring)| *coeff * expect_opstring_pure(reg, opstring))
        .sum()
}

fn expect_opstring_pure(reg: &ArrayReg, opstring: &OperatorString) -> Complex64 {
    let mut state = reg.state_vec().to_vec();
    for &(loc, op) in opstring.ops() {
        apply_single_op(&mut state, loc, &op);
    }

    reg.state_vec()
        .iter()
        .zip(state.iter())
        .map(|(lhs, rhs)| lhs.conj() * rhs)
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
    let nbits = dm.nbits();
    let dim = 1usize << nbits;
    let op_sites = opstring.ops();

    let mut trace = Complex64::new(0.0, 0.0);
    for row in 0..dim {
        for col in 0..dim {
            let mut op_entry = Complex64::new(1.0, 0.0);

            for &(loc, op) in op_sites {
                let matrix = op_matrix(&op);
                let row_bit = (row >> loc) & 1;
                let col_bit = (col >> loc) & 1;
                op_entry *= matrix[[row_bit, col_bit]];
            }

            let untouched_match = (0..nbits)
                .filter(|loc| !op_sites.iter().any(|(site, _)| site == loc))
                .all(|loc| ((row >> loc) & 1) == ((col >> loc) & 1));

            if untouched_match {
                trace += op_entry * dm.state_data()[col * dim + row];
            }
        }
    }

    trace
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
