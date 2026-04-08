use crate::operator_parser;
use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use yao_rs::{ArrayReg, op_matrix};

pub fn expect(input: &str, op_str: &str, out: &OutputConfig) -> Result<()> {
    let reg = state_io::read_state(input)?;
    let operator = operator_parser::parse_operator(op_str)?;
    let value = compute_expectation(&reg, &operator);

    let (human, json_value) = super::format_expectation(op_str, value);

    out.emit(&human, &json_value)
}

/// Compute <psi|O|psi> via brute-force double-loop over the full Hilbert space.
///
/// **Warning**: O(D^2) where D = 2^n for n qubits.
/// Suitable for small systems only (n <= ~15).
pub fn compute_expectation(reg: &ArrayReg, operator: &yao_rs::OperatorPolynomial) -> Complex64 {
    let n = reg.nqubits();
    let total_dim = 1usize << n;
    let state = reg.state_vec();
    let mut result = Complex64::new(0.0, 0.0);
    let identity = op_matrix(&yao_rs::Op::I);

    for (coeff, opstring) in operator.iter() {
        let ops = opstring.ops();
        let mut term_val = Complex64::new(0.0, 0.0);

        let op_matrices: Vec<Array2<Complex64>> = ops.iter().map(|(_, op)| op_matrix(op)).collect();
        let op_sites: Vec<usize> = ops.iter().map(|(site, _)| *site).collect();
        let site_ops: Vec<&Array2<Complex64>> = (0..n)
            .map(|site| {
                op_sites
                    .iter()
                    .position(|&s| s == site)
                    .map(|idx| &op_matrices[idx])
                    .unwrap_or(&identity)
            })
            .collect();

        for i in 0..total_dim {
            let psi_i_conj = state[i].conj();
            if psi_i_conj.norm() < 1e-15 {
                continue;
            }

            for (j, &psi_j) in state.iter().enumerate().take(total_dim) {
                if psi_j.norm() < 1e-15 {
                    continue;
                }

                let mut matrix_elem = Complex64::new(1.0, 0.0);
                for (site, op) in site_ops.iter().enumerate().take(n) {
                    // Extract bit for this site from basis index
                    // Site 0 is MSB: bit = (index >> (n-1-site)) & 1
                    let i_bit = (i >> (n - 1 - site)) & 1;
                    let j_bit = (j >> (n - 1 - site)) & 1;
                    matrix_elem *= op[[i_bit, j_bit]];
                }

                term_val += psi_i_conj * matrix_elem * psi_j;
            }
        }

        result += *coeff * term_val;
    }

    result
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use yao_rs::{ArrayReg, Op, OperatorPolynomial};

    use super::compute_expectation;

    #[test]
    fn computes_z_expectation_for_one_state() {
        // |1⟩ state
        let reg = ArrayReg::from_vec(1, vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
        let operator = OperatorPolynomial::single(0, Op::Z, Complex64::new(1.0, 0.0));

        let value = compute_expectation(&reg, &operator);

        assert!((value - Complex64::new(-1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn computes_x_expectation_for_plus_state() {
        let amplitude = 1.0 / 2.0_f64.sqrt();
        let reg = ArrayReg::from_vec(
            1,
            vec![
                Complex64::new(amplitude, 0.0),
                Complex64::new(amplitude, 0.0),
            ],
        );
        let operator = OperatorPolynomial::single(0, Op::X, Complex64::new(1.0, 0.0));

        let value = compute_expectation(&reg, &operator);

        assert!((value - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn computes_zz_expectation_for_bell_state() {
        // Bell state |00> + |11> / sqrt(2): <ZZ> = 1
        let amp = 1.0 / 2.0_f64.sqrt();
        let reg = ArrayReg::from_vec(
            2,
            vec![
                Complex64::new(amp, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(amp, 0.0),
            ],
        );
        let operator = OperatorPolynomial::new(
            vec![Complex64::new(1.0, 0.0)],
            vec![yao_rs::OperatorString::new(vec![(0, Op::Z), (1, Op::Z)])],
        );

        let value = compute_expectation(&reg, &operator);
        assert!((value - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn computes_sum_operator_on_two_qubits() {
        // |01>: <Z(0)> = 1, <Z(1)> = -1, so <Z(0) + Z(1)> = 0
        let reg = ArrayReg::from_vec(
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let operator = OperatorPolynomial::new(
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![
                yao_rs::OperatorString::new(vec![(0, Op::Z)]),
                yao_rs::OperatorString::new(vec![(1, Op::Z)]),
            ],
        );

        let value = compute_expectation(&reg, &operator);
        assert!(value.norm() < 1e-12);
    }
}
