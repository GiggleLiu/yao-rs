use crate::operator_parser;
use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
#[cfg(test)]
use num_complex::Complex64;
#[cfg(test)]
use yao_rs::ArrayReg;

pub fn expect(input: &str, op_str: &str, out: &OutputConfig) -> Result<()> {
    let reg = state_io::read_state(input)?;
    let operator = operator_parser::parse_operator_for_qubits(op_str, reg.nqubits())?;
    let value = reg.expectation(&operator);

    let (human, json_value) = super::format_expectation(op_str, value);

    out.emit(&human, &json_value)
}

/// Reuse the library's O(terms * sites * 2^n) expectation implementation.
#[cfg(test)]
fn compute_expectation(reg: &ArrayReg, operator: &yao_rs::OperatorPolynomial) -> Complex64 {
    crate::state_io::State::Pure(reg.clone()).expectation(operator)
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
