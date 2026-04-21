use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use crate::ad::expect_grad;
use crate::circuit::{Circuit, put};
use crate::gate::Gate;
use crate::operator::{Op, OperatorPolynomial};
use crate::register::ArrayReg;

#[test]
fn expect_grad_forward_value_only() {
    let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::H)]).unwrap();
    let psi0 = ArrayReg::zero_state(1);
    let h = OperatorPolynomial::single(0, Op::Z, Complex64::new(1.0, 0.0));
    let (value, grad) = expect_grad(&h, &circuit, &psi0);
    assert_abs_diff_eq!(value, 0.0, epsilon = 1e-12);
    assert!(grad.iter().all(|&x| x == 0.0));
    assert_eq!(grad.len(), 0);
}
