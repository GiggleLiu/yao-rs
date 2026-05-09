use num_complex::Complex64;

use yao_rs::ad::expect_grad;
use yao_rs::circuit::{Circuit, control, put};
use yao_rs::gate::Gate;
use yao_rs::operator::{Op, OperatorPolynomial};
use yao_rs::register::ArrayReg;

fn heisenberg_zz_xx(n: usize) -> OperatorPolynomial {
    use yao_rs::operator::OperatorString;

    let one = Complex64::new(1.0, 0.0);
    let mut h = OperatorPolynomial::zero();
    for i in 0..n - 1 {
        let zz = OperatorPolynomial::new(
            vec![one],
            vec![OperatorString::new(vec![(i, Op::Z), (i + 1, Op::Z)])],
        );
        let xx = OperatorPolynomial::new(
            vec![one],
            vec![OperatorString::new(vec![(i, Op::X), (i + 1, Op::X)])],
        );
        h = &(&h + &zz) + &xx;
    }
    h
}

fn small_ansatz(n: usize) -> Circuit {
    let mut elems = Vec::new();
    for q in 0..n {
        elems.push(put(vec![q], Gate::H));
    }
    for _layer in 0..2 {
        for q in 0..n {
            elems.push(put(vec![q], Gate::Ry(0.0)));
        }
        for q in 0..n - 1 {
            elems.push(control(vec![q], vec![q + 1], Gate::X));
        }
    }
    Circuit::qubits(n, elems).unwrap()
}

#[test]
fn plain_sgd_drives_heisenberg_energy_down() {
    let n = 4usize;
    let h = heisenberg_zz_xx(n);
    let psi0 = ArrayReg::zero_state(n);
    let mut circuit = small_ansatz(n);

    let nparams = circuit.num_params();
    let mut params: Vec<f64> = (0..nparams).map(|i| 0.01 + 0.005 * (i as f64)).collect();
    circuit.dispatch(&params);

    let (initial, _) = expect_grad(&h, &circuit, &psi0);
    let lr = 0.05_f64;
    let steps = 150;
    let mut history = Vec::with_capacity(steps + 1);
    history.push(initial);

    for _ in 0..steps {
        circuit.dispatch(&params);
        let (val, grad) = expect_grad(&h, &circuit, &psi0);
        history.push(val);
        for (p, g) in params.iter_mut().zip(&grad) {
            *p -= lr * g;
        }
    }

    let final_val = *history.last().unwrap();
    assert!(
        final_val < initial - 0.5,
        "final energy {final_val} did not drop at least 0.5 below initial {initial}"
    );
    let down = history.windows(2).filter(|w| w[1] < w[0]).count();
    assert!(
        down as f64 / (history.len() - 1) as f64 > 0.85,
        "fewer than 85% steps decreased energy"
    );
}
