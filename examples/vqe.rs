//! VQE demo: minimize the 4-qubit Heisenberg (ZZ+XX) energy with plain SGD
//! using adjoint-mode gradients.
//!
//! Run with: cargo run --release --example vqe

use num_complex::Complex64;

use yao_rs::ad::expect_grad;
use yao_rs::circuit::{Circuit, control, put};
use yao_rs::gate::Gate;
use yao_rs::operator::{Op, OperatorPolynomial, OperatorString};
use yao_rs::register::ArrayReg;

fn heisenberg(n: usize) -> OperatorPolynomial {
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

fn ansatz(n: usize, layers: usize) -> Circuit {
    let mut elems = Vec::new();
    for q in 0..n {
        elems.push(put(vec![q], Gate::H));
    }
    for _ in 0..layers {
        for q in 0..n {
            elems.push(put(vec![q], Gate::Ry(0.0)));
        }
        for q in 0..n - 1 {
            elems.push(control(vec![q], vec![q + 1], Gate::X));
        }
    }
    Circuit::qubits(n, elems).unwrap()
}

fn main() {
    let n = 4usize;
    let h = heisenberg(n);
    let psi0 = ArrayReg::zero_state(n);
    let mut circuit = ansatz(n, 3);

    let mut params: Vec<f64> = (0..circuit.num_params())
        .map(|i| 0.01 + 0.007 * (i as f64))
        .collect();
    circuit.dispatch(&params);

    let lr = 0.05_f64;
    let steps = 300;
    let (initial, _) = expect_grad(&h, &circuit, &psi0);
    println!("step 0: energy = {:.6}", initial);
    for step in 1..=steps {
        circuit.dispatch(&params);
        let (val, grad) = expect_grad(&h, &circuit, &psi0);
        for (p, g) in params.iter_mut().zip(&grad) {
            *p -= lr * g;
        }
        if step % 50 == 0 || step == steps {
            println!("step {step}: energy = {:.6}", val);
        }
    }
}
