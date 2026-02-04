use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

use yao_rs::apply::apply;
use yao_rs::circuit::{Circuit, CircuitElement, PositionedGate};
use yao_rs::gate::Gate;
use yao_rs::state::State;

const ATOL: f64 = 1e-10;

fn assert_state_approx(result: &State, expected: &[Complex64]) {
    assert_eq!(result.data.len(), expected.len());
    for (i, (r, e)) in result.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).norm() < ATOL,
            "State mismatch at index {}: got {:?}, expected {:?}",
            i,
            r,
            e
        );
    }
}

#[test]
fn test_x_gate_on_zero() {
    // X|0> = |1>
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![CircuitElement::Gate(PositionedGate::new(Gate::X, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let result = apply(&circuit, &state);
    let expected = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    assert_state_approx(&result, &expected);
}

#[test]
fn test_h_gate_on_zero() {
    // H|0> = (|0> + |1>) / sqrt(2)
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![CircuitElement::Gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![]))],
    )
    .unwrap();

    let result = apply(&circuit, &state);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let expected = vec![s, s];
    assert_state_approx(&result, &expected);
}

#[test]
fn test_cnot_10_to_11() {
    // CNOT|10> = |11> (control on qubit 0, target on qubit 1)
    let dims = vec![2, 2];
    let state = State::product_state(&dims, &[1, 0]); // |10>
    let circuit = Circuit::new(
        dims.clone(),
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::X,
            vec![1],       // target
            vec![0],       // control
            vec![true],    // trigger on |1>
        ))],
    )
    .unwrap();

    let result = apply(&circuit, &state);
    // |11> is index 3 in a 2-qubit system: [|00>, |01>, |10>, |11>]
    let expected = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    assert_state_approx(&result, &expected);
}

#[test]
fn test_cnot_00_unchanged() {
    // CNOT|00> = |00> (control not triggered)
    let dims = vec![2, 2];
    let state = State::zero_state(&dims); // |00>
    let circuit = Circuit::new(
        dims.clone(),
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::X,
            vec![1],       // target
            vec![0],       // control
            vec![true],    // trigger on |1>
        ))],
    )
    .unwrap();

    let result = apply(&circuit, &state);
    let expected = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    assert_state_approx(&result, &expected);
}

#[test]
fn test_bell_state() {
    // H on qubit 0, then CNOT(0->1) on |00> gives (|00> + |11>) / sqrt(2)
    let dims = vec![2, 2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            CircuitElement::Gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            CircuitElement::Gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
        ],
    )
    .unwrap();

    let result = apply(&circuit, &state);
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let expected = vec![s, zero, zero, s];
    assert_state_approx(&result, &expected);
}

#[test]
fn test_norm_preservation() {
    // Apply H then X then H on a single qubit, check norm is preserved
    let dims = vec![2];
    let state = State::zero_state(&dims);
    let circuit = Circuit::new(
        dims.clone(),
        vec![
            CircuitElement::Gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            CircuitElement::Gate(PositionedGate::new(Gate::X, vec![0], vec![], vec![])),
            CircuitElement::Gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
        ],
    )
    .unwrap();

    let result = apply(&circuit, &state);
    assert!((result.norm() - 1.0).abs() < ATOL);
}

#[test]
fn test_qutrit_cyclic_permutation() {
    // Cyclic permutation on a qutrit: |0> -> |1>, |1> -> |2>, |2> -> |0>
    let dims = vec![3];
    let state = State::zero_state(&dims); // |0>

    // Build cyclic permutation matrix
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    // P|i> = |i+1 mod 3>
    // P = [[0,0,1],[1,0,0],[0,1,0]]
    let perm_matrix = Array2::from_shape_vec(
        (3, 3),
        vec![zero, zero, one, one, zero, zero, zero, one, zero],
    )
    .unwrap();

    let circuit = Circuit::new(
        dims.clone(),
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::Custom {
                matrix: perm_matrix,
                is_diagonal: false,
                label: "qutrit_cyclic_perm".to_string(),
            },
            vec![0],
            vec![],
            vec![],
        )],
    )
    .unwrap();

    let result = apply(&circuit, &state);
    // |0> -> |1>
    let expected = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    assert_state_approx(&result, &expected);
}

#[test]
fn test_x_on_second_qubit() {
    // X on qubit 1 of |00> -> |01>
    let dims = vec![2, 2];
    let state = State::zero_state(&dims); // |00>
    let circuit = Circuit::new(
        dims.clone(),
        vec![CircuitElement::Gate(PositionedGate::new(Gate::X, vec![1], vec![], vec![]))],
    )
    .unwrap();

    let result = apply(&circuit, &state);
    // |01> is index 1
    let expected = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    assert_state_approx(&result, &expected);
}
