mod common;

use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use yao_rs::apply::{apply, apply_inplace};
use yao_rs::circuit::{Circuit, CircuitElement, PositionedGate, control, put};
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
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::X,
            vec![0],
            vec![],
            vec![],
        ))],
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
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::H,
            vec![0],
            vec![],
            vec![],
        ))],
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
            vec![1],    // target
            vec![0],    // control
            vec![true], // trigger on |1>
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
            vec![1],    // target
            vec![0],    // control
            vec![true], // trigger on |1>
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
        ))],
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
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::X,
            vec![1],
            vec![],
            vec![],
        ))],
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

// =============================================================================
// Tests migrated from src/apply.rs inline tests
// =============================================================================

#[test]
fn test_linear_to_multi() {
    assert_eq!(common::linear_to_multi(0, &[2, 3]), vec![0, 0]);
    assert_eq!(common::linear_to_multi(1, &[2, 3]), vec![0, 1]);
    assert_eq!(common::linear_to_multi(3, &[2, 3]), vec![1, 0]);
    assert_eq!(common::linear_to_multi(5, &[2, 3]), vec![1, 2]);
}

#[test]
fn test_multi_to_linear() {
    assert_eq!(common::multi_to_linear(&[0, 0], &[2, 3]), 0);
    assert_eq!(common::multi_to_linear(&[0, 1], &[2, 3]), 1);
    assert_eq!(common::multi_to_linear(&[1, 0], &[2, 3]), 3);
    assert_eq!(common::multi_to_linear(&[1, 2], &[2, 3]), 5);
}

#[test]
fn test_roundtrip() {
    let dims = [2, 3, 2];
    let total: usize = dims.iter().product();
    for i in 0..total {
        let multi = common::linear_to_multi(i, &dims);
        assert_eq!(common::multi_to_linear(&multi, &dims), i);
    }
}

// Tests comparing new apply with old full_matrix approach

fn states_approx_equal(a: &State, b: &State, tol: f64) -> bool {
    if a.dims != b.dims {
        return false;
    }
    for i in 0..a.data.len() {
        if (a.data[i] - b.data[i]).norm() > tol {
            return false;
        }
    }
    true
}

#[test]
fn test_apply_vs_apply_old_single_h() {
    // Single H gate on 2-qubit system
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::H)]).unwrap();

    let state = State::zero_state(&[2, 2]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_diagonal_gates() {
    // Z, S, T gates (all diagonal)
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            put(vec![0], Gate::Z),
            put(vec![1], Gate::S),
            put(vec![2], Gate::T),
        ],
    )
    .unwrap();

    // Start with superposition state
    let mut state = State::zero_state(&[2, 2, 2]);
    // Create |+++> state manually by setting all amplitudes equal
    let amp = Complex64::new(1.0 / (8.0_f64).sqrt(), 0.0);
    for i in 0..8 {
        state.data[i] = amp;
    }

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_phase_gate() {
    // Phase gate with arbitrary angle
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::Phase(PI / 3.0))]).unwrap();

    // Start with superposition
    let mut state = State::zero_state(&[2, 2]);
    let s = Complex64::new(0.5, 0.0);
    for i in 0..4 {
        state.data[i] = s;
    }

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_rz_gate() {
    // Rz gate (diagonal)
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![1], Gate::Rz(PI / 4.0))]).unwrap();

    let mut state = State::zero_state(&[2, 2]);
    state.data[0] = Complex64::new(0.5, 0.0);
    state.data[1] = Complex64::new(0.5, 0.0);
    state.data[2] = Complex64::new(0.5, 0.0);
    state.data[3] = Complex64::new(0.5, 0.0);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_bell_circuit() {
    // Bell state circuit: H on qubit 0, then CNOT
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();

    let state = State::zero_state(&[2, 2]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_cnot() {
    // CNOT gate
    let circuit = Circuit::new(vec![2, 2], vec![control(vec![0], vec![1], Gate::X)]).unwrap();

    // Test with |10>
    let state = State::product_state(&[2, 2], &[1, 0]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_toffoli() {
    // Toffoli (CCX) gate
    let circuit = Circuit::new(vec![2, 2, 2], vec![control(vec![0, 1], vec![2], Gate::X)]).unwrap();

    // Test with |110>
    let state = State::product_state(&[2, 2, 2], &[1, 1, 0]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_controlled_phase() {
    // Controlled-Z gate
    let circuit = Circuit::new(vec![2, 2], vec![control(vec![0], vec![1], Gate::Z)]).unwrap();

    // Start with superposition
    let mut state = State::zero_state(&[2, 2]);
    let s = Complex64::new(0.5, 0.0);
    for i in 0..4 {
        state.data[i] = s;
    }

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_rx_ry() {
    // Rx and Ry gates (not diagonal)
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            put(vec![0], Gate::Rx(PI / 5.0)),
            put(vec![1], Gate::Ry(PI / 7.0)),
        ],
    )
    .unwrap();

    let state = State::zero_state(&[2, 2]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_complex_circuit() {
    // More complex circuit with multiple gates
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            put(vec![0], Gate::H),
            put(vec![1], Gate::H),
            put(vec![2], Gate::H),
            control(vec![0], vec![1], Gate::X),
            put(vec![0], Gate::T),
            control(vec![1], vec![2], Gate::X),
            put(vec![1], Gate::S),
            control(vec![0, 1], vec![2], Gate::X),
        ],
    )
    .unwrap();

    let state = State::zero_state(&[2, 2, 2]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_4_qubit() {
    // 4 qubit circuit
    let circuit = Circuit::new(
        vec![2, 2, 2, 2],
        vec![
            put(vec![0], Gate::H),
            put(vec![1], Gate::H),
            control(vec![0], vec![2], Gate::X),
            control(vec![1], vec![3], Gate::X),
            put(vec![2], Gate::Z),
            put(vec![3], Gate::T),
        ],
    )
    .unwrap();

    let state = State::zero_state(&[2, 2, 2, 2]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_random_gates() {
    // Circuit with X, Y, Z, H gates in various combinations
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            put(vec![0], Gate::X),
            put(vec![1], Gate::Y),
            put(vec![2], Gate::Z),
            put(vec![0], Gate::H),
            control(vec![0], vec![1], Gate::X),
            put(vec![2], Gate::H),
        ],
    )
    .unwrap();

    let state = State::zero_state(&[2, 2, 2]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_swap_gate() {
    // SWAP gate (multi-target, no controls)
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0, 1], Gate::SWAP)]).unwrap();

    // Test with |10> -> |01>
    let state = State::product_state(&[2, 2], &[1, 0]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_qft_like_circuit() {
    // QFT-like circuit (simplified)
    // QFT on 3 qubits has H gates and controlled phase rotations
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            // First qubit
            put(vec![0], Gate::H),
            control(vec![1], vec![0], Gate::Phase(PI / 2.0)),
            control(vec![2], vec![0], Gate::Phase(PI / 4.0)),
            // Second qubit
            put(vec![1], Gate::H),
            control(vec![2], vec![1], Gate::Phase(PI / 2.0)),
            // Third qubit
            put(vec![2], Gate::H),
        ],
    )
    .unwrap();

    let state = State::zero_state(&[2, 2, 2]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_inplace_bell_state() {
    // Test that apply_inplace modifies state correctly
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();

    let mut state = State::zero_state(&[2, 2]);
    apply_inplace(&circuit, &mut state);

    // Should be Bell state (|00> + |11>) / sqrt(2)
    let s = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    assert!((state.data[0] - s).norm() < 1e-10);
    assert!(state.data[1].norm() < 1e-10);
    assert!(state.data[2].norm() < 1e-10);
    assert!((state.data[3] - s).norm() < 1e-10);
}

#[test]
fn test_apply_preserves_input() {
    // Test that apply does not modify the input state
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::X)]).unwrap();

    let state = State::zero_state(&[2, 2]);
    let state_clone = state.clone();
    let _result = apply(&circuit, &state);

    // Original state should be unchanged
    assert!(states_approx_equal(&state, &state_clone, 1e-10));
}

#[test]
fn test_is_diagonal() {
    use yao_rs::apply::is_diagonal;
    assert!(is_diagonal(&Gate::Z));
    assert!(is_diagonal(&Gate::S));
    assert!(is_diagonal(&Gate::T));
    assert!(is_diagonal(&Gate::Phase(1.0)));
    assert!(is_diagonal(&Gate::Rz(1.0)));
    assert!(!is_diagonal(&Gate::X));
    assert!(!is_diagonal(&Gate::Y));
    assert!(!is_diagonal(&Gate::H));
    assert!(!is_diagonal(&Gate::Rx(1.0)));
    assert!(!is_diagonal(&Gate::Ry(1.0)));
    assert!(!is_diagonal(&Gate::SWAP));
}

#[test]
fn test_extract_diagonal_phases() {
    use yao_rs::apply::extract_diagonal_phases;
    // Z gate: diag(1, -1)
    let z_matrix = Gate::Z.matrix(2);
    let phases = extract_diagonal_phases(&z_matrix);
    assert_eq!(phases.len(), 2);
    assert!((phases[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((phases[1] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);

    // S gate: diag(1, i)
    let s_matrix = Gate::S.matrix(2);
    let phases = extract_diagonal_phases(&s_matrix);
    assert_eq!(phases.len(), 2);
    assert!((phases[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((phases[1] - Complex64::new(0.0, 1.0)).norm() < 1e-10);
}

// ============================================================
// Ground truth validation against Yao.jl
// ============================================================

#[test]
fn test_apply_ground_truth() {
    let data = common::load_apply_data();
    let mut tested = 0;
    for case in &data.cases {
        let circuit = common::circuit_from_case(case);
        let input_state = if let (Some(re), Some(im)) = (&case.input_state_re, &case.input_state_im)
        {
            State::new(case.dims.clone(), common::state_from_json(re, im))
        } else {
            State::zero_state(&case.dims)
        };
        let result = apply(&circuit, &input_state);
        let expected = common::state_from_json(&case.output_state_re, &case.output_state_im);
        for (i, (r, e)) in result.data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).norm() < 1e-10,
                "Case '{}': mismatch at index {}: got {:?}, expected {:?}",
                case.label,
                i,
                r,
                e
            );
        }
        tested += 1;
    }
    assert_eq!(tested, 72, "Expected 72 apply cases in ground truth data");
}
