use crate::common;

use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use yao_rs::apply::{apply, apply_inplace};
use yao_rs::circuit::{Circuit, CircuitElement, PositionedGate, control, put};
use yao_rs::gate::Gate;
use yao_rs::register::ArrayReg;

const ATOL: f64 = 1e-10;

fn assert_state_approx(result: &ArrayReg, expected: &[Complex64]) {
    assert_eq!(result.state_vec().len(), expected.len());
    for (i, (r, e)) in result.state_vec().iter().zip(expected.iter()).enumerate() {
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
    let state = ArrayReg::zero_state(dims.len());
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
    let state = ArrayReg::zero_state(dims.len());
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
    // |1,0> on 2 qubits: basis index = 1*2 + 0 = 2
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let state = ArrayReg::from_vec(2, vec![zero, zero, one, zero]);
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
    let state = ArrayReg::zero_state(dims.len()); // |00>
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
    let state = ArrayReg::zero_state(dims.len());
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
fn test_arrayreg_apply_inplace_bell_state() {
    let mut reg = ArrayReg::zero_state(2);
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            CircuitElement::Gate(PositionedGate::new(Gate::H, vec![0], vec![], vec![])),
            CircuitElement::Gate(PositionedGate::new(Gate::X, vec![1], vec![0], vec![true])),
        ],
    )
    .unwrap();

    apply_inplace(&circuit, &mut reg);

    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    assert_eq!(reg.state_vec(), &[s, zero, zero, s]);
}

#[test]
fn test_arrayreg_apply_inplace_x_gate() {
    let mut reg = ArrayReg::zero_state(1);
    let circuit = Circuit::new(
        vec![2],
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::X,
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();

    apply_inplace(&circuit, &mut reg);

    assert_eq!(
        reg.state_vec(),
        &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    );
}

#[test]
fn test_arrayreg_apply_inplace_swap_gate() {
    let mut reg = ArrayReg::from_vec(
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0, 1], Gate::SWAP)]).unwrap();

    apply_inplace(&circuit, &mut reg);

    assert_eq!(
        reg.state_vec(),
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    );
}

#[test]
fn test_arrayreg_apply_inplace_custom_three_qubit_gate() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let mut matrix = Array2::zeros((8, 8));
    for idx in 0..8 {
        matrix[[idx, idx]] = one;
    }
    matrix[[0, 0]] = zero;
    matrix[[7, 7]] = zero;
    matrix[[0, 7]] = one;
    matrix[[7, 0]] = one;

    let mut reg = ArrayReg::zero_state(3);
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![put(
            vec![0, 1, 2],
            Gate::Custom {
                matrix,
                is_diagonal: false,
                label: "flip-000-111".to_string(),
            },
        )],
    )
    .unwrap();

    apply_inplace(&circuit, &mut reg);

    assert_eq!(
        reg.state_vec(),
        &[zero, zero, zero, zero, zero, zero, zero, one,]
    );
}

#[test]
fn test_norm_preservation() {
    // Apply H then X then H on a single qubit, check norm is preserved
    let dims = vec![2];
    let state = ArrayReg::zero_state(dims.len());
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
fn test_x_on_second_qubit() {
    // X on qubit 1 of |00> -> |01>
    let dims = vec![2, 2];
    let state = ArrayReg::zero_state(dims.len()); // |00>
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

fn states_approx_equal(a: &ArrayReg, b: &ArrayReg, tol: f64) -> bool {
    if a.nqubits() != b.nqubits() {
        return false;
    }
    for i in 0..a.state_vec().len() {
        if (a.state_vec()[i] - b.state_vec()[i]).norm() > tol {
            return false;
        }
    }
    true
}

#[test]
fn test_apply_vs_apply_old_single_h() {
    // Single H gate on 2-qubit system
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::H)]).unwrap();

    let state = ArrayReg::zero_state(2);

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
    let amp = Complex64::new(1.0 / (8.0_f64).sqrt(), 0.0);
    let state = ArrayReg::from_vec(3, vec![amp; 8]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_phase_gate() {
    // Phase gate with arbitrary angle
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::Phase(PI / 3.0))]).unwrap();

    // Start with superposition
    let s = Complex64::new(0.5, 0.0);
    let state = ArrayReg::from_vec(2, vec![s; 4]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_rz_gate() {
    // Rz gate (diagonal)
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![1], Gate::Rz(PI / 4.0))]).unwrap();

    let s = Complex64::new(0.5, 0.0);
    let state = ArrayReg::from_vec(2, vec![s, s, s, s]);

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

    let state = ArrayReg::zero_state(2);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_cnot() {
    // CNOT gate
    let circuit = Circuit::new(vec![2, 2], vec![control(vec![0], vec![1], Gate::X)]).unwrap();

    // Test with |10>: basis index = 1*2 + 0 = 2
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let state = ArrayReg::from_vec(2, vec![zero, zero, one, zero]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_toffoli() {
    // Toffoli (CCX) gate
    let circuit = Circuit::new(vec![2, 2, 2], vec![control(vec![0, 1], vec![2], Gate::X)]).unwrap();

    // Test with |110>: basis index = 1*4 + 1*2 + 0 = 6
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let state = ArrayReg::from_vec(3, vec![zero, zero, zero, zero, zero, zero, one, zero]);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_controlled_phase() {
    // Controlled-Z gate
    let circuit = Circuit::new(vec![2, 2], vec![control(vec![0], vec![1], Gate::Z)]).unwrap();

    // Start with superposition
    let s = Complex64::new(0.5, 0.0);
    let state = ArrayReg::from_vec(2, vec![s; 4]);

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

    let state = ArrayReg::zero_state(2);

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

    let state = ArrayReg::zero_state(3);

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

    let state = ArrayReg::zero_state(4);

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

    let state = ArrayReg::zero_state(3);

    let result_new = apply(&circuit, &state);
    let result_old = common::apply_old(&circuit, &state);

    assert!(states_approx_equal(&result_new, &result_old, 1e-10));
}

#[test]
fn test_apply_vs_apply_old_swap_gate() {
    // SWAP gate (multi-target, no controls)
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0, 1], Gate::SWAP)]).unwrap();

    // Test with |10> -> |01>: basis index = 1*2 + 0 = 2
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let state = ArrayReg::from_vec(2, vec![zero, zero, one, zero]);

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

    let state = ArrayReg::zero_state(3);

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

    let mut state = ArrayReg::zero_state(2);
    apply_inplace(&circuit, &mut state);

    // Should be Bell state (|00> + |11>) / sqrt(2)
    let s = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    assert!((state.state[0] - s).norm() < 1e-10);
    assert!(state.state[1].norm() < 1e-10);
    assert!(state.state[2].norm() < 1e-10);
    assert!((state.state[3] - s).norm() < 1e-10);
}

#[test]
fn test_apply_preserves_input() {
    // Test that apply does not modify the input state
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::X)]).unwrap();

    let state = ArrayReg::zero_state(2);
    let state_clone = state.clone();
    let _result = apply(&circuit, &state);

    // Original state should be unchanged
    assert!(states_approx_equal(&state, &state_clone, 1e-10));
}

/// Test that diagonal gates (Z, S, T, Phase, Rz) apply correctly and
/// produce consistent results via the public apply API.
/// This validates diagonal gate handling without reaching into internal helpers.
#[test]
fn test_diagonal_gate_behavior_via_apply() {
    // Apply each diagonal gate to |+> = (|0> + |1>)/sqrt(2) and verify results
    let s = Complex64::new(FRAC_1_SQRT_2, 0.0);

    // Z|+> = (|0> - |1>)/sqrt(2)
    let circuit = Circuit::new(
        vec![2],
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::Z,
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();
    let state = ArrayReg::from_vec(1, vec![s, s]);
    let result = apply(&circuit, &state);
    assert!((result.state_vec()[0] - s).norm() < ATOL);
    assert!((result.state_vec()[1] + s).norm() < ATOL);

    // S|+> = (|0> + i|1>)/sqrt(2)
    let circuit = Circuit::new(
        vec![2],
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::S,
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();
    let result = apply(&circuit, &state);
    assert!((result.state_vec()[0] - s).norm() < ATOL);
    assert!((result.state_vec()[1] - Complex64::new(0.0, FRAC_1_SQRT_2)).norm() < ATOL);

    // Non-diagonal gates should also work fine (sanity check)
    let circuit = Circuit::new(
        vec![2],
        vec![CircuitElement::Gate(PositionedGate::new(
            Gate::X,
            vec![0],
            vec![],
            vec![],
        ))],
    )
    .unwrap();
    let state0 = ArrayReg::zero_state(1);
    let result = apply(&circuit, &state0);
    assert!((result.state_vec()[0]).norm() < ATOL);
    assert!((result.state_vec()[1] - Complex64::new(1.0, 0.0)).norm() < ATOL);
}

// ============================================================
// Ground truth validation against Yao.jl
// ============================================================

#[test]
fn test_apply_ground_truth() {
    let data = common::load_apply_data();
    let mut tested = 0;
    for case in &data.cases {
        // Skip qudit (non-qubit) cases — ArrayReg only supports d=2
        if !case.dims.iter().all(|&d| d == 2) {
            continue;
        }
        let circuit = common::circuit_from_case(case);
        let input_state = if let (Some(re), Some(im)) = (&case.input_state_re, &case.input_state_im)
        {
            ArrayReg::from_vec(case.dims.len(), common::state_from_json(re, im).to_vec())
        } else {
            ArrayReg::zero_state(case.dims.len())
        };
        let result = apply(&circuit, &input_state);
        let expected = common::state_from_json(&case.output_state_re, &case.output_state_im);
        for (i, (r, e)) in result.state_vec().iter().zip(expected.iter()).enumerate() {
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
    assert!(
        tested >= 60,
        "Expected at least 60 qubit apply cases (got {tested})"
    );
}
