use ndarray::Array2;
use num_complex::Complex64;
use yao_rs::circuit::{Circuit, CircuitError, PositionedGate, put, control};
use yao_rs::gate::Gate;

// ============================================================
// PositionedGate tests
// ============================================================

#[test]
fn test_positioned_gate_new() {
    let pg = PositionedGate::new(Gate::X, vec![0], vec![], vec![]);
    assert_eq!(pg.target_locs, vec![0]);
    assert_eq!(pg.control_locs, vec![]);
    assert_eq!(pg.control_configs, vec![]);
}

#[test]
fn test_positioned_gate_all_locs_no_controls() {
    let pg = PositionedGate::new(Gate::X, vec![1], vec![], vec![]);
    assert_eq!(pg.all_locs(), vec![1]);
}

#[test]
fn test_positioned_gate_all_locs_with_controls() {
    let pg = PositionedGate::new(Gate::X, vec![2], vec![0, 1], vec![true, true]);
    assert_eq!(pg.all_locs(), vec![0, 1, 2]);
}

#[test]
fn test_positioned_gate_all_locs_swap() {
    let pg = PositionedGate::new(Gate::SWAP, vec![0, 1], vec![], vec![]);
    assert_eq!(pg.all_locs(), vec![0, 1]);
}

// ============================================================
// Valid circuit tests
// ============================================================

#[test]
fn test_valid_single_qubit_gate() {
    let pg = PositionedGate::new(Gate::H, vec![0], vec![], vec![]);
    let circuit = Circuit::new(vec![2, 2], vec![pg]);
    assert!(circuit.is_ok());
    let c = circuit.unwrap();
    assert_eq!(c.num_sites(), 2);
    assert_eq!(c.total_dim(), 4);
}

#[test]
fn test_valid_cnot() {
    // CNOT = controlled-X: control on qubit 0, target on qubit 1
    let pg = PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]);
    let circuit = Circuit::new(vec![2, 2], vec![pg]);
    assert!(circuit.is_ok());
    let c = circuit.unwrap();
    assert_eq!(c.num_sites(), 2);
    assert_eq!(c.total_dim(), 4);
}

#[test]
fn test_valid_toffoli() {
    // Toffoli = doubly-controlled X: controls on 0,1, target on 2
    let pg = PositionedGate::new(Gate::X, vec![2], vec![0, 1], vec![true, true]);
    let circuit = Circuit::new(vec![2, 2, 2], vec![pg]);
    assert!(circuit.is_ok());
    let c = circuit.unwrap();
    assert_eq!(c.num_sites(), 3);
    assert_eq!(c.total_dim(), 8);
}

#[test]
fn test_valid_custom_qudit_gate_on_two_qutrits() {
    // 9x9 custom gate on two qutrits (d=3)
    let one = Complex64::new(1.0, 0.0);
    let mut m: Array2<Complex64> = Array2::zeros((9, 9));
    for i in 0..9 {
        m[[i, i]] = one;
    }
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: true,
    };
    let pg = PositionedGate::new(gate, vec![0, 1], vec![], vec![]);
    let circuit = Circuit::new(vec![3, 3], vec![pg]);
    assert!(circuit.is_ok());
    let c = circuit.unwrap();
    assert_eq!(c.num_sites(), 2);
    assert_eq!(c.total_dim(), 9);
}

#[test]
fn test_valid_multi_gate_circuit() {
    // H on qubit 0, then CNOT (control 0, target 1), then Z on qubit 1
    let pg1 = PositionedGate::new(Gate::H, vec![0], vec![], vec![]);
    let pg2 = PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]);
    let pg3 = PositionedGate::new(Gate::Z, vec![1], vec![], vec![]);
    let circuit = Circuit::new(vec![2, 2], vec![pg1, pg2, pg3]);
    assert!(circuit.is_ok());
    let c = circuit.unwrap();
    assert_eq!(c.num_sites(), 2);
    assert_eq!(c.total_dim(), 4);
    assert_eq!(c.gates.len(), 3);
}

#[test]
fn test_valid_swap_gate() {
    let pg = PositionedGate::new(Gate::SWAP, vec![0, 1], vec![], vec![]);
    let circuit = Circuit::new(vec![2, 2], vec![pg]);
    assert!(circuit.is_ok());
}

#[test]
fn test_valid_rotation_gates() {
    let pg1 = PositionedGate::new(Gate::Rx(1.0), vec![0], vec![], vec![]);
    let pg2 = PositionedGate::new(Gate::Ry(2.0), vec![1], vec![], vec![]);
    let pg3 = PositionedGate::new(Gate::Rz(3.0), vec![2], vec![], vec![]);
    let circuit = Circuit::new(vec![2, 2, 2], vec![pg1, pg2, pg3]);
    assert!(circuit.is_ok());
}

#[test]
fn test_valid_empty_circuit() {
    let circuit = Circuit::new(vec![2, 2, 2], vec![]);
    assert!(circuit.is_ok());
    let c = circuit.unwrap();
    assert_eq!(c.num_sites(), 3);
    assert_eq!(c.total_dim(), 8);
}

#[test]
fn test_valid_controlled_with_false_config() {
    // Control with config = false (control on |0>)
    let pg = PositionedGate::new(Gate::X, vec![1], vec![0], vec![false]);
    let circuit = Circuit::new(vec![2, 2], vec![pg]);
    assert!(circuit.is_ok());
}

// ============================================================
// Invalid circuit tests
// ============================================================

#[test]
fn test_invalid_qubit_gate_on_qutrit() {
    // Attempt to put a named (X) gate on a qutrit site
    let pg = PositionedGate::new(Gate::X, vec![0], vec![], vec![]);
    let result = Circuit::new(vec![3, 2], vec![pg]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::NamedGateTargetNotQubit { loc: 0, dim: 3 }
    );
}

#[test]
fn test_invalid_control_on_qutrit() {
    // Attempt to use a qutrit site as a control
    let pg = PositionedGate::new(Gate::X, vec![1], vec![0], vec![true]);
    let result = Circuit::new(vec![3, 2], vec![pg]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::ControlSiteNotQubit { loc: 0, dim: 3 }
    );
}

#[test]
fn test_invalid_overlapping_locs() {
    // Target and control at the same location
    let pg = PositionedGate::new(Gate::X, vec![0], vec![0], vec![true]);
    let result = Circuit::new(vec![2, 2], vec![pg]);
    assert!(result.is_err());
    match result.unwrap_err() {
        CircuitError::OverlappingLocs { overlapping } => {
            assert!(overlapping.contains(&0));
        }
        e => panic!("Expected OverlappingLocs, got {:?}", e),
    }
}

#[test]
fn test_invalid_loc_out_of_range() {
    // Location 5 on a 2-site circuit
    let pg = PositionedGate::new(Gate::X, vec![5], vec![], vec![]);
    let result = Circuit::new(vec![2, 2], vec![pg]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::LocOutOfRange {
            loc: 5,
            num_sites: 2
        }
    );
}

#[test]
fn test_invalid_control_loc_out_of_range() {
    // Control location out of range
    let pg = PositionedGate::new(Gate::X, vec![0], vec![3], vec![true]);
    let result = Circuit::new(vec![2, 2], vec![pg]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::LocOutOfRange {
            loc: 3,
            num_sites: 2
        }
    );
}

#[test]
fn test_invalid_matrix_size_mismatch() {
    // 4x4 custom gate applied to a single qubit (expects 2x2)
    let one = Complex64::new(1.0, 0.0);
    let mut m: Array2<Complex64> = Array2::zeros((4, 4));
    for i in 0..4 {
        m[[i, i]] = one;
    }
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: true,
    };
    let pg = PositionedGate::new(gate, vec![0], vec![], vec![]);
    let result = Circuit::new(vec![2, 2], vec![pg]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::MatrixSizeMismatch {
            expected: 2,
            actual: 4
        }
    );
}

#[test]
fn test_invalid_control_config_length_mismatch() {
    // 2 control locs but 1 config
    let pg = PositionedGate::new(Gate::X, vec![2], vec![0, 1], vec![true]);
    let result = Circuit::new(vec![2, 2, 2], vec![pg]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::ControlConfigLengthMismatch {
            control_locs_len: 2,
            control_configs_len: 1
        }
    );
}

#[test]
fn test_invalid_control_config_too_many() {
    // 1 control loc but 2 configs
    let pg = PositionedGate::new(Gate::X, vec![1], vec![0], vec![true, false]);
    let result = Circuit::new(vec![2, 2], vec![pg]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::ControlConfigLengthMismatch {
            control_locs_len: 1,
            control_configs_len: 2
        }
    );
}

// ============================================================
// Circuit methods tests
// ============================================================

#[test]
fn test_num_sites() {
    let circuit = Circuit::new(vec![2, 2, 2, 2], vec![]).unwrap();
    assert_eq!(circuit.num_sites(), 4);
}

#[test]
fn test_total_dim_qubits() {
    let circuit = Circuit::new(vec![2, 2, 2], vec![]).unwrap();
    assert_eq!(circuit.total_dim(), 8);
}

#[test]
fn test_total_dim_mixed() {
    // 1 qubit and 1 qutrit: total_dim = 2*3 = 6
    let circuit = Circuit::new(vec![2, 3], vec![]).unwrap();
    assert_eq!(circuit.total_dim(), 6);
}

#[test]
fn test_total_dim_qutrits() {
    let circuit = Circuit::new(vec![3, 3, 3], vec![]).unwrap();
    assert_eq!(circuit.total_dim(), 27);
}

// ============================================================
// CircuitError Display tests
// ============================================================

#[test]
fn test_circuit_error_display_control_config_mismatch() {
    let err = CircuitError::ControlConfigLengthMismatch {
        control_locs_len: 2,
        control_configs_len: 1,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("control_configs length"));
    assert!(msg.contains("1"));
    assert!(msg.contains("2"));
}

#[test]
fn test_circuit_error_display_loc_out_of_range() {
    let err = CircuitError::LocOutOfRange {
        loc: 5,
        num_sites: 3,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("5"));
    assert!(msg.contains("3"));
}

#[test]
fn test_circuit_error_display_overlapping_locs() {
    let err = CircuitError::OverlappingLocs {
        overlapping: vec![1, 2],
    };
    let msg = format!("{}", err);
    assert!(msg.contains("overlap"));
}

#[test]
fn test_circuit_error_display_control_not_qubit() {
    let err = CircuitError::ControlSiteNotQubit { loc: 1, dim: 3 };
    let msg = format!("{}", err);
    assert!(msg.contains("control site"));
    assert!(msg.contains("1"));
    assert!(msg.contains("3"));
}

#[test]
fn test_circuit_error_display_named_gate_not_qubit() {
    let err = CircuitError::NamedGateTargetNotQubit { loc: 0, dim: 4 };
    let msg = format!("{}", err);
    assert!(msg.contains("named gate"));
    assert!(msg.contains("0"));
    assert!(msg.contains("4"));
}

#[test]
fn test_circuit_error_display_matrix_size_mismatch() {
    let err = CircuitError::MatrixSizeMismatch {
        expected: 4,
        actual: 2,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("4"));
    assert!(msg.contains("2"));
}

#[test]
fn test_circuit_error_is_error_trait() {
    let err = CircuitError::LocOutOfRange {
        loc: 0,
        num_sites: 0,
    };
    // Verify it implements std::error::Error by calling source()
    let _: &dyn std::error::Error = &err;
}

// ============================================================
// Additional edge case tests
// ============================================================

#[test]
fn test_valid_custom_single_qutrit_gate() {
    // 3x3 custom gate on a single qutrit
    let one = Complex64::new(1.0, 0.0);
    let mut m: Array2<Complex64> = Array2::zeros((3, 3));
    for i in 0..3 {
        m[[i, i]] = one;
    }
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: true,
    };
    let pg = PositionedGate::new(gate, vec![0], vec![], vec![]);
    let circuit = Circuit::new(vec![3], vec![pg]);
    assert!(circuit.is_ok());
    let c = circuit.unwrap();
    assert_eq!(c.num_sites(), 1);
    assert_eq!(c.total_dim(), 3);
}

#[test]
fn test_valid_custom_gate_with_qubit_control() {
    // Custom 3x3 gate on qutrit with qubit control
    let one = Complex64::new(1.0, 0.0);
    let mut m: Array2<Complex64> = Array2::zeros((3, 3));
    for i in 0..3 {
        m[[i, i]] = one;
    }
    let gate = Gate::Custom {
        matrix: m,
        is_diagonal: false,
    };
    let pg = PositionedGate::new(gate, vec![1], vec![0], vec![true]);
    let circuit = Circuit::new(vec![2, 3], vec![pg]);
    assert!(circuit.is_ok());
}

#[test]
fn test_invalid_named_h_gate_on_qutrit() {
    let pg = PositionedGate::new(Gate::H, vec![1], vec![], vec![]);
    let result = Circuit::new(vec![2, 3], vec![pg]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::NamedGateTargetNotQubit { loc: 1, dim: 3 }
    );
}

#[test]
fn test_second_gate_fails_validation() {
    // First gate is valid, second is invalid
    let pg1 = PositionedGate::new(Gate::H, vec![0], vec![], vec![]);
    let pg2 = PositionedGate::new(Gate::X, vec![5], vec![], vec![]);
    let result = Circuit::new(vec![2, 2], vec![pg1, pg2]);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        CircuitError::LocOutOfRange {
            loc: 5,
            num_sites: 2
        }
    );
}

#[test]
fn test_clone_positioned_gate() {
    let pg = PositionedGate::new(Gate::X, vec![0], vec![1], vec![true]);
    let pg2 = pg.clone();
    assert_eq!(pg2.target_locs, vec![0]);
    assert_eq!(pg2.control_locs, vec![1]);
    assert_eq!(pg2.control_configs, vec![true]);
}

#[test]
fn test_clone_circuit() {
    let pg = PositionedGate::new(Gate::H, vec![0], vec![], vec![]);
    let circuit = Circuit::new(vec![2, 2], vec![pg]).unwrap();
    let circuit2 = circuit.clone();
    assert_eq!(circuit2.num_sites(), 2);
    assert_eq!(circuit2.total_dim(), 4);
    assert_eq!(circuit2.gates.len(), 1);
}

// === put() and control() builder tests ===

#[test]
fn test_put_single_gate() {
    let pg = put(vec![0], Gate::H);
    assert_eq!(pg.target_locs, vec![0]);
    assert!(pg.control_locs.is_empty());
    assert!(pg.control_configs.is_empty());
}

#[test]
fn test_put_multi_site_gate() {
    let pg = put(vec![1, 3], Gate::SWAP);
    assert_eq!(pg.target_locs, vec![1, 3]);
    assert!(pg.control_locs.is_empty());
}

#[test]
fn test_control_cnot() {
    let pg = control(vec![0], vec![1], Gate::X);
    assert_eq!(pg.control_locs, vec![0]);
    assert_eq!(pg.target_locs, vec![1]);
    assert_eq!(pg.control_configs, vec![true]);
}

#[test]
fn test_control_toffoli() {
    let pg = control(vec![0, 1], vec![2], Gate::X);
    assert_eq!(pg.control_locs, vec![0, 1]);
    assert_eq!(pg.target_locs, vec![2]);
    assert_eq!(pg.control_configs, vec![true, true]);
}

#[test]
fn test_put_in_circuit() {
    let gates = vec![
        put(vec![0], Gate::H),
        put(vec![1], Gate::X),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    assert_eq!(circuit.gates.len(), 2);
}

#[test]
fn test_control_in_circuit() {
    let gates = vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    assert_eq!(circuit.gates.len(), 2);
}

#[test]
fn test_qft_circuit_builds() {
    use std::f64::consts::PI;
    let n = 4;
    let mut gates: Vec<PositionedGate> = Vec::new();
    for i in 0..n {
        gates.push(put(vec![i], Gate::H));
        for j in 1..(n - i) {
            let theta = 2.0 * PI / (1 << (j + 1)) as f64;
            gates.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
    }
    for i in 0..(n / 2) {
        gates.push(PositionedGate::new(Gate::SWAP, vec![i, n - 1 - i], vec![], vec![]));
    }
    let circuit = Circuit::new(vec![2; n], gates).unwrap();
    assert_eq!(circuit.num_sites(), 4);
    assert_eq!(circuit.gates.len(), 12); // 4 H + 3+2+1 CPhase + 2 SWAP = 12
}
