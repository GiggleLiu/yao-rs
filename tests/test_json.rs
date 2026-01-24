use yao_rs::{Gate, Circuit, put, control, circuit_to_json, circuit_from_json};
use ndarray::Array2;
use num_complex::Complex64;
use approx::assert_abs_diff_eq;
use std::f64::consts::PI;

#[test]
fn test_roundtrip_named_gates() {
    let gates = vec![
        put(vec![0], Gate::H),
        put(vec![1], Gate::X),
        put(vec![0], Gate::Phase(1.5)),
        put(vec![1], Gate::Rx(0.5)),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    assert_eq!(restored.num_sites(), 2);
    assert_eq!(restored.gates.len(), 4);
}

#[test]
fn test_roundtrip_controlled_gate() {
    let gates = vec![control(vec![0], vec![1], Gate::X)];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    assert_eq!(restored.gates[0].control_locs, vec![0]);
    assert_eq!(restored.gates[0].target_locs, vec![1]);
    assert_eq!(restored.gates[0].control_configs, vec![true]);
}

#[test]
fn test_roundtrip_custom_gate() {
    let matrix = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
    ]).unwrap();
    let gates = vec![put(vec![0], Gate::Custom {
        matrix: matrix.clone(), is_diagonal: false, label: "MyGate".to_string(),
    })];
    let circuit = Circuit::new(vec![2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    if let Gate::Custom { matrix: m, is_diagonal, label } = &restored.gates[0].gate {
        assert_eq!(label, "MyGate");
        assert!(!is_diagonal);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(m[[i,j]].re, matrix[[i,j]].re, epsilon = 1e-15);
                assert_abs_diff_eq!(m[[i,j]].im, matrix[[i,j]].im, epsilon = 1e-15);
            }
        }
    } else {
        panic!("Expected Custom gate");
    }
}

#[test]
fn test_roundtrip_fsim() {
    let gates = vec![put(vec![0, 1], Gate::FSim(1.0, 0.5))];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    if let Gate::FSim(theta, phi) = &restored.gates[0].gate {
        assert_abs_diff_eq!(*theta, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(*phi, 0.5, epsilon = 1e-15);
    } else {
        panic!("Expected FSim gate");
    }
}

#[test]
fn test_roundtrip_new_gates() {
    let gates = vec![
        put(vec![0], Gate::SqrtX),
        put(vec![0], Gate::SqrtY),
        put(vec![0], Gate::SqrtW),
        put(vec![0, 1], Gate::ISWAP),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    assert_eq!(restored.gates.len(), 4);
}

#[test]
fn test_json_structure() {
    let gates = vec![put(vec![0], Gate::H)];
    let circuit = Circuit::new(vec![2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["num_qubits"], 1);
    assert_eq!(parsed["gates"][0]["gate"], "H");
    assert_eq!(parsed["gates"][0]["targets"][0], 0);
    // controls should not be present for non-controlled gates
    assert!(parsed["gates"][0]["controls"].is_null());
}

// ============================================================
// Roundtrip tests for all remaining gate types
// ============================================================

#[test]
fn test_roundtrip_all_named_gates() {
    let gates = vec![
        put(vec![0], Gate::Y),
        put(vec![0], Gate::Z),
        put(vec![0], Gate::S),
        put(vec![0], Gate::T),
        put(vec![0, 1], Gate::SWAP),
    ];
    let circuit = Circuit::new(vec![2, 2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    assert_eq!(restored.gates.len(), 5);

    // Verify gate types are preserved
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["gates"][0]["gate"], "Y");
    assert_eq!(parsed["gates"][1]["gate"], "Z");
    assert_eq!(parsed["gates"][2]["gate"], "S");
    assert_eq!(parsed["gates"][3]["gate"], "T");
    assert_eq!(parsed["gates"][4]["gate"], "SWAP");
}

#[test]
fn test_roundtrip_ry_rz() {
    let gates = vec![
        put(vec![0], Gate::Ry(PI / 3.0)),
        put(vec![0], Gate::Rz(PI / 6.0)),
    ];
    let circuit = Circuit::new(vec![2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    if let Gate::Ry(theta) = &restored.gates[0].gate {
        assert_abs_diff_eq!(*theta, PI / 3.0, epsilon = 1e-15);
    } else {
        panic!("Expected Ry gate");
    }
    if let Gate::Rz(theta) = &restored.gates[1].gate {
        assert_abs_diff_eq!(*theta, PI / 6.0, epsilon = 1e-15);
    } else {
        panic!("Expected Rz gate");
    }
}

// ============================================================
// Deserialization error path tests
// ============================================================

#[test]
fn test_from_json_malformed() {
    let result = circuit_from_json("not valid json{{{");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("JSON parse error"));
}

#[test]
fn test_from_json_unknown_gate() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "FooBar", "targets": [0]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown gate type: FooBar"));
}

#[test]
fn test_from_json_phase_missing_params() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "Phase", "targets": [0]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Phase gate requires params"));
}

#[test]
fn test_from_json_phase_empty_params() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "Phase", "targets": [0], "params": []}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Phase gate requires at least 1 parameter"));
}

#[test]
fn test_from_json_rx_missing_params() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "Rx", "targets": [0]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Rx gate requires params"));
}

#[test]
fn test_from_json_ry_missing_params() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "Ry", "targets": [0]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Ry gate requires params"));
}

#[test]
fn test_from_json_rz_missing_params() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "Rz", "targets": [0]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Rz gate requires params"));
}

#[test]
fn test_from_json_fsim_missing_params() {
    let json = r#"{"num_qubits": 2, "gates": [{"gate": "FSim", "targets": [0, 1]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("FSim gate requires params"));
}

#[test]
fn test_from_json_fsim_insufficient_params() {
    let json = r#"{"num_qubits": 2, "gates": [{"gate": "FSim", "targets": [0, 1], "params": [1.0]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("FSim gate requires 2 parameters"));
}

#[test]
fn test_from_json_custom_missing_matrix() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "Custom", "targets": [0]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Custom gate requires matrix"));
}

#[test]
fn test_from_json_custom_empty_matrix() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "Custom", "targets": [0], "matrix": []}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Custom gate matrix cannot be empty"));
}

#[test]
fn test_from_json_custom_ragged_matrix() {
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "Custom", "targets": [0], "matrix": [[[1,0],[0,0]], [[0,0]]]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Custom gate matrix rows must have equal length"));
}

#[test]
fn test_from_json_circuit_validation_error() {
    // Target out of range triggers circuit validation error
    let json = r#"{"num_qubits": 1, "gates": [{"gate": "H", "targets": [5]}]}"#;
    let result = circuit_from_json(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Circuit validation error"));
}

#[test]
fn test_roundtrip_custom_diagonal() {
    let matrix = Array2::from_shape_vec((2, 2), vec![
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0),
    ]).unwrap();
    let gates = vec![put(vec![0], Gate::Custom {
        matrix: matrix.clone(), is_diagonal: true, label: "DiagGate".to_string(),
    })];
    let circuit = Circuit::new(vec![2], gates).unwrap();
    let json = circuit_to_json(&circuit);
    let restored = circuit_from_json(&json).unwrap();
    if let Gate::Custom { is_diagonal, label, .. } = &restored.gates[0].gate {
        assert!(is_diagonal);
        assert_eq!(label, "DiagGate");
    } else {
        panic!("Expected Custom gate");
    }
}

#[test]
fn test_roundtrip_control_configs_default() {
    // When control_configs is not in JSON, it should default to all-true
    let json = r#"{"num_qubits": 3, "gates": [{"gate": "X", "targets": [2], "controls": [0, 1]}]}"#;
    let restored = circuit_from_json(json).unwrap();
    assert_eq!(restored.gates[0].control_configs, vec![true, true]);
}
