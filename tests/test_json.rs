use yao_rs::{Gate, Circuit, put, control, circuit_to_json, circuit_from_json};
use ndarray::Array2;
use num_complex::Complex64;
use approx::assert_abs_diff_eq;

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
