use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

/// Load instruct test data from tests/data/instruct.json
fn load_instruct_data() -> serde_json::Value {
    let data = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/instruct.json"
    ))
    .unwrap();
    serde_json::from_str(&data).unwrap()
}

fn parse_state(val: &serde_json::Value) -> Vec<Complex64> {
    val.as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let arr = pair.as_array().unwrap();
            Complex64::new(arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap())
        })
        .collect()
}

fn parse_matrix(val: &serde_json::Value) -> Vec<Vec<Complex64>> {
    val.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|pair| {
                    let arr = pair.as_array().unwrap();
                    Complex64::new(arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap())
                })
                .collect()
        })
        .collect()
}

fn states_approx_eq(a: &[Complex64], b: &[Complex64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < tol)
}

// ---------- 1q without controls ----------

#[test]
fn test_instruct_1q_from_julia_data() {
    use crate::instruct_qubit::instruct_1q;

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();

    for case in cases {
        let label = case["label"].as_str().unwrap();
        // Only test 1q cases without controls and with a gate_matrix (2x2)
        if !label.contains("1q") || case.get("ctrl_locs").is_some() {
            continue;
        }
        if case.get("gate_matrix").is_none() {
            continue;
        }

        let locs: Vec<usize> = case["locs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        if locs.len() != 1 {
            continue;
        }

        let mat = parse_matrix(&case["gate_matrix"]);
        if mat.len() != 2 {
            continue;
        }

        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);
        let loc = locs[0];

        let a = mat[0][0];
        let b = mat[0][1];
        let c = mat[1][0];
        let d = mat[1][1];

        instruct_1q(&mut state, loc, a, b, c, d);

        assert!(states_approx_eq(&state, &expected, 1e-10), "FAIL: {label}");
    }
}

#[test]
fn test_instruct_1q_diag_z_gate() {
    use crate::instruct_qubit::instruct_1q_diag;

    // Z gate on qubit 0: diag(1, -1)
    let d0 = Complex64::new(1.0, 0.0);
    let d1 = Complex64::new(-1.0, 0.0);
    let s = FRAC_1_SQRT_2;

    // |+> = [1/sqrt2, 1/sqrt2] -> Z -> [1/sqrt2, -1/sqrt2] = |->
    let mut state = vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0)];
    instruct_1q_diag(&mut state, 0, d0, d1);
    assert!((state[0] - Complex64::new(s, 0.0)).norm() < 1e-10);
    assert!((state[1] - Complex64::new(-s, 0.0)).norm() < 1e-10);
}

// ---------- 2q without controls ----------

#[test]
fn test_instruct_2q_from_julia_data() {
    use crate::instruct_qubit::instruct_2q;

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();

    for case in cases {
        let label = case["label"].as_str().unwrap();
        // Only test 2q cases without controls
        if !label.contains("2q") || case.get("ctrl_locs").is_some() {
            continue;
        }
        if case.get("gate_matrix").is_none() || case.get("locs").is_none() {
            continue;
        }

        let nbits = case["nbits"].as_u64().unwrap() as usize;
        let locs: Vec<usize> = case["locs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        if locs.len() != 2 {
            continue;
        }

        let mat_2d = parse_matrix(&case["gate_matrix"]);
        if mat_2d.len() != 4 {
            continue;
        }

        // Flatten matrix to row-major vec
        let gate: Vec<Complex64> = mat_2d.iter().flatten().cloned().collect();

        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);

        instruct_2q(&mut state, nbits, &locs, &gate);

        assert!(states_approx_eq(&state, &expected, 1e-10), "FAIL: {label}");
    }
}

// ---------- Controlled instruct ----------

#[test]
fn test_controlled_from_julia_data() {
    use crate::instruct_qubit::{instruct_1q_controlled, instruct_2q_controlled};

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();

    for case in cases {
        let label = case["label"].as_str().unwrap();
        if case.get("ctrl_locs").is_none() {
            continue;
        }
        if case.get("gate_matrix").is_none() || case.get("locs").is_none() {
            continue;
        }

        let nbits = case["nbits"].as_u64().unwrap() as usize;
        let locs: Vec<usize> = case["locs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let ctrl_locs: Vec<usize> = case["ctrl_locs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let ctrl_bits: Vec<usize> = case["ctrl_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let mat_2d = parse_matrix(&case["gate_matrix"]);

        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);

        if mat_2d.len() == 2 && locs.len() == 1 {
            let a = mat_2d[0][0];
            let b = mat_2d[0][1];
            let c = mat_2d[1][0];
            let d = mat_2d[1][1];
            instruct_1q_controlled(
                &mut state, nbits, locs[0], a, b, c, d, &ctrl_locs, &ctrl_bits,
            );
        } else if mat_2d.len() == 4 && locs.len() == 2 {
            let gate: Vec<Complex64> = mat_2d.iter().flatten().cloned().collect();
            instruct_2q_controlled(&mut state, nbits, &locs, &gate, &ctrl_locs, &ctrl_bits);
        } else {
            continue;
        }

        assert!(
            states_approx_eq(&state, &expected, 1e-10),
            "FAIL (controlled): {label}"
        );
    }
}

// ---------- Comprehensive ground truth ----------

/// Get gate matrix as a flat row-major Vec from either gate_matrix or gate_name.
fn get_gate_flat(case: &serde_json::Value) -> Option<Vec<Complex64>> {
    if let Some(gm) = case.get("gate_matrix") {
        let mat_2d = parse_matrix(gm);
        Some(mat_2d.iter().flatten().cloned().collect())
    } else if let Some(gn) = case.get("gate_name") {
        let name = gn.as_str().unwrap();
        let theta = case.get("theta").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let gate = match name {
            "X" => crate::Gate::X,
            "Y" => crate::Gate::Y,
            "Z" => crate::Gate::Z,
            "H" => crate::Gate::H,
            "S" => crate::Gate::S,
            "T" => crate::Gate::T,
            "SWAP" => crate::Gate::SWAP,
            "Rx" => crate::Gate::Rx(theta),
            "Ry" => crate::Gate::Ry(theta),
            "Rz" => crate::Gate::Rz(theta),
            // PSWAP/CPHASE don't exist in Gate enum; skip (covered by gate_matrix cases)
            "PSWAP" | "CPHASE" => return None,
            _ => return None,
        };
        let mat = gate.matrix();
        let d = mat.nrows();
        let mut flat = Vec::with_capacity(d * d);
        for i in 0..d {
            for j in 0..d {
                flat.push(mat[[i, j]]);
            }
        }
        Some(flat)
    } else {
        None
    }
}

/// Run ALL instruct.json test cases through the appropriate qubit instruct function.
#[test]
fn test_all_julia_ground_truth() {
    use crate::instruct_qubit::*;

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();
    let mut tested = 0;

    for case in cases {
        let label = case["label"].as_str().unwrap();

        // Skip regression (separate test) and probability cases
        if label.contains("regression") || label.contains("measure") {
            continue;
        }
        if case.get("locs").is_none() {
            continue;
        }

        let gate_flat = match get_gate_flat(case) {
            Some(g) => g,
            None => continue,
        };

        let nbits = case["nbits"].as_u64().unwrap() as usize;
        let locs: Vec<usize> = case["locs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let has_controls = case.get("ctrl_locs").is_some();
        let ctrl_locs: Vec<usize> = if has_controls {
            case["ctrl_locs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        } else {
            vec![]
        };
        let ctrl_bits: Vec<usize> = if has_controls {
            case["ctrl_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        } else {
            vec![]
        };

        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);

        let d_sq = gate_flat.len(); // 4 for 1q (2x2), 16 for 2q (4x4)

        match (d_sq, locs.len(), has_controls) {
            (4, 1, false) => {
                instruct_1q(
                    &mut state,
                    locs[0],
                    gate_flat[0],
                    gate_flat[1],
                    gate_flat[2],
                    gate_flat[3],
                );
            }
            (4, 1, true) => {
                instruct_1q_controlled(
                    &mut state,
                    nbits,
                    locs[0],
                    gate_flat[0],
                    gate_flat[1],
                    gate_flat[2],
                    gate_flat[3],
                    &ctrl_locs,
                    &ctrl_bits,
                );
            }
            (16, 2, false) => {
                instruct_2q(&mut state, nbits, &locs, &gate_flat);
            }
            (16, 2, true) => {
                instruct_2q_controlled(
                    &mut state, nbits, &locs, &gate_flat, &ctrl_locs, &ctrl_bits,
                );
            }
            _ => continue,
        }

        assert!(
            states_approx_eq(&state, &expected, 1e-8),
            "FAIL (ground truth): {label}\n  got:      {:?}\n  expected: {:?}",
            &state[..state.len().min(8)],
            &expected[..expected.len().min(8)],
        );
        tested += 1;
    }

    assert!(
        tested >= 40,
        "Expected at least 40 test cases, got {tested}"
    );
}

/// Test regression: 20 random 2q gates applied sequentially.
#[test]
fn test_regression_20_random_2q_gates() {
    use crate::instruct_qubit::instruct_2q;

    let data = load_instruct_data();
    let cases = data["cases"].as_array().unwrap();

    for case in cases {
        let label = case["label"].as_str().unwrap();
        if !label.contains("regression") {
            continue;
        }

        let nbits = case["nbits"].as_u64().unwrap() as usize;
        let gate_pairs = case["gate_pairs"].as_array().unwrap();
        let mat_2d = parse_matrix(&case["gate_matrix"]);
        let gate: Vec<Complex64> = mat_2d.iter().flatten().cloned().collect();

        let mut state = parse_state(&case["input_state"]);
        let expected = parse_state(&case["output_state"]);

        for pair in gate_pairs {
            let locs: Vec<usize> = pair
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            instruct_2q(&mut state, nbits, &locs, &gate);
        }

        assert!(states_approx_eq(&state, &expected, 1e-8), "FAIL: {label}");
    }
}

// ---------- Integration with apply ----------

#[test]
fn test_apply_inplace_uses_qubit_path() {
    use crate::*;

    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let reg = apply(&circuit, &ArrayReg::zero_state(2));

    let s = std::f64::consts::FRAC_1_SQRT_2;
    let expected_0 = Complex64::new(s, 0.0);
    assert!((reg.state_vec()[0] - expected_0).norm() < 1e-10);
    assert!(reg.state_vec()[1].norm() < 1e-10);
    assert!(reg.state_vec()[2].norm() < 1e-10);
    assert!((reg.state_vec()[3] - expected_0).norm() < 1e-10);
}

// ---------- Direct diagonal instruct tests ----------

#[test]
fn test_instruct_2q_diag_cz() {
    use crate::instruct_qubit::instruct_2q_diag;

    // CZ gate = diag(1, 1, 1, -1)
    // Apply to |++⟩ = 0.5 * (|00⟩ + |01⟩ + |10⟩ + |11⟩)
    let h = 0.5_f64;
    let mut state = vec![
        Complex64::new(h, 0.0),
        Complex64::new(h, 0.0),
        Complex64::new(h, 0.0),
        Complex64::new(h, 0.0),
    ];
    let diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    instruct_2q_diag(&mut state, 2, &[0, 1], &diag);
    assert!((state[0] - Complex64::new(h, 0.0)).norm() < 1e-10);
    assert!((state[1] - Complex64::new(h, 0.0)).norm() < 1e-10);
    assert!((state[2] - Complex64::new(h, 0.0)).norm() < 1e-10);
    assert!((state[3] - Complex64::new(-h, 0.0)).norm() < 1e-10);
}

#[test]
fn test_instruct_1q_diag_controlled_cz_via_ctrl() {
    use crate::instruct_qubit::instruct_1q_diag_controlled;

    // CZ as controlled-Z: Z on qubit 1 controlled by qubit 0
    // Apply to |++⟩ = 0.5 * (|00⟩ + |01⟩ + |10⟩ + |11⟩)
    let h = 0.5_f64;
    let mut state = vec![
        Complex64::new(h, 0.0),
        Complex64::new(h, 0.0),
        Complex64::new(h, 0.0),
        Complex64::new(h, 0.0),
    ];
    let d0 = Complex64::new(1.0, 0.0);
    let d1 = Complex64::new(-1.0, 0.0);
    instruct_1q_diag_controlled(&mut state, 2, 1, d0, d1, &[0], &[1]);
    // Same as CZ: only |11⟩ gets sign flip
    assert!((state[0] - Complex64::new(h, 0.0)).norm() < 1e-10);
    assert!((state[1] - Complex64::new(h, 0.0)).norm() < 1e-10);
    assert!((state[2] - Complex64::new(h, 0.0)).norm() < 1e-10);
    assert!((state[3] - Complex64::new(-h, 0.0)).norm() < 1e-10);
}

#[test]
fn test_instruct_1q_diag_controlled_active_low() {
    use crate::instruct_qubit::instruct_1q_diag_controlled;

    let mut state = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    instruct_1q_diag_controlled(
        &mut state,
        2,
        1,
        Complex64::new(10.0, 0.0),
        Complex64::new(20.0, 0.0),
        &[0],
        &[0],
    );

    assert_eq!(state[0], Complex64::new(10.0, 0.0));
    assert_eq!(state[1], Complex64::new(40.0, 0.0));
    assert_eq!(state[2], Complex64::new(3.0, 0.0));
    assert_eq!(state[3], Complex64::new(4.0, 0.0));
}

#[test]
fn test_instruct_1q_controlled_active_low() {
    use crate::instruct_qubit::instruct_1q_controlled;

    let mut state = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    instruct_1q_controlled(
        &mut state,
        2,
        1,
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        &[0],
        &[0],
    );

    assert_eq!(state[0], Complex64::new(2.0, 0.0));
    assert_eq!(state[1], Complex64::new(1.0, 0.0));
    assert_eq!(state[2], Complex64::new(3.0, 0.0));
    assert_eq!(state[3], Complex64::new(4.0, 0.0));
}

#[test]
fn test_instruct_2q_diag_controlled_3q() {
    use crate::instruct_qubit::instruct_2q_diag_controlled;

    // 3-qubit system: apply CZ on qubits 1,2 controlled by qubit 0
    // Only |1,1,1⟩ = index 7 gets the -1 phase
    let n = 8;
    let amp = Complex64::new(1.0 / (n as f64).sqrt(), 0.0);
    let mut state = vec![amp; n];
    let diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    instruct_2q_diag_controlled(&mut state, 3, &[1, 2], &diag, &[0], &[1]);
    // All amplitudes unchanged except |111⟩ (index 7) which gets -1
    for (i, s) in state.iter().enumerate() {
        if i == 7 {
            assert!((*s - (-amp)).norm() < 1e-10, "index {i} should be -amp");
        } else {
            assert!((*s - amp).norm() < 1e-10, "index {i} should be amp");
        }
    }
}

// ---------- >2 target qubit fallback via apply ----------

#[test]
fn test_apply_3q_custom_gate() {
    use crate::*;

    // Create a 3-qubit identity-like custom gate and verify it's a no-op
    let dim = 8;
    let mut mat = ndarray::Array2::zeros((dim, dim));
    for i in 0..dim {
        mat[[i, i]] = Complex64::new(1.0, 0.0);
    }
    let gate = Gate::Custom {
        matrix: mat,
        is_diagonal: false,
        label: "I8".to_string(),
    };
    let circuit = Circuit::new(vec![2, 2, 2], vec![put(vec![0, 1, 2], gate)]).unwrap();
    // |1,0,1⟩ = basis index 0b101 = 5
    let mut sv = vec![Complex64::new(0.0, 0.0); 8];
    sv[5] = Complex64::new(1.0, 0.0);
    let input = ArrayReg::from_vec(3, sv);
    let output = apply(&circuit, &input);

    // Identity should not change the state
    for i in 0..8 {
        assert!(
            (output.state_vec()[i] - input.state_vec()[i]).norm() < 1e-10,
            "3q identity gate changed amplitude at index {i}"
        );
    }
}
