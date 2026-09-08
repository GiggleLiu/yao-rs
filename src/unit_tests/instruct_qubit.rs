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

#[test]
fn single_qubit_kernels_match_dense_action_at_every_stride() {
    use crate::instruct_qubit::{instruct_1q, instruct_1q_diag};
    use num_complex::Complex64 as C;

    // Nonunitary matrices also exercise the kernels used by Kraus channels.
    let matrices = [
        [
            C::new(1.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(-0.5, 0.7),
        ],
        [
            C::new(0.3, 0.0),
            C::new(0.0, 0.4),
            C::new(0.0, -0.7),
            C::new(-0.2, 0.0),
        ],
        [
            C::new(0.3, 0.0),
            C::new(-0.4, 0.0),
            C::new(0.7, 0.0),
            C::new(0.2, 0.0),
        ],
        [
            C::new(0.2, 0.3),
            C::new(-0.4, 0.5),
            C::new(0.6, -0.7),
            C::new(0.8, 0.1),
        ],
        [
            C::new(0.2, 0.3),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(-0.5, 0.7),
        ],
    ];
    for n in 1..=7 {
        let size = 1 << n;
        let initial: Vec<C> = (0..size)
            .map(|k| C::new((k as f64 * 0.17).sin(), (k as f64 * 0.31).cos()))
            .collect();
        for loc in 0..n {
            let mask = 1 << (n - 1 - loc);
            for matrix in matrices {
                let expected: Vec<C> = (0..size)
                    .map(|row| {
                        (0..size)
                            .filter(|&column| row & !mask == column & !mask)
                            .map(|column| {
                                let r = usize::from(row & mask != 0);
                                let c = usize::from(column & mask != 0);
                                matrix[2 * r + c] * initial[column]
                            })
                            .sum()
                    })
                    .collect();
                let mut actual = initial.clone();
                instruct_1q(&mut actual, loc, matrix[0], matrix[1], matrix[2], matrix[3]);
                for (got, want) in actual.iter().zip(&expected) {
                    assert!((*got - *want).norm() < 1e-12, "n={n}, loc={loc}");
                }
                if matrix[1] == C::new(0., 0.) && matrix[2] == C::new(0., 0.) {
                    let mut actual = initial.clone();
                    instruct_1q_diag(&mut actual, loc, matrix[0], matrix[3]);
                    for (got, want) in actual.iter().zip(&expected) {
                        assert!((*got - *want).norm() < 1e-12, "diagonal n={n}, loc={loc}");
                    }
                }
            }
        }
    }
}

#[test]
fn two_qubit_blocks_match_dense_action_for_all_target_orders_and_controls() {
    use crate::instruct_qubit::{instruct_2q, instruct_2q_controlled};
    use num_complex::Complex64 as C;
    let zero = C::new(0., 0.);
    // Asymmetric and nonunitary: catches target-order errors and assumptions
    // that the |00> and |11> amplitudes are unchanged.
    let mut sparse = [zero; 16];
    for (index, value) in [
        (0, C::new(0.7, 0.2)),
        (5, C::new(0.1, -0.3)),
        (6, C::new(-0.4, 0.6)),
        (9, C::new(0.2, 0.5)),
        (10, C::new(0.8, -0.2)),
        (15, C::new(-0.3, 0.4)),
    ] {
        sparse[index] = value;
    }
    let mut dense = sparse;
    dense[3] = C::new(0.1, -0.2);
    dense[12] = C::new(0.3, 0.2);
    for n in 2..=5 {
        let size = 1 << n;
        let initial: Vec<C> = (0..size)
            .map(|k| C::new((k as f64 * 0.13).sin(), (k as f64 * 0.19).cos()))
            .collect();
        for first in 0..n {
            for second in 0..n {
                if first == second {
                    continue;
                }
                let locs = [first, second];
                let masks = [1 << (n - 1 - first), 1 << (n - 1 - second)];
                let target_mask = masks[0] | masks[1];
                let local =
                    |i: usize| usize::from(i & masks[0] != 0) + 2 * usize::from(i & masks[1] != 0);
                let mut controls = vec![None];
                for control in (0..n).filter(|q| !locs.contains(q)) {
                    controls.extend([Some((control, 0)), Some((control, 1))]);
                }
                for control in controls {
                    for matrix in [sparse, dense] {
                        let expected: Vec<C> = (0..size)
                            .map(|row| {
                                if let Some((q, value)) = control
                                    && (row >> (n - 1 - q)) & 1 != value
                                {
                                    return initial[row];
                                }
                                (0..size)
                                    .filter(|&column| row & !target_mask == column & !target_mask)
                                    .map(|column| {
                                        matrix[4 * local(row) + local(column)] * initial[column]
                                    })
                                    .sum()
                            })
                            .collect();
                        let mut actual = initial.clone();
                        if let Some((q, value)) = control {
                            instruct_2q_controlled(&mut actual, n, &locs, &matrix, &[q], &[value]);
                        } else {
                            instruct_2q(&mut actual, n, &locs, &matrix);
                        }
                        for (got, want) in actual.iter().zip(&expected) {
                            assert!(
                                (*got - *want).norm() < 1e-12,
                                "n={n}, targets={locs:?}, control={control:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn permutation_kernels_match_basis_permutations_at_every_position() {
    use crate::instruct_qubit::{instruct_swap, instruct_x};
    use num_complex::Complex64 as C;
    for n in 2..=7 {
        let initial: Vec<C> = (0..1usize << n)
            .map(|k| C::new(k as f64, -(k as f64) - 0.5))
            .collect();
        for first in 0..n {
            let a = 1 << (n - 1 - first);
            let mut actual = initial.clone();
            instruct_x(&mut actual, n, first);
            for (row, got) in actual.iter().enumerate() {
                assert_eq!(*got, initial[row ^ a]);
            }
            for second in 0..n {
                if first == second {
                    continue;
                }
                let b = 1 << (n - 1 - second);
                let mut actual = initial.clone();
                instruct_swap(&mut actual, n, &[first, second]);
                for (row, got) in actual.iter().enumerate() {
                    let column = if (row & a != 0) == (row & b != 0) {
                        row
                    } else {
                        row ^ a ^ b
                    };
                    assert_eq!(*got, initial[column], "n={n}, targets={first},{second}");
                }
            }
        }
    }
}

#[test]
fn custom_matrix_application_preserves_row_order_for_strided_storage() {
    use crate::{ArrayReg, Circuit, Gate, apply, put};
    use ndarray::{Array2, s};
    use num_complex::Complex64 as C;
    for n in 2..=4 {
        let dimension = 1usize << n;
        let base = Array2::from_shape_fn((dimension, dimension), |(r, c)| {
            C::new(
                ((r * 7 + c) as f64 * 0.11).sin(),
                ((r + c * 3) as f64 * 0.17).cos(),
            )
        });
        let input: Vec<C> = (0..dimension)
            .map(|i| C::new(i as f64 * 0.13, 0.5 - i as f64 * 0.07))
            .collect();
        for matrix in [
            base.clone(),
            base.clone().reversed_axes(),
            base.slice_move(s![..;-1, ..]),
        ] {
            let expected: Vec<C> = (0..dimension)
                .map(|r| (0..dimension).map(|c| matrix[[r, c]] * input[c]).sum())
                .collect();
            let gate = Gate::Custom {
                matrix,
                is_diagonal: false,
                label: "strided".into(),
            };
            let circuit = Circuit::qubits(n, vec![put((0..n).rev().collect(), gate)]).unwrap();
            let got = apply(&circuit, &ArrayReg::from_vec(n, input.clone()));
            for (actual, expected) in got.state_vec().iter().zip(expected) {
                assert!((*actual - expected).norm() < 1e-12, "n={n}");
            }
        }
    }
}

#[test]
fn generic_gate_reuses_scratch_across_free_bits_and_mixed_controls() {
    use crate::instruct_qubit::instruct_nq;
    use num_complex::Complex64 as C;
    let n = 6;
    let initial: Vec<C> = (0..64)
        .map(|i| C::new((i as f64 * 0.2).sin(), (i as f64 * 0.3).cos()))
        .collect();
    let matrix: Vec<C> = (0..64)
        .map(|i| C::new((i as f64 * 0.17).sin(), (i as f64 * 0.11).cos()))
        .collect();
    for targets in [[0, 4, 2], [2, 4, 0]] {
        let mask = targets.iter().fold(0, |acc, &q| acc | (1 << (n - 1 - q)));
        let local = |basis: usize| {
            targets.iter().enumerate().fold(0usize, |acc, (i, &q)| {
                acc | (((basis >> (n - 1 - q)) & 1) << i)
            })
        };
        for values in [[0, 1], [1, 0]] {
            let active = |basis: usize| (basis & 1) == values[0] && ((basis >> 4) & 1) == values[1];
            let expected: Vec<C> = (0..64)
                .map(|row| {
                    if !active(row) {
                        return initial[row];
                    }
                    (0..64)
                        .filter(|&col| row & !mask == col & !mask)
                        .map(|col| matrix[8 * local(row) + local(col)] * initial[col])
                        .sum()
                })
                .collect();
            let mut actual = initial.clone();
            instruct_nq(&mut actual, n, &targets, &matrix, &[5, 1], &values);
            for (got, want) in actual.iter().zip(expected) {
                assert!((*got - want).norm() < 1e-12);
            }
        }
    }
}

#[test]
fn controlled_phase_preserves_every_unselected_amplitude() {
    use crate::instruct_qubit::instruct_1q_diag_controlled;
    let initial: Vec<Complex64> = (0..16)
        .map(|i| Complex64::new(i as f64 * 0.1, 0.3 - i as f64 * 0.07))
        .collect();
    let phase = Complex64::from_polar(1., 0.37);
    let mut expected = initial.clone();
    // q0=0, q3=1, and target q2=1: only |0011> and |0111> gain a phase.
    for index in [3, 7] {
        expected[index] *= phase;
    }
    let mut actual = initial;
    instruct_1q_diag_controlled(
        &mut actual,
        4,
        2,
        Complex64::new(1., 0.),
        phase,
        &[0, 3],
        &[0, 1],
    );
    assert_eq!(actual, expected);
}
