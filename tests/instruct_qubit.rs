use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

mod common;

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
    use yao_rs::instruct_qubit::instruct_1q;

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

        assert!(
            states_approx_eq(&state, &expected, 1e-10),
            "FAIL: {label}"
        );
    }
}

#[test]
fn test_instruct_1q_diag_z_gate() {
    use yao_rs::instruct_qubit::instruct_1q_diag;

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
