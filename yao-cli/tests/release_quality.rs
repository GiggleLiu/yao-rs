//! User-facing regression tests for simulation and malformed inputs.
use assert_cmd::Command;
use serde_json::{Value, json};

fn yao(args: &[&str], input: impl AsRef<[u8]>) -> std::process::Output {
    Command::cargo_bin("yao")
        .unwrap()
        .args(args)
        .write_stdin(input.as_ref())
        .output()
        .unwrap()
}

fn success(args: &[&str], input: impl AsRef<[u8]>) -> Vec<u8> {
    let out = yao(args, input);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn failure(args: &[&str], input: impl AsRef<[u8]>, expected: &str) {
    let out = yao(args, input);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "unexpected success");
    assert!(stderr.contains(expected), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

fn circuit(elements: Value) -> String {
    json!({"num_qubits": 2, "elements": elements}).to_string()
}

#[test]
fn noisy_simulation_pipelines_and_seed_agree() {
    // |10> followed by bit flip on q0: P(00)=1/4, P(10)=3/4.
    let input = circuit(json!([
        {"type":"gate", "gate":"X", "targets":[0]},
        {"type":"channel", "channel":"BitFlip", "p":0.25, "locs":[0]}
    ]));
    let state = success(&["simulate", "-"], &input);
    let header: Value =
        serde_json::from_slice(state.split(|&b| b == b'\n').next().unwrap()).unwrap();
    assert_eq!(header["format"], "yao-density-v1");
    assert_eq!(header["num_elements"], 16);
    let probs: Value = serde_json::from_slice(&success(&["probs", "-"], &state)).unwrap();
    for (actual, expected) in probs["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .zip([0.25, 0.0, 0.75, 0.0])
    {
        assert!((actual.as_f64().unwrap() - expected).abs() < 1e-12);
    }
    for (op, expected) in [("Z(0)", -0.5), ("Z(1)", 1.0), ("Z(0) + Z(1)", 0.5)] {
        let direct: Value =
            serde_json::from_slice(&success(&["run", "-", "--op", op], &input)).unwrap();
        let piped: Value =
            serde_json::from_slice(&success(&["expect", "-", "--op", op], &state)).unwrap();
        assert_eq!(direct, piped);
        assert!((direct["expectation_value"]["re"].as_f64().unwrap() - expected).abs() < 1e-12);
    }
    let direct = success(
        &[
            "run", "-", "--shots", "512", "--seed", "42", "--locs", "1,0",
        ],
        &input,
    );
    let piped = success(
        &[
            "measure", "-", "--shots", "512", "--seed", "42", "--locs", "1,0",
        ],
        &state,
    );
    assert_eq!(direct, piped);
    assert_eq!(
        direct,
        success(
            &[
                "run", "-", "--shots", "512", "--seed", "42", "--locs", "1,0"
            ],
            &input
        )
    );
    let data: Value = serde_json::from_slice(&direct).unwrap();
    let counts = data["counts"].as_object().unwrap();
    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts.values().map(|n| n.as_u64().unwrap()).sum::<u64>(),
        512
    );
    assert!(counts["01"].as_u64().unwrap() > 300);
}

#[test]
fn damping_and_entanglement_match_analytic_results() {
    let input = circuit(json!([
        {"type":"gate", "gate":"H", "targets":[0]},
        {"type":"gate", "gate":"X", "targets":[1], "controls":[0]},
        {"type":"channel", "channel":"AmplitudeDamping", "gamma":0.4, "excited_population":0.0, "locs":[0]}
    ]));
    let state = success(&["simulate", "-"], &input);
    let p: Value = serde_json::from_slice(&success(&["probs", "-"], &state)).unwrap();
    for (actual, expected) in p["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .zip([0.5, 0.2, 0.0, 0.3])
    {
        assert!((actual.as_f64().unwrap() - expected).abs() < 1e-12);
    }
    let e: Value =
        serde_json::from_slice(&success(&["expect", "-", "--op", "X(0)X(1)"], state)).unwrap();
    assert!((e["expectation_value"]["re"].as_f64().unwrap() - 0.6f64.sqrt()).abs() < 1e-12);
}

#[test]
fn invalid_locations_and_operators_are_errors() {
    let input = circuit(json!([]));
    for op in ["Z(99)", "I(99)"] {
        failure(&["run", "-", "--op", op], &input, "out of range");
    }
    failure(
        &["run", "-", "--op", "X(0)Y(0)"],
        &input,
        "Duplicate operator site",
    );
    failure(&["run", "-", "--op", "NaN*Z(0)"], &input, "finite");
    failure(&["run", "-", "--op", "X😀😀😀"], &input, "Expected '('");
    let state = success(&["simulate", "-"], &input);
    failure(&["probs", "-", "--locs", "2"], &state, "out of range");
    failure(
        &["measure", "-", "--locs", "0,0"],
        &state,
        "Duplicate measurement",
    );
    failure(
        &["run", "-", "--shots", "0", "--locs", "2"],
        &input,
        "out of range",
    );
    failure(
        &["toeinsum", "-", "--op", "Z(0) + Z(2)"],
        &input,
        "out of range",
    );
}

#[test]
fn malformed_state_headers_fail_without_panicking() {
    for (n, dims, elements, expected) in [
        (3, vec![2, 2], 4, "num_qubits"),
        (2, vec![2, 2], 999, "num_elements"),
        (2, vec![2, 3], 6, "Only qubit"),
        (
            usize::BITS as usize,
            vec![2; usize::BITS as usize],
            0,
            "Too many qubits",
        ),
        (2, vec![2, 2], 4, "Truncated"),
    ] {
        let input = format!(
            "{}\n",
            json!({"format":"yao-state-v1", "num_qubits":n, "dims":dims, "num_elements":elements, "dtype":"complex128"})
        );
        failure(&["probs", "-"], input, expected);
    }
}

#[test]
fn pure_state_format_is_backward_compatible_and_seeded() {
    let input = circuit(json!([{"type":"gate", "gate":"H", "targets":[0]}]));
    let state = success(&["simulate", "-"], &input);
    assert!(
        std::str::from_utf8(state.split(|&b| b == b'\n').next().unwrap())
            .unwrap()
            .contains("yao-state-v1")
    );
    let expected = success(&["measure", "-", "--shots", "100", "--seed", "7"], &state);
    assert_eq!(
        expected,
        success(&["run", "-", "--shots", "100", "--seed", "7"], &input)
    );
}

#[cfg(any(feature = "omeinsum", feature = "tenferro"))]
#[test]
fn malformed_tensor_networks_are_errors() {
    let valid = json!({"format":"yao-tn-v1", "mode":"pure",
        "eincode":{"input_indices":[["0"]], "output_indices":["0"]},
        "tensors":[{"shape":[2], "data_re":[1.0,0.0], "data_im":[0.0,0.0]}],
        "size_dict":{"0":2}});
    for (pointer, value, expected) in [
        (
            "/format",
            json!("future-version"),
            "Unknown tensor-network format",
        ),
        ("/tensors/0/data_im", json!([0.0]), "lengths differ"),
        ("/size_dict/0", json!(3), "dimension disagrees"),
        (
            "/eincode/output_indices",
            json!(["99"]),
            "absent from inputs",
        ),
        ("/eincode/input_indices", json!([]), "tensor count"),
        (
            "/eincode/output_indices",
            json!(["0", "0"]),
            "Duplicate output",
        ),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(pointer).unwrap() = value;
        failure(&["optimize", "-"], invalid.to_string(), expected);
    }
    let mut invalid = valid;
    invalid["contraction_order"] = json!({"isleaf":true, "tensorindex":99});
    failure(&["contract", "-"], invalid.to_string(), "out of range");
}

#[test]
fn saved_density_state_can_continue_through_a_noiseless_circuit() {
    let noisy = circuit(json!([
        {"type":"channel", "channel":"BitFlip", "p":0.25, "locs":[0]}
    ]));
    let state = success(&["simulate", "-"], noisy);
    let directory = std::env::temp_dir().join(format!(
        "yao-density-resume-{}-{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    std::fs::create_dir(&directory).unwrap();
    let state_path = directory.join("state.bin");
    std::fs::write(&state_path, state).unwrap();
    let gates = circuit(json!([{"type":"gate", "gate":"X", "targets":[1]}]));
    let resumed = success(
        &["simulate", "-", "--input", state_path.to_str().unwrap()],
        gates,
    );
    let result: Value = serde_json::from_slice(&success(&["probs", "-"], &resumed)).unwrap();
    for (actual, expected) in result["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .zip([0.0, 0.75, 0.0, 0.25])
    {
        assert!((actual.as_f64().unwrap() - expected).abs() < 1e-12);
    }
    failure(
        &["simulate", "-", "--input", state_path.to_str().unwrap()],
        r#"{"num_qubits":1,"elements":[]}"#,
        "Input state has 2 qubits but circuit has 1",
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn state_payload_validation_handles_zero_qubits_and_nonfinite_values() {
    let scalar = success(&["simulate", "-"], r#"{"num_qubits":0,"elements":[]}"#);
    let p: Value = serde_json::from_slice(&success(&["probs", "-"], scalar)).unwrap();
    assert_eq!(p["probabilities"], json!([1.0]));
    let mut input = format!("{}\n", json!({"format":"yao-state-v1","num_qubits":0,"dims":[],"num_elements":1,"dtype":"complex128"})).into_bytes();
    input.extend(f64::NAN.to_le_bytes());
    input.extend(0.0f64.to_le_bytes());
    failure(&["probs", "-"], input, "finite");
}
