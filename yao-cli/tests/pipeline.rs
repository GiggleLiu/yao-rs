//! Integration tests for the toeinsum | optimize | contract pipeline.

#![cfg(any(feature = "omeinsum", feature = "tenferro"))]

use assert_cmd::Command;

fn yao() -> Command {
    Command::cargo_bin("yao").unwrap()
}

/// Run the 3-stage pipeline: example bell -> toeinsum -> optimize -> contract
fn run_pipeline(mode_args: &[&str]) -> String {
    run_pipeline_with_backend(mode_args, &[])
}

fn run_pipeline_with_backend(mode_args: &[&str], backend_args: &[&str]) -> String {
    // Step 1: generate example circuit
    let example = yao()
        .args(["example", "bell", "--json"])
        .output()
        .expect("failed to run yao example");
    assert!(example.status.success());

    // Step 2: toeinsum (pipe circuit via stdin)
    let mut toeinsum = yao();
    toeinsum.args(["toeinsum", "-", "--json"]);
    toeinsum.args(mode_args);
    let toeinsum_out = toeinsum
        .write_stdin(example.stdout)
        .output()
        .expect("failed to run toeinsum");
    assert!(
        toeinsum_out.status.success(),
        "toeinsum failed: {}",
        String::from_utf8_lossy(&toeinsum_out.stderr)
    );

    // Step 3: optimize
    let optimize_out = yao()
        .args(["optimize", "-", "--json"])
        .write_stdin(toeinsum_out.stdout)
        .output()
        .expect("failed to run optimize");
    assert!(
        optimize_out.status.success(),
        "optimize failed: {}",
        String::from_utf8_lossy(&optimize_out.stderr)
    );

    // Step 4: contract
    let contract_out = yao()
        .args(["contract", "-", "--json"])
        .args(backend_args)
        .write_stdin(optimize_out.stdout)
        .output()
        .expect("failed to run contract");
    assert!(
        contract_out.status.success(),
        "contract failed: {}",
        String::from_utf8_lossy(&contract_out.stderr)
    );

    String::from_utf8(contract_out.stdout).unwrap()
}

#[test]
fn test_pipeline_overlap() {
    let output = run_pipeline(&["--mode", "overlap"]);
    let val: serde_json::Value = serde_json::from_str(&output).unwrap();
    let re = val["re"].as_f64().unwrap();
    // Bell circuit overlap <0|U|0> = 1/sqrt(2)
    assert!(
        (re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10,
        "Expected 1/sqrt(2), got {re}"
    );
}

#[test]
fn test_pipeline_state() {
    let output = run_pipeline(&["--mode", "state"]);
    let data: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
    // Bell state: |00> and |11> with equal amplitudes
    assert_eq!(data.len(), 2);
    let bitstrings: Vec<&str> = data
        .iter()
        .map(|e| e["bitstring"].as_str().unwrap())
        .collect();
    assert!(bitstrings.contains(&"00"));
    assert!(bitstrings.contains(&"11"));
}

#[test]
fn test_contract_rejects_unoptimized_tn() {
    let example = yao().args(["example", "bell", "--json"]).output().unwrap();
    assert!(example.status.success());

    let tn_out = yao()
        .args(["toeinsum", "-", "--json"])
        .write_stdin(example.stdout)
        .output()
        .unwrap();
    assert!(tn_out.status.success());

    // Contract without optimize should fail
    let contract_out = yao()
        .args(["contract", "-", "--json"])
        .write_stdin(tn_out.stdout)
        .output()
        .unwrap();

    assert!(
        !contract_out.status.success(),
        "contract should reject unoptimized TN"
    );
    let stderr = String::from_utf8_lossy(&contract_out.stderr);
    assert!(
        stderr.contains("contraction order"),
        "Error should mention contraction order, got: {stderr}"
    );
}

#[test]
fn test_pipeline_density_matrix_mode() {
    let output = run_pipeline(&["--mode", "dm"]);
    let data: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();

    assert_eq!(data.len(), 4);

    let bitstrings: Vec<&str> = data
        .iter()
        .map(|e| e["bitstring"].as_str().unwrap())
        .collect();
    assert!(bitstrings.contains(&"0000"));
    assert!(bitstrings.contains(&"0011"));
    assert!(bitstrings.contains(&"1100"));
    assert!(bitstrings.contains(&"1111"));

    for entry in data {
        assert!((entry["re"].as_f64().unwrap() - 0.5).abs() < 1e-10);
        assert_eq!(entry["im"].as_f64().unwrap(), 0.0);
    }
}

#[test]
fn test_contract_formats_mixed_radix_state_indices() {
    let tn_json = serde_json::json!({
        "format": "yao-tn-v1",
        "mode": "pure",
        "eincode": {
            "input_indices": [["0", "1"]],
            "output_indices": ["0", "1"],
        },
        "tensors": [{
            "shape": [2, 3],
            "data_re": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "data_im": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }],
        "size_dict": {
            "0": 2,
            "1": 3,
        },
        "contraction_order": {
            "isleaf": true,
            "tensorindex": 0,
        },
    });

    let contract_out = yao()
        .args(["contract", "-", "--json"])
        .write_stdin(tn_json.to_string())
        .output()
        .expect("failed to run contract");

    assert!(
        contract_out.status.success(),
        "contract failed: {}",
        String::from_utf8_lossy(&contract_out.stderr)
    );

    let data: Vec<serde_json::Value> = serde_json::from_slice(&contract_out.stdout).unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["index"].as_u64(), Some(2));
    assert_eq!(data[0]["bitstring"].as_str(), Some("02"));
}

#[cfg(feature = "tenferro")]
#[test]
fn tenferro_pipeline_matches_default_for_every_mode() {
    for args in [
        vec!["--mode", "state"],
        vec!["--mode", "overlap"],
        vec!["--mode", "dm"],
        vec!["--op", "Z(0)Z(1)"],
    ] {
        let reference: serde_json::Value = serde_json::from_str(&run_pipeline(&args)).unwrap();
        for threads in ["1", "2"] {
            let actual: serde_json::Value = serde_json::from_str(&run_pipeline_with_backend(
                &args,
                &["--backend", "tenferro", "--threads", threads],
            ))
            .unwrap();
            if reference.is_array() {
                let a = actual.as_array().unwrap();
                let b = reference.as_array().unwrap();
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert_eq!(a["index"], b["index"]);
                    assert_eq!(a["bitstring"], b["bitstring"]);
                    for key in ["re", "im", "prob"] {
                        assert!(
                            (a[key].as_f64().unwrap() - b[key].as_f64().unwrap()).abs() < 1e-12
                        );
                    }
                }
            } else {
                for key in ["re", "im"] {
                    assert!(
                        (actual[key].as_f64().unwrap() - reference[key].as_f64().unwrap()).abs()
                            < 1e-12
                    );
                }
            }
        }
    }
}

#[test]
fn rejects_unavailable_backend_and_zero_threads() {
    let output = yao()
        .args(["contract", "-", "--backend", "not-a-backend"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
    let output = yao()
        .args(["contract", "-", "--threads", "0"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[cfg(feature = "tenferro")]
#[test]
fn tenferro_unary_and_empty_network_cli() {
    for (inputs, outputs, tensors, sizes, expected) in [
        (
            serde_json::json!([["0", "1"]]),
            serde_json::json!(["1", "0"]),
            serde_json::json!([{"shape":[2,3],"data_re":[1,2,3,4,5,6],"data_im":[1,0,0,0,0,-1]}]),
            serde_json::json!({"0":2,"1":3}),
            vec![1., 4., 2., 5., 3., 6.],
        ),
        (
            serde_json::json!([["0", "0"]]),
            serde_json::json!([]),
            serde_json::json!([{"shape":[2,2],"data_re":[1,7,8,2],"data_im":[1,0,0,-1]}]),
            serde_json::json!({"0":2}),
            vec![3.],
        ),
        (
            serde_json::json!([]),
            serde_json::json!([]),
            serde_json::json!([]),
            serde_json::json!({}),
            vec![1.],
        ),
    ] {
        let dto = serde_json::json!({"format":"yao-tn-v1", "mode":"pure", "eincode":{
            "input_indices": inputs, "output_indices": outputs}, "size_dict":sizes, "tensors":tensors});
        let optimized = yao()
            .args(["optimize", "-"])
            .write_stdin(dto.to_string())
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let result = yao()
            .args(["contract", "-", "--backend", "tenferro"])
            .write_stdin(optimized.clone())
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let result: serde_json::Value = serde_json::from_slice(&result).unwrap();
        let values = if result.is_array() {
            result
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x["re"].as_f64().unwrap())
                .collect()
        } else {
            vec![result["re"].as_f64().unwrap()]
        };
        assert_eq!(values, expected);
        #[cfg(feature = "omeinsum")]
        {
            let rejected = yao()
                .args(["contract", "-", "--backend", "omeinsum", "--threads", "2"])
                .write_stdin(optimized)
                .assert()
                .failure()
                .get_output()
                .stderr
                .clone();
            assert!(
                String::from_utf8_lossy(&rejected)
                    .contains("--threads requires --backend tenferro")
            );
        }
    }
}
