//! Integration tests for the toeinsum | optimize | contract pipeline.

use assert_cmd::Command;

fn yao() -> Command {
    Command::cargo_bin("yao").unwrap()
}

/// Run the 3-stage pipeline: example bell -> toeinsum -> optimize -> contract
fn run_pipeline(mode_args: &[&str]) -> String {
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
