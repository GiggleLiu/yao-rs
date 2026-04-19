use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(prefix: &str, extension: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{unique}.{extension}"))
}

fn write_bell_circuit(path: &Path) {
    fs::write(
        path,
        r#"{
  "num_qubits": 2,
  "elements": [
    { "type": "gate", "gate": "H", "targets": [0] },
    { "type": "gate", "gate": "X", "targets": [1], "controls": [0] }
  ]
}"#,
    )
    .unwrap();
}

fn run_yao(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_yao"))
        .args(args)
        .output()
        .unwrap()
}

fn run_yao_with_stdin(args: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_yao"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    child.stdin.as_mut().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn run_cli_script_json(script: &str) -> Value {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let output = Command::new("bash")
        .arg(script)
        .current_dir(repo_root)
        .env("YAO_BIN", env!("CARGO_BIN_EXE_yao"))
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn simulate_measure_probs_and_expect_pipeline() {
    let circuit_path = temp_path("yao-bell", "json");
    write_bell_circuit(&circuit_path);

    let simulate = run_yao(&["simulate", circuit_path.to_str().unwrap()]);
    assert!(simulate.status.success(), "{simulate:?}");
    assert!(!simulate.stdout.is_empty());

    let probs = run_yao_with_stdin(&["probs", "-"], &simulate.stdout);
    assert!(probs.status.success(), "{probs:?}");
    let probs_json: Value = serde_json::from_slice(&probs.stdout).unwrap();
    let probabilities = probs_json["probabilities"].as_array().unwrap();
    assert_eq!(probabilities.len(), 4);
    assert!((probabilities[0].as_f64().unwrap() - 0.5).abs() < 1e-10);
    assert!(probabilities[1].as_f64().unwrap().abs() < 1e-10);
    assert!(probabilities[2].as_f64().unwrap().abs() < 1e-10);
    assert!((probabilities[3].as_f64().unwrap() - 0.5).abs() < 1e-10);

    let measure = run_yao_with_stdin(&["measure", "-", "--shots", "32"], &simulate.stdout);
    assert!(measure.status.success(), "{measure:?}");
    let measure_json: Value = serde_json::from_slice(&measure.stdout).unwrap();
    let outcomes = measure_json["outcomes"].as_array().unwrap();
    assert_eq!(outcomes.len(), 32);
    for outcome in outcomes {
        let outcome = outcome
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap())
            .collect::<Vec<_>>();
        assert!(outcome == vec![0, 0] || outcome == vec![1, 1]);
    }

    let expect = run_yao_with_stdin(&["expect", "-", "--op", "Z(0)Z(1)"], &simulate.stdout);
    assert!(expect.status.success(), "{expect:?}");
    let expect_json: Value = serde_json::from_slice(&expect.stdout).unwrap();
    assert!((expect_json["expectation_value"]["re"].as_f64().unwrap() - 1.0).abs() < 1e-10);
    assert!(
        expect_json["expectation_value"]["im"]
            .as_f64()
            .unwrap()
            .abs()
            < 1e-10
    );

    let _ = fs::remove_file(circuit_path);
}

#[test]
fn inspect_and_toeinsum_emit_expected_json() {
    let circuit_path = temp_path("yao-inspect", "json");
    write_bell_circuit(&circuit_path);

    let inspect = run_yao(&["--json", "inspect", circuit_path.to_str().unwrap()]);
    assert!(inspect.status.success(), "{inspect:?}");
    let inspect_json: Value = serde_json::from_slice(&inspect.stdout).unwrap();
    assert_eq!(inspect_json["num_qubits"].as_u64().unwrap(), 2);
    assert_eq!(inspect_json["gate_count"].as_u64().unwrap(), 2);
    assert_eq!(inspect_json["gates"].as_array().unwrap().len(), 2);

    let toeinsum = run_yao(&["toeinsum", circuit_path.to_str().unwrap()]);
    assert!(toeinsum.status.success(), "{toeinsum:?}");
    let toeinsum_json: Value = serde_json::from_slice(&toeinsum.stdout).unwrap();
    assert_eq!(toeinsum_json["format"].as_str().unwrap(), "yao-tn-v1");
    assert_eq!(toeinsum_json["mode"].as_str().unwrap(), "pure");
    assert_eq!(toeinsum_json["tensors"].as_array().unwrap().len(), 2);

    let _ = fs::remove_file(circuit_path);
}

#[test]
fn algorithm_examples_are_script_workflows_not_builtin_examples() {
    let output = run_yao(&["example", "bernstein-vazirani"]);
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Available examples: bell, ghz, qft"),
        "{stderr}"
    );
}

#[test]
fn visualize_writes_svg_in_default_build() {
    let circuit_path = temp_path("yao-visualize", "json");
    let svg_path = temp_path("yao-visualize", "svg");
    write_bell_circuit(&circuit_path);

    let visualize = run_yao(&[
        "--output",
        svg_path.to_str().unwrap(),
        "visualize",
        circuit_path.to_str().unwrap(),
    ]);

    assert!(visualize.status.success(), "{visualize:?}");
    let svg = fs::read_to_string(&svg_path).unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("</svg>"));

    let _ = fs::remove_file(circuit_path);
    let _ = fs::remove_file(svg_path);
}

#[test]
fn cli_script_phase_estimation_z_outputs_phase_bit() {
    let json = run_cli_script_json("examples/cli/phase_estimation_z.sh");
    let probabilities = json["probabilities"].as_array().unwrap();
    assert_eq!(probabilities.len(), 4);
    assert!((probabilities[3].as_f64().unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn cli_script_hadamard_test_z_runs() {
    let json = run_cli_script_json("examples/cli/hadamard_test_z.sh");
    assert_eq!(json["probabilities"].as_array().unwrap().len(), 4);
}

#[test]
fn cli_script_swap_test_runs() {
    let json = run_cli_script_json("examples/cli/swap_test.sh");
    assert_eq!(json["probabilities"].as_array().unwrap().len(), 8);
}

#[test]
fn cli_script_bernstein_vazirani_outputs_secret_probability() {
    let json = run_cli_script_json("examples/cli/bernstein_vazirani.sh");
    let probabilities = json["probabilities"].as_array().unwrap();
    assert!((probabilities[11].as_f64().unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn cli_script_grover_amplifies_marked_state() {
    let json = run_cli_script_json("examples/cli/grover_marked_state.sh");
    let probabilities = json["probabilities"].as_array().unwrap();
    assert!(
        probabilities[5].as_f64().unwrap() > 0.9,
        "marked probability = {}",
        probabilities[5]
    );
}

#[test]
fn cli_script_qaoa_maxcut_line4_runs() {
    let json = run_cli_script_json("examples/cli/qaoa_maxcut_line4.sh");
    assert!(json["expectation_value"]["re"].is_number());
}

#[test]
fn cli_script_qcbm_static_outputs_distribution() {
    let json = run_cli_script_json("examples/cli/qcbm_static.sh");
    assert_eq!(json["probabilities"].as_array().unwrap().len(), 64);
}

#[test]
fn visualize_requires_output_argument() {
    let circuit_path = temp_path("yao-visualize-missing-output", "json");
    write_bell_circuit(&circuit_path);

    let visualize = run_yao(&["visualize", circuit_path.to_str().unwrap()]);

    assert!(!visualize.status.success(), "{visualize:?}");
    let stderr = String::from_utf8_lossy(&visualize.stderr);
    assert!(
        stderr.contains("required") && stderr.contains("output"),
        "unexpected stderr: {stderr}"
    );

    let _ = fs::remove_file(circuit_path);
}

#[test]
fn visualize_rejects_non_svg_output_extension() {
    let circuit_path = temp_path("yao-visualize-bad-extension", "json");
    let pdf_path = temp_path("yao-visualize-bad-extension", "pdf");
    write_bell_circuit(&circuit_path);

    let visualize = run_yao(&[
        "--output",
        pdf_path.to_str().unwrap(),
        "visualize",
        circuit_path.to_str().unwrap(),
    ]);

    assert!(!visualize.status.success(), "{visualize:?}");
    let stderr = String::from_utf8_lossy(&visualize.stderr);
    assert!(
        stderr.contains("Only SVG output is supported"),
        "unexpected stderr: {stderr}"
    );

    let _ = fs::remove_file(circuit_path);
    let _ = fs::remove_file(pdf_path);
}
