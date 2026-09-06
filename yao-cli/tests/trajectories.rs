use assert_cmd::Command;
use serde_json::{Value, json};
fn invoke(args: &[&str], input: &str) -> std::process::Output {
    Command::cargo_bin("yao")
        .unwrap()
        .args(args)
        .write_stdin(input)
        .output()
        .unwrap()
}
fn good(args: &[&str], input: &str) -> Value {
    let out = invoke(args, input);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}
fn noisy() -> String {
    json!({"num_qubits":2,"elements":[{"type":"gate","gate":"X","targets":[0]},
        {"type":"channel","channel":"BitFlip","p":0.25,"locs":[0]}]})
    .to_string()
}
#[test]
fn trajectory_statistics_are_seeded_and_separate_from_exact_results() {
    let args = [
        "run",
        "-",
        "--trajectories",
        "4096",
        "--op",
        "Z(0)",
        "--seed",
        "19",
    ];
    let result = good(&args, &noisy());
    assert_eq!(result, good(&args, &noisy()));
    assert_eq!(result["mode"], "trajectories");
    let stats = &result["statistics"];
    assert_eq!(stats["trajectories"], 4096);
    assert_eq!(stats["seed"], 19);
    let mean = stats["mean"][0].as_f64().unwrap();
    let se = stats["standard_error"][0].as_f64().unwrap();
    assert!((mean + 0.5).abs() < 6. * se);
    assert!(se > 0.);
    assert_eq!(stats["standard_error"][1], 0.);
    assert!(result.get("shots").is_none());
    let one = good(
        &[
            "run",
            "-",
            "--trajectories",
            "1",
            "--op",
            "Z(0)",
            "--seed",
            "19",
        ],
        &noisy(),
    );
    assert!(one["statistics"]["standard_error"].is_null());
    #[cfg(feature = "parallel")]
    {
        let mut args = args.to_vec();
        args.extend(["--threads", "4"]);
        let parallel = good(&args, &noisy());
        assert_eq!(parallel["statistics"]["mean"], stats["mean"]);
        assert_eq!(
            parallel["statistics"]["standard_error"],
            stats["standard_error"]
        );
    }
}
#[test]
fn conflicting_and_invalid_trajectory_options_fail_without_panics() {
    for args in [
        vec!["run", "-", "--trajectories", "10"],
        vec![
            "run",
            "-",
            "--trajectories",
            "10",
            "--shots",
            "10",
            "--op",
            "Z(0)",
        ],
        vec!["run", "-", "--trajectories", "0", "--op", "Z(0)"],
        vec![
            "run",
            "-",
            "--trajectories",
            "10",
            "--op",
            "Z(0)",
            "--threads",
            "0",
        ],
        vec!["run", "-", "--op", "Z(0)", "--threads", "2"],
        vec!["run", "-", "--op", "Z(0)", "--seed", "1"],
        vec!["run", "-", "--trajectories", "10", "--op", "Z(2)"],
    ] {
        let out = invoke(&args, &noisy());
        assert!(!out.status.success(), "{args:?}");
        assert!(!String::from_utf8_lossy(&out.stderr).contains("panicked"));
    }
    #[cfg(not(feature = "parallel"))]
    assert!(
        !invoke(
            &[
                "run",
                "-",
                "--trajectories",
                "10",
                "--op",
                "Z(0)",
                "--threads",
                "2"
            ],
            &noisy()
        )
        .status
        .success()
    );
}

#[test]
fn trajectory_mode_reads_pure_input_and_rejects_density_input() {
    let file =
        std::env::temp_dir().join(format!("yao-trajectory-input-{}.json", std::process::id()));
    std::fs::write(&file, json!({"num_qubits":1,"elements":[]}).to_string()).unwrap();
    let pure = invoke(
        &["simulate", "-"],
        &json!({"num_qubits":1,"elements":[{"type":"gate","gate":"X","targets":[0]}]}).to_string(),
    );
    assert!(pure.status.success());
    let args = [
        "run",
        file.to_str().unwrap(),
        "--input",
        "-",
        "--trajectories",
        "16",
        "--op",
        "Z(0)",
        "--seed",
        "4",
    ];
    let result = Command::cargo_bin("yao")
        .unwrap()
        .args(args)
        .write_stdin(pure.stdout)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let result: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(result["statistics"]["mean"][0], -1.);
    let density=invoke(&["simulate","-"],&json!({"num_qubits":1,"elements":[{"type":"channel","channel":"BitFlip","p":0.3,"locs":[0]}]}).to_string());
    assert!(density.status.success());
    let result = Command::cargo_bin("yao")
        .unwrap()
        .args(args)
        .write_stdin(density.stdout)
        .output()
        .unwrap();
    std::fs::remove_file(file).unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("normalized pure state"));
}
