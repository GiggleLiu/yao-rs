#![cfg(any(feature = "omeinsum", feature = "tenferro"))]

use assert_cmd::Command;
use serde_json::{Value, json};

fn run(args: &[&str], input: &Value) -> std::process::Output {
    Command::cargo_bin("yao")
        .unwrap()
        .args(args)
        .write_stdin(input.to_string())
        .output()
        .unwrap()
}
fn ok(args: &[&str], input: &Value) -> Value {
    let out = run(args, input);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}
fn bad(args: &[&str], input: &Value, expected: &str) {
    let out = run(args, input);
    let message = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(message.contains(expected), "{message}");
    assert!(!message.contains("panicked"), "{message}");
}
fn matrix_network() -> Value {
    json!({"format":"yao-tn-v1","mode":"pure",
        "eincode":{"input_indices":[["0","1"],["1","2"]],"output_indices":["0","2"]},
        "tensors":[{"shape":[2,3],"data_re":[1.,2.,3.,4.,5.,6.],"data_im":[0.,0.,0.,0.,0.,0.]},
                   {"shape":[3,2],"data_re":[1.,2.,3.,4.,5.,6.],"data_im":[0.,0.,0.,0.,0.,0.]}],
        "size_dict":{"0":2,"1":3,"2":2}})
}
fn backends() -> Vec<&'static str> {
    vec![
        #[cfg(feature = "omeinsum")]
        "omeinsum",
        #[cfg(feature = "tenferro")]
        "tenferro",
    ]
}

#[test]
fn fixed_and_automatic_plans_roundtrip_preserve_execution_and_limits() {
    for args in [
        vec!["optimize", "-", "--slice=1"],
        vec!["optimize", "-", "--slice=0,2"],
        vec!["optimize", "-", "--memory-budget", "512"],
    ] {
        let plan = ok(&args, &matrix_network());
        assert_eq!(plan["format"], "yao-tn-v2");
        assert_eq!(plan["slice_plan"]["estimate"]["concurrent_slices"], 1);
        if args.contains(&"--memory-budget") {
            assert!(
                plan["slice_plan"]["estimate"]["estimated_total_bytes"]
                    .as_u64()
                    .unwrap()
                    <= 512
            );
        }
        for serialized in [plan.clone(), ok(&["optimize", "-"], &plan)] {
            assert_eq!(
                serialized["slice_plan"]["labels"],
                plan["slice_plan"]["labels"]
            );
            for backend in backends() {
                let output = ok(&["contract", "-", "--backend", backend], &serialized);
                let values: Vec<_> = output
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|entry| entry["re"].as_f64().unwrap())
                    .collect();
                for (a, b) in values.iter().zip([22., 28., 49., 64.]) {
                    assert!((a - b).abs() < 1e-10);
                }
            }
        }
    }
    let old = ok(&["optimize", "-"], &matrix_network());
    assert_eq!(old["format"], "yao-tn-v1");
    assert!(old.get("slice_plan").is_none());
}

#[test]
fn invalid_versions_slices_estimates_and_impossible_budgets_fail_before_execution() {
    let plan = ok(&["optimize", "-", "--slice=1"], &matrix_network());
    let mut invalid = plan.clone();
    invalid["format"] = json!("yao-tn-v1");
    bad(&["contract", "-"], &invalid, "must not contain slice_plan");
    let mut invalid = plan.clone();
    invalid.as_object_mut().unwrap().remove("slice_plan");
    bad(&["contract", "-"], &invalid, "requires slice_plan");
    let mut invalid = plan.clone();
    invalid["slice_plan"]["estimate"]["estimated_total_bytes"] = json!(1);
    bad(&["contract", "-"], &invalid, "estimate disagrees");
    let mut invalid = plan.clone();
    invalid["slice_plan"]["labels"] = json!([1, 1]);
    bad(&["contract", "-"], &invalid, "duplicate slice label");
    let mut invalid = plan;
    invalid["format"] = json!("yao-tn-v99");
    bad(
        &["contract", "-"],
        &invalid,
        "Unknown tensor-network format",
    );
    for (args, message) in [
        (vec!["optimize", "-", "--slice=42"], "Unknown or duplicate"),
        (
            vec!["optimize", "-", "--memory-budget", "256"],
            "cannot hold",
        ),
        (
            vec!["optimize", "-", "--slice=1", "--max-slices", "2"],
            "exceeding limit",
        ),
        (vec!["optimize", "-", "--memory-budget", "0"], "positive"),
    ] {
        bad(&args, &matrix_network(), message);
    }
}

#[test]
fn multi_term_noisy_observable_pipeline_matches_direct_simulation() {
    let mut circuit = json!({"num_qubits":2,"elements":[
        {"type":"gate","gate":"Ry","params":[0.4],"targets":[0]},
        {"type":"gate","gate":"Rx","params":[0.7],"targets":[1]}
    ]});
    for noisy in [false, true] {
        if noisy {
            circuit["elements"]
                .as_array_mut()
                .unwrap()
                .push(json!({"type":"channel","channel":"BitFlip","p":0.2,"locs":[0]}));
        }
        let op = "0.3 * Z(0) + 0.7 * Y(1)";
        let expected = ok(&["run", "-", "--op", op], &circuit)["expectation_value"].clone();
        let tn = ok(&["toeinsum", "-", "--op", op], &circuit);
        assert_eq!(tn["mode"], if noisy { "dm" } else { "pure" });
        let first_label = tn["eincode"]["input_indices"][0][0].as_str().unwrap();
        let arg = format!("--slice={first_label}");
        let plan = ok(&["optimize", "-", &arg], &tn);
        for backend in backends() {
            let result = ok(&["contract", "-", "--backend", backend], &plan);
            for part in ["re", "im"] {
                assert!(
                    (result[part].as_f64().unwrap() - expected[part].as_f64().unwrap()).abs()
                        < 1e-10
                );
            }
        }
    }
}
