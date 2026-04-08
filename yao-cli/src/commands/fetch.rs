use crate::output::OutputConfig;
use anyhow::{Context, Result, bail};
use std::io::Write;
use std::process::Command;

/// Known QASMBench circuits organized by scale.
const QASMBENCH_SMALL: &[(&str, &str)] = &[
    ("adder_n4", "4-qubit adder"),
    ("adder_n10", "10-qubit adder"),
    ("basis_change_n3", "3-qubit basis change"),
    ("bb84_n8", "8-qubit BB84 protocol"),
    ("bell_n4", "4-qubit Bell state"),
    ("cat_state_n4", "4-qubit cat state"),
    ("deutsch_n2", "2-qubit Deutsch algorithm"),
    ("dnn_n2", "2-qubit deep neural network"),
    ("error_correctiond3_n5", "5-qubit error correction"),
    ("fredkin_n3", "3-qubit Fredkin gate"),
    ("grover_n2", "2-qubit Grover search"),
    ("hs4_n4", "4-qubit hidden subgroup"),
    ("iswap_n2", "2-qubit iSWAP"),
    ("linearsolver_n3", "3-qubit linear solver"),
    ("lpn_n5", "5-qubit LPN"),
    ("pea_n5", "5-qubit phase estimation"),
    ("qaoa_n3", "3-qubit QAOA"),
    ("qec_en_n5", "5-qubit QEC encoder"),
    ("qec_sm_n5", "5-qubit QEC syndrome"),
    ("qft_n4", "4-qubit QFT"),
    ("qrng_n4", "4-qubit QRNG"),
    ("quantumwalks_n2", "2-qubit quantum walk"),
    ("simon_n6", "6-qubit Simon's algorithm"),
    ("teleportation_n3", "3-qubit teleportation"),
    ("toffoli_n3", "3-qubit Toffoli"),
    ("variational_n4", "4-qubit variational"),
    ("vqe_n4", "4-qubit VQE"),
    ("vqe_uccsd_n4", "4-qubit VQE-UCCSD"),
    ("wstate_n3", "3-qubit W state"),
];

const QASMBENCH_BASE_URL: &str =
    "https://raw.githubusercontent.com/pnnl/QASMBench/master";

pub fn fetch(source: &str, name: &str, out: &OutputConfig) -> Result<()> {
    match source {
        "qasmbench" => fetch_qasmbench(name, out),
        _ => bail!(
            "Unknown source: '{source}'\n\n\
             Available sources:\n  \
             qasmbench  — QASMBench benchmark suite (QASM 2.0)\n\n\
             Examples:\n  \
             yao fetch qasmbench grover\n  \
             yao fetch qasmbench list"
        ),
    }
}

fn fetch_qasmbench(name: &str, out: &OutputConfig) -> Result<()> {
    if name == "list" {
        return list_qasmbench(out);
    }

    // Support explicit scale/name paths: "small/grover_n2", "medium/shor_n5"
    let (scale, circuit) = if name.contains('/') {
        let parts: Vec<&str> = name.splitn(2, '/').collect();
        (parts[0].to_string(), parts[1].to_string())
    } else {
        find_qasmbench_circuit(name)?
    };

    let url = format!("{QASMBENCH_BASE_URL}/{scale}/{circuit}/{circuit}.qasm");
    out.info(&format!("Fetching {circuit} from QASMBench ({scale})..."));

    let qasm = download_url(&url)
        .with_context(|| format!("Failed to download from QASMBench: {circuit}"))?;

    if let Some(ref path) = out.output {
        std::fs::write(path, &qasm)
            .with_context(|| format!("Failed to write {}", path.display()))?;
        out.info(&format!("Saved to {}", path.display()));
    } else {
        print!("{qasm}");
        std::io::stdout().flush()?;
    }
    Ok(())
}

fn find_qasmbench_circuit(name: &str) -> Result<(String, String)> {
    // Try exact match
    for &(dir, _) in QASMBENCH_SMALL {
        if dir == name {
            return Ok(("small".to_string(), dir.to_string()));
        }
    }

    // Try prefix match (e.g. "grover" matches "grover_n2")
    let mut matches: Vec<&str> = Vec::new();
    for &(dir, _) in QASMBENCH_SMALL {
        if dir.starts_with(name) || dir.starts_with(&format!("{name}_")) {
            matches.push(dir);
        }
    }

    match matches.len() {
        0 => bail!(
            "Unknown QASMBench circuit: '{name}'\n\n\
             Use 'yao fetch qasmbench list' to see available circuits.\n\
             Or specify scale/name: yao fetch qasmbench medium/shor_n5"
        ),
        1 => Ok(("small".to_string(), matches[0].to_string())),
        _ => bail!(
            "Ambiguous name '{name}'. Matches:\n  {}\n\nPlease be more specific.",
            matches.join("\n  ")
        ),
    }
}

fn list_qasmbench(out: &OutputConfig) -> Result<()> {
    let json_value = serde_json::json!({
        "source": "qasmbench",
        "circuits": QASMBENCH_SMALL.iter().map(|(name, desc)| {
            serde_json::json!({"name": name, "description": desc})
        }).collect::<Vec<_>>(),
    });

    let mut human = String::from("QASMBench circuits (small, 2-10 qubits):\n\n");
    for (name, desc) in QASMBENCH_SMALL {
        human.push_str(&format!("  {name:<30} {desc}\n"));
    }
    human.push_str("\nUsage:\n");
    human.push_str("  yao fetch qasmbench grover              # prefix match → grover_n2\n");
    human.push_str("  yao fetch qasmbench qft_n4 -o qft.qasm  # save to file\n");
    human.push_str("  yao fetch qasmbench medium/shor_n5       # medium/large by path\n");
    human.push_str("  yao fetch qasmbench grover | yao fromqasm - | yao run - --shots 100\n");

    out.emit(&human, &json_value)
}

fn download_url(url: &str) -> Result<String> {
    let output = Command::new("curl")
        .args(["-fsSL", url])
        .output()
        .context("Failed to run curl. Is curl installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Download failed: {stderr}");
    }

    String::from_utf8(output.stdout).context("Downloaded file is not valid UTF-8")
}
