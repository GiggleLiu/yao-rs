use crate::output::OutputConfig;
use anyhow::{Context, Result, bail};
use std::io::Write;
use std::process::Command;

const QASMBENCH_RAW_URL: &str = "https://raw.githubusercontent.com/pnnl/QASMBench/master";
const QASMBENCH_API_URL: &str = "https://api.github.com/repos/pnnl/QASMBench/contents";
const SCALES: &[&str] = &["small", "medium", "large"];

pub fn fetch(source: &str, name: &str, scale: Option<&str>, out: &OutputConfig) -> Result<()> {
    match source {
        "qasmbench" => fetch_qasmbench(name, scale, out),
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

fn fetch_qasmbench(name: &str, scale: Option<&str>, out: &OutputConfig) -> Result<()> {
    if name == "list" {
        return list_qasmbench(scale, out);
    }

    // Support explicit scale/name paths: "small/grover_n2", "medium/shor_n5"
    let (scale, circuit) = if name.contains('/') {
        let parts: Vec<&str> = name.splitn(2, '/').collect();
        (parts[0].to_string(), parts[1].to_string())
    } else {
        // Search all scales for a match
        find_qasmbench_circuit(name)?
    };

    let url = format!("{QASMBENCH_RAW_URL}/{scale}/{circuit}/{circuit}.qasm");
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

/// Fetch directory listing from GitHub API for a given scale.
fn list_qasmbench_dirs(scale: &str) -> Result<Vec<String>> {
    let url = format!("{QASMBENCH_API_URL}/{scale}");
    let json_str = download_url(&url)?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&json_str).context("Failed to parse GitHub API response")?;
    let mut dirs: Vec<String> = entries
        .iter()
        .filter(|e| e["type"].as_str() == Some("dir"))
        .filter_map(|e| e["name"].as_str().map(String::from))
        .collect();
    dirs.sort();
    Ok(dirs)
}

fn find_qasmbench_circuit(name: &str) -> Result<(String, String)> {
    for &scale in SCALES {
        let dirs = list_qasmbench_dirs(scale)?;

        // Exact match
        if dirs.iter().any(|d| d == name) {
            return Ok((scale.to_string(), name.to_string()));
        }

        // Prefix match
        let matches: Vec<&String> = dirs
            .iter()
            .filter(|d| d.starts_with(name) || d.starts_with(&format!("{name}_")))
            .collect();

        match matches.len() {
            1 => return Ok((scale.to_string(), matches[0].clone())),
            2.. => {
                let names: Vec<&str> = matches.iter().map(|s| s.as_str()).collect();
                bail!(
                    "Ambiguous name '{name}' in {scale}. Matches:\n  {}\n\nPlease be more specific.",
                    names.join("\n  ")
                );
            }
            _ => continue,
        }
    }

    bail!(
        "Unknown QASMBench circuit: '{name}'\n\n\
         Use 'yao fetch qasmbench list' to see all available circuits."
    )
}

fn list_qasmbench(filter_scale: Option<&str>, out: &OutputConfig) -> Result<()> {
    let scales: Vec<&str> = match filter_scale {
        Some(s) => {
            if !SCALES.contains(&s) {
                bail!("Unknown scale '{s}'. Use: small, medium, or large");
            }
            vec![s]
        }
        None => SCALES.to_vec(),
    };

    let mut all_circuits: Vec<serde_json::Value> = Vec::new();
    let mut human = String::new();

    for &scale in &scales {
        let dirs = list_qasmbench_dirs(scale)?;

        human.push_str(&format!("QASMBench — {scale} ({} circuits):\n", dirs.len()));
        for dir in &dirs {
            human.push_str(&format!("  {dir}\n"));
            all_circuits.push(serde_json::json!({"name": dir, "scale": scale}));
        }
        human.push('\n');
    }

    human.push_str("Usage:\n");
    human.push_str("  yao fetch qasmbench <name>              # auto-detect scale\n");
    human.push_str("  yao fetch qasmbench small/qft_n4        # explicit scale/name\n");
    human.push_str("  yao fetch qasmbench grover -o g.qasm    # save to file\n");
    human.push_str("  yao fetch qasmbench list small           # list only small circuits\n");

    let json_value = serde_json::json!({
        "source": "qasmbench",
        "circuits": all_circuits,
    });

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
