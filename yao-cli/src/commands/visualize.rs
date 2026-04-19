use crate::output::OutputConfig;
use anyhow::{Context, Result, anyhow, bail};

pub fn visualize(circuit_path: &str, out: &OutputConfig) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    let output_path = out
        .output
        .as_ref()
        .ok_or_else(|| anyhow!("--output is required for visualize (e.g. --output circuit.svg)"))?;

    let extension = output_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    if !extension.eq_ignore_ascii_case("svg") {
        bail!(
            "Only SVG output is supported in v1. Got extension: '.{}'",
            extension
        );
    }

    let svg = circuit.to_svg();
    std::fs::write(output_path, svg)
        .with_context(|| format!("Failed to write SVG to {}", output_path.display()))?;
    out.info(&format!("SVG written to {}", output_path.display()));

    Ok(())
}
