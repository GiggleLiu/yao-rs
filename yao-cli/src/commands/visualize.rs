use crate::output::OutputConfig;
use anyhow::{Context, Result, bail};
use std::path::Path;

pub fn visualize(circuit_path: &str, output_path: &Path, out: &OutputConfig) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

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
