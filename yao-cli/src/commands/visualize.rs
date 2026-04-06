use crate::output::OutputConfig;
use anyhow::{Result, anyhow, bail};

pub fn visualize(circuit_path: &str, out: &OutputConfig) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    let output_path = out
        .output
        .as_ref()
        .ok_or_else(|| anyhow!("--output is required for visualize (e.g. --output circuit.pdf)"))?;

    let extension = output_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    if extension != "pdf" {
        bail!(
            "Only PDF output is supported in v1. Got extension: '.{}'",
            extension
        );
    }

    let pdf_bytes =
        yao_rs::to_pdf(&circuit).map_err(|err| anyhow!("PDF generation failed: {err:?}"))?;
    std::fs::write(output_path, &pdf_bytes)?;
    out.info(&format!("PDF written to {}", output_path.display()));

    Ok(())
}
