//! Example: Generate a PDF of a quantum circuit.
//!
//! This example demonstrates:
//! - Building circuits with the `put` and `control` builder functions
//! - Adding label annotations for visual markers in circuit diagrams
//! - Exporting circuits to PDF using the typst feature
//!
//! Run with: cargo run --example circuit_pdf --features typst

use yao_rs::{Circuit, Gate, control, label, put};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a Bell state circuit with annotations:
    // - Label on qubit 0 showing input state
    // - H gate on qubit 0
    // - CNOT (controlled-X) from qubit 0 to qubit 1
    // - Label showing the resulting Bell state
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            label(0, "|0⟩"),                    // Input state annotation on qubit 0
            label(1, "|0⟩"),                    // Input state annotation on qubit 1
            put(vec![0], Gate::H),              // Hadamard on qubit 0
            control(vec![0], vec![1], Gate::X), // CNOT: control=0, target=1
            label(0, "|Φ+⟩"),                   // Bell state annotation
        ],
    )?;

    println!("Generating PDF for Bell state circuit with annotations:");
    println!("{}", circuit);

    let pdf_bytes = circuit.to_pdf()?;

    let output_path = "bell_circuit.pdf";
    std::fs::write(output_path, &pdf_bytes)?;
    println!("PDF written to {} ({} bytes)", output_path, pdf_bytes.len());

    Ok(())
}
