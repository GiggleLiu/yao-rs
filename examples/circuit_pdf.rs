//! Example: Generate a PDF of a quantum circuit.
//!
//! Run with: cargo run --example circuit_pdf --features typst

use yao_rs::{Circuit, Gate, put, control};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a Bell state circuit: H on qubit 0, then CNOT
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            put(vec![0], Gate::H),
            control(vec![0], vec![1], Gate::X),
        ],
    )?;

    println!("Generating PDF for circuit:");
    println!("{}", circuit);

    let pdf_bytes = circuit.to_pdf()?;

    let output_path = "bell_circuit.pdf";
    std::fs::write(output_path, &pdf_bytes)?;
    println!("PDF written to {} ({} bytes)", output_path, pdf_bytes.len());

    Ok(())
}
