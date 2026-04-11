//! Example: Generate an SVG of a quantum circuit.
//!
//! This example demonstrates:
//! - Building circuits with the `put` and `control` builder functions
//! - Adding label annotations for visual markers in circuit diagrams
//! - Exporting circuits to SVG markup
//!
//! Run with: cargo run --example circuit_svg

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

    println!("Generating SVG for Bell state circuit with annotations:");
    println!("{}", circuit);

    let svg = circuit.to_svg();

    let output_path = "bell_circuit.svg";
    std::fs::write(output_path, &svg)?;
    println!("SVG written to {} ({} bytes)", output_path, svg.len());

    Ok(())
}
