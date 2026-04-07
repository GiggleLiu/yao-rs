//! Quantum Fourier Transform (QFT) circuit demo.
//!
//! This example demonstrates building a QFT circuit using the `put` and `control`
//! builder API, equivalent to Yao.jl's:
//!
//! ```julia
//! function qft_circuit(n)
//!     circuit = chain(n)
//!     for i in 1:n
//!         push!(circuit, put(n, i => H))
//!         for j in 1:n-i
//!             push!(circuit, control(n, i+j, i => shift(2π/2^(j+1))))
//!         end
//!     end
//!     for i in 1:div(n, 2)
//!         push!(circuit, swap(n, i, n-i+1))
//!     end
//!     circuit
//! end
//! ```

use std::f64::consts::PI;
use yao_rs::circuit::PositionedGate;
use yao_rs::{ArrayReg, Circuit, CircuitElement, Gate, apply, circuit_to_einsum, control, put};

/// Build an n-qubit QFT circuit (textbook version with bit-reversal SWAPs).
///
/// Note: `yao_rs::easybuild::qft_circuit` matches Yao.jl and omits the
/// final SWAP layer. This example includes it for the standard textbook QFT.
fn qft_circuit(n: usize) -> Circuit {
    let mut elements: Vec<CircuitElement> = Vec::new();

    for i in 0..n {
        // H gate on qubit i
        elements.push(put(vec![i], Gate::H));

        // Controlled phase rotations
        for j in 1..(n - i) {
            let theta = 2.0 * PI / (1 << (j + 1)) as f64;
            elements.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
    }

    // Reverse qubit order with SWAPs
    for i in 0..(n / 2) {
        elements.push(CircuitElement::Gate(PositionedGate::new(
            Gate::SWAP,
            vec![i, n - 1 - i],
            vec![],
            vec![],
        )));
    }

    Circuit::new(vec![2; n], elements).unwrap()
}

fn main() {
    let n = 4;
    println!("=== {}-qubit Quantum Fourier Transform ===\n", n);

    // Build the QFT circuit
    let circuit = qft_circuit(n);
    println!(
        "Circuit: {} elements on {} qubits",
        circuit.elements.len(),
        n
    );

    // Apply QFT to |0000⟩ — should give uniform superposition
    let reg = ArrayReg::zero_state(n);
    let result = apply(&circuit, &reg);
    println!("\nQFT|0000⟩ (should be uniform superposition):");
    let total_dim = result.state_vec().len();
    let expected_amp = 1.0 / (total_dim as f64).sqrt();
    println!("  Expected amplitude: {:.6}", expected_amp);
    println!("  First few amplitudes:");
    for i in 0..total_dim.min(8) {
        let amp = result.state_vec()[i];
        println!(
            "    |{:0width$b}⟩: {:.6} + {:.6}i",
            i,
            amp.re,
            amp.im,
            width = n
        );
    }

    // Apply QFT to |0001⟩ — should have phase progression
    let reg1 = ArrayReg::product_state(bitbasis::BitStr::<4>::new(0b0001));
    let result1 = apply(&circuit, &reg1);
    println!("\nQFT|0001⟩ (should have phase progression):");
    for i in 0..total_dim.min(8) {
        let amp = result1.state_vec()[i];
        println!(
            "    |{:0width$b}⟩: {:.6} + {:.6}i  (|amp|={:.6})",
            i,
            amp.re,
            amp.im,
            amp.norm(),
            width = n
        );
    }

    // Verify norm preservation
    println!("\nNorm of QFT|0000⟩: {:.10}", result.norm());
    println!("Norm of QFT|0001⟩: {:.10}", result1.norm());

    // Show tensor network structure
    let tn = circuit_to_einsum(&circuit);
    println!("\n=== Tensor Network ===");
    println!("Number of tensors: {}", tn.tensors.len());
    println!("Number of unique labels: {}", tn.size_dict.len());
    println!("EinCode: {:?}", tn.code);
}
