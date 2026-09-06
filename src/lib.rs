//! Quantum circuits, qubit simulation, automatic differentiation, and tensor networks.
//!
//! Qubit 0 is the most significant bit of a computational-basis index.
//!
//! ```
//! use yao_rs::{ArrayReg, Circuit, Gate, apply, control, probs, put};
//! let circuit = Circuit::qubits(2, vec![
//!     put(vec![0], Gate::H),
//!     control(vec![0], vec![1], Gate::X),
//! ]).unwrap();
//! let p = probs(&apply(&circuit, &ArrayReg::zero_state(2)), None);
//! assert!((p[0] - 0.5).abs() < 1e-12);
//! assert!((p[3] - 0.5).abs() < 1e-12);
//! ```
//!
//! Use [`DensityMatrix`] with [`Register::apply`] for noise channels, and
//! [`expect_grad`] for gradients of unitary, parameterized circuits.
//! [`Circuit::parameters`] and [`Circuit::dispatch`] provide parameter access.
//!
//! # Features
//! - `qasm`: OpenQASM 2.0 import and export.
//! - `omeinsum`: native tensor-network contraction.
//! - `tenferro`: tenferro CPU contraction with reusable plans (Rust 1.96+).
//! - `parallel`: Rayon operations.
//!
//! All features are optional. Qudit circuits support tensor-network export;
//! [`ArrayReg`] and [`DensityMatrix`] simulate qubits only.

pub mod ad;
pub mod apply;
pub mod circuit;
pub mod contraction_plan;

#[cfg(feature = "omeinsum")]
pub mod contractor;
pub mod density_matrix;
pub mod easybuild;
pub mod einsum;
pub mod expect;
pub mod gate;
pub mod instruct_qubit;
pub mod json;
pub mod measure;
pub mod noise;
pub mod operator;
#[cfg(feature = "qasm")]
pub mod qasm;
pub mod register;
pub mod svg;
#[cfg(feature = "tenferro")]
pub mod tenferro;
pub mod tensors;

pub use ad::expect_grad;
pub use apply::{apply, apply_inplace};
pub use circuit::{
    Annotation, Circuit, CircuitElement, PositionedAnnotation, PositionedChannel, PositionedGate,
    channel, control, label, put,
};
#[cfg(feature = "omeinsum")]
pub use contractor::{contract as contract_tn, contract_dm, contract_dm_with_tree};
pub use density_matrix::{DensityMatrix, density_matrix_from_reg};
pub use einsum::{
    TensorNetwork, TensorNetworkDM, circuit_to_einsum, circuit_to_einsum_dm,
    circuit_to_einsum_with_boundary, circuit_to_expectation, circuit_to_expectation_dm,
    circuit_to_overlap,
};
pub use expect::{expect_arrayreg, expect_dm};
pub use gate::Gate;
pub use json::{circuit_from_json, circuit_to_json};
pub use measure::{MeasureResult, PostProcess, measure_with_postprocess, probs};
pub use noise::NoiseChannel;
pub use operator::{Op, OperatorPolynomial, OperatorString, op_matrix};
pub use register::{ArrayReg, Register};
pub use svg::to_svg;
