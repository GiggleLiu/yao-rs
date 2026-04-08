//! Deserialization helpers for benchmark ground-truth JSON data.

#![allow(dead_code)]

use std::collections::HashMap;

use num_complex::Complex64;
use serde::Deserialize;

/// State vectors stored as interleaved [re0, im0, re1, im1, ...].
/// Outer key = gate name, inner key = nqubits (as string).
pub type GateData = HashMap<String, HashMap<String, Vec<f64>>>;

/// QFT data: key = nqubits (as string), value = interleaved state vector.
pub type QftData = HashMap<String, Vec<f64>>;

/// A single noisy-circuit entry for one qubit count.
#[derive(Deserialize)]
pub struct NoisyEntry {
    pub trace: f64,
    pub purity: f64,
    pub entropy: f64,
    pub expect_ising: ExpectComplex,
    #[serde(default)]
    pub density_matrix: Option<Vec<f64>>,
    #[serde(default)]
    pub reduced_dm: Option<Vec<f64>>,
}

#[derive(Deserialize)]
pub struct ExpectComplex {
    pub re: f64,
    pub im: f64,
}

/// Noisy circuit data: key = nqubits (as string).
pub type NoisyData = HashMap<String, NoisyEntry>;

/// Convert interleaved [re0, im0, re1, im1, ...] to Vec<Complex64>.
pub fn interleaved_to_complex(data: &[f64]) -> Vec<Complex64> {
    data.chunks_exact(2)
        .map(|chunk| Complex64::new(chunk[0], chunk[1]))
        .collect()
}

/// Load JSON file from benchmarks/data/ by filename.
/// Returns None if the file doesn't exist (benchmark data not yet generated).
fn try_load_benchmark_json(filename: &str) -> Option<String> {
    let path = format!(
        "{}/benchmarks/data/{}",
        env!("CARGO_MANIFEST_DIR"),
        filename
    );
    std::fs::read_to_string(path).ok()
}

pub fn load_single_gate_1q() -> Option<GateData> {
    Some(serde_json::from_str(&try_load_benchmark_json("single_gate_1q.json")?).unwrap())
}

pub fn load_single_gate_2q() -> Option<GateData> {
    Some(serde_json::from_str(&try_load_benchmark_json("single_gate_2q.json")?).unwrap())
}

pub fn load_single_gate_multi() -> Option<GateData> {
    Some(serde_json::from_str(&try_load_benchmark_json("single_gate_multi.json")?).unwrap())
}

pub fn load_qft() -> Option<QftData> {
    Some(serde_json::from_str(&try_load_benchmark_json("qft.json")?).unwrap())
}

pub fn load_noisy_circuit() -> Option<NoisyData> {
    Some(serde_json::from_str(&try_load_benchmark_json("noisy_circuit.json")?).unwrap())
}
