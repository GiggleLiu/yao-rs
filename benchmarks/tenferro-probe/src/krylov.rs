//! Shared accuracy/work identities for adaptive Hamiltonian evolution.
use crate::Result;
use serde::{Deserialize, Serialize};
use yao_rs::{
    ArrayReg, OperatorPolynomial,
    evolution::{EvolutionOptions, EvolutionResult},
    hamiltonian::{Boundary, PauliHamiltonian, heisenberg, ising},
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Specification {
    pub model: String,
    pub hamiltonian: OperatorPolynomial,
    pub time: f64,
    pub rtol: f64,
    pub krylov_dim: usize,
}

impl Specification {
    pub fn model(&self, n: usize) -> Result<PauliHamiltonian> {
        Ok(PauliHamiltonian::new(n, &self.hamiltonian)?)
    }
    pub fn options(&self) -> EvolutionOptions {
        EvolutionOptions {
            krylov_dim: self.krylov_dim,
            atol: 0.,
            rtol: self.rtol,
            ..Default::default()
        }
    }
    pub fn execute(&self, h: &PauliHamiltonian, input: &ArrayReg) -> Result<EvolutionResult> {
        Ok(h.evolve_krylov(input, self.time, self.options())?)
    }
}

pub fn model(name: &str, n: usize) -> Result<PauliHamiltonian> {
    Ok(match name {
        "ising" => ising(n, -0.7, 0.4, Boundary::Open)?,
        "heisenberg" => heisenberg(n, [0.4, 0.7, -0.3], 0.2, Boundary::Periodic)?,
        _ => return Err("unknown Krylov benchmark model".into()),
    })
}

pub fn record(id: &str, result: &EvolutionResult) -> Result<()> {
    use std::io::Write;
    if let Ok(path) = std::env::var("YAO_BENCH_KRYLOV_LOG") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(
            file,
            "{}",
            serde_json::json!({"id":id,"status":"complete","info":result.info})
        )?;
    }
    Ok(())
}
