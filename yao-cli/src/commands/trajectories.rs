use crate::{
    output::OutputConfig,
    state_io::{self, State},
};
use anyhow::{Result, anyhow, bail};
use rand::RngExt;
use yao_rs::{
    ArrayReg, Circuit,
    trajectories::{TrajectoryCircuit, TrajectoryOptions},
};

pub fn run(
    circuit: Circuit,
    input_path: Option<&str>,
    op: Option<&str>,
    count: usize,
    seed: Option<u64>,
    threads: usize,
    out: &OutputConfig,
) -> Result<()> {
    let op = op.ok_or_else(|| anyhow!("Trajectory mode requires --op"))?;
    let n = circuit.nbits;
    let operator = crate::operator_parser::parse_operator_for_qubits(op, n)?;
    let simulator = TrajectoryCircuit::new(circuit).map_err(|e| anyhow!(e))?;
    state_io::checked_elements(n, false)?;
    let input = match input_path {
        Some(path) => match state_io::read_state(path)? {
            State::Pure(reg) => reg,
            State::Density(_) => bail!(
                "Trajectory input must be a normalized pure state; use exact simulation for density input"
            ),
        },
        None => ArrayReg::zero_state(n),
    };
    let seed = seed.unwrap_or_else(|| rand::rng().random());
    let statistics = simulator
        .expectation(
            &input,
            &operator,
            TrajectoryOptions {
                trajectories: count,
                seed,
                threads,
            },
        )
        .map_err(|e| anyhow!(e))?;
    let uncertainty = statistics.standard_error.map_or_else(
        || "unavailable (one trajectory)".to_owned(),
        |se| format!("real {:.6e}, imaginary {:.6e}", se.re, se.im),
    );
    let human = format!(
        "Trajectory expectation ⟨{op}⟩ = {}\nTrajectories: {count}; seed: {seed}; threads: {threads}\nStandard error: {uncertainty}",
        statistics.mean
    );
    let json = serde_json::json!({"mode":"trajectories","num_qubits":n,"operator":op,"statistics":statistics});
    out.emit(&human, &json)
}
