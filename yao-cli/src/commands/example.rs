use crate::output::OutputConfig;
use anyhow::{Result, bail};
use yao_rs::{Circuit, Gate, circuit_to_json, control, put};

#[derive(Debug, Default, Clone)]
pub struct ExampleOptions {
    pub nqubits: Option<usize>,
    pub preset: Option<String>,
    pub secret: Option<String>,
    pub marked: Option<usize>,
    pub iterations: Option<String>,
    pub depth: Option<usize>,
    pub phase: Option<f64>,
    pub nqubits_per_state: Option<usize>,
    pub nstates: Option<usize>,
}

pub fn example(name: &str, opts: ExampleOptions, out: &OutputConfig) -> Result<()> {
    let circuit = match name {
        "bell" => bell(opts.nqubits.unwrap_or(2))?,
        "ghz" => ghz(opts.nqubits.unwrap_or(3))?,
        "qft" => qft(opts.nqubits.unwrap_or(4))?,
        _ => bail!("Unknown example: '{name}'\n\nAvailable examples: bell, ghz, qft"),
    };
    let json_value: serde_json::Value = serde_json::from_str(&circuit_to_json(&circuit))?;
    out.emit(&format!("{circuit}"), &json_value)
}

fn bell(n: usize) -> Result<Circuit> {
    if n < 2 {
        bail!("Bell circuit requires at least 2 qubits");
    }
    let elements = vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)];
    Circuit::qubits(n, elements).map_err(|e| anyhow::anyhow!("{e}"))
}

fn ghz(n: usize) -> Result<Circuit> {
    if n < 2 {
        bail!("GHZ circuit requires at least 2 qubits");
    }
    let mut elements = vec![put(vec![0], Gate::H)];
    for i in 1..n {
        elements.push(control(vec![0], vec![i], Gate::X));
    }
    Circuit::qubits(n, elements).map_err(|e| anyhow::anyhow!("{e}"))
}

fn qft(n: usize) -> Result<Circuit> {
    if n < 1 {
        bail!("QFT circuit requires at least 1 qubit");
    }
    let circuit = yao_rs::easybuild::qft_circuit(n);
    Ok(circuit)
}
