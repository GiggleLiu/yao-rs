use crate::output::OutputConfig;
use anyhow::{Result, bail};
use yao_rs::{Circuit, Gate, circuit_to_json, control, put};

const SUPPORTED_EXAMPLES: &str =
    "bell, ghz, qft, phase-estimation, hadamard-test, swap-test, bernstein-vazirani, grover, qaoa-maxcut";

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
        "phase-estimation" => {
            let n_reg = opts.nqubits.unwrap_or(3);
            let unitary = preset_unitary(opts.preset.as_deref(), opts.phase)?;
            yao_rs::easybuild::phase_estimation_circuit(unitary, n_reg, 1)
        }
        "hadamard-test" => {
            let unitary = preset_unitary(opts.preset.as_deref(), opts.phase)?;
            yao_rs::easybuild::hadamard_test_circuit(unitary, opts.phase.unwrap_or(0.0))
        }
        "swap-test" => {
            let nbit = opts.nqubits_per_state.unwrap_or(1);
            let nstate = opts.nstates.unwrap_or(2);
            if nbit == 0 {
                bail!("swap-test requires --nqubits-per-state >= 1");
            }
            if nstate < 2 {
                bail!("swap-test requires --nstates >= 2");
            }
            yao_rs::easybuild::swap_test_circuit(nbit, nstate, opts.phase.unwrap_or(0.0))
        }
        "bernstein-vazirani" => {
            let secret = parse_secret_bits(opts.secret.as_deref())?;
            yao_rs::easybuild::bernstein_vazirani_circuit(&secret)
        }
        "grover" => {
            let n = opts.nqubits.unwrap_or(3);
            if n == 0 || n > 8 {
                bail!("grover requires 1 <= --nqubits <= 8");
            }
            let marked = opts.marked.unwrap_or((1usize << n) - 1);
            if marked >= (1usize << n) {
                bail!("marked state out of range for {n} qubits");
            }
            let iterations = grover_iterations(n, opts.iterations.as_deref())?;
            yao_rs::easybuild::marked_state_grover_circuit(n, marked, iterations)
        }
        "qaoa-maxcut" => {
            let depth = opts.depth.unwrap_or(1);
            if depth == 0 {
                bail!("qaoa-maxcut requires --depth >= 1");
            }
            let (n, edges) = qaoa_preset_edges(opts.preset.as_deref())?;
            let gammas = vec![0.2; depth];
            let betas = vec![0.3; depth];
            yao_rs::easybuild::qaoa_maxcut_circuit(n, &edges, &gammas, &betas)
        }
        _ => bail!("Unknown example: '{name}'\n\nAvailable examples: {SUPPORTED_EXAMPLES}"),
    };
    let json_value: serde_json::Value = serde_json::from_str(&circuit_to_json(&circuit))?;
    out.emit(&format!("{circuit}"), &json_value)
}

fn preset_unitary(name: Option<&str>, phase: Option<f64>) -> Result<Gate> {
    match name.unwrap_or("z") {
        "z" => Ok(Gate::Z),
        "x" => Ok(Gate::X),
        "t" => Ok(Gate::T),
        "phase" => Ok(Gate::Phase(
            phase.unwrap_or(std::f64::consts::PI / 4.0),
        )),
        other => bail!("Unknown unitary preset: '{other}' (available: z, x, t, phase)"),
    }
}

fn parse_secret_bits(secret: Option<&str>) -> Result<Vec<bool>> {
    let secret = secret.ok_or_else(|| anyhow::anyhow!("bernstein-vazirani requires --secret"))?;
    if secret.is_empty() {
        bail!("bernstein-vazirani requires a non-empty --secret");
    }
    secret
        .chars()
        .map(|ch| match ch {
            '0' => Ok(false),
            '1' => Ok(true),
            _ => bail!("secret must contain only 0 and 1"),
        })
        .collect()
}

fn qaoa_preset_edges(preset: Option<&str>) -> Result<(usize, Vec<(usize, usize, f64)>)> {
    match preset.unwrap_or("line4") {
        "line4" => Ok((4, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])),
        "triangle" => Ok((3, vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])),
        other => bail!("Unknown qaoa-maxcut preset: '{other}' (available: line4, triangle)"),
    }
}

fn grover_iterations(n: usize, iterations: Option<&str>) -> Result<usize> {
    match iterations.unwrap_or("auto") {
        "auto" => Ok(yao_rs::easybuild::grover_auto_iterations(n, 1)),
        value => value
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("--iterations must be a non-negative integer or auto")),
    }
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
