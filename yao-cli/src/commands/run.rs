use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use std::io::{BufWriter, IsTerminal};

#[allow(clippy::too_many_arguments)]
pub fn run(
    circuit_path: &str,
    input_path: Option<&str>,
    shots: Option<usize>,
    op: Option<&str>,
    locs: Option<&[usize]>,
    seed: Option<u64>,
    out: &OutputConfig,
) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    anyhow::ensure!(
        circuit_path != "-" || input_path != Some("-"),
        "Circuit and input state cannot both read from stdin"
    );
    let mut result = super::simulation_input(&circuit, input_path)?;

    super::validate_locs(circuit.nbits, locs)?;
    result.apply(&circuit)?;
    let nbits = result.nqubits();

    if let Some(nshots) = shots {
        let outcomes = super::sample_state(&result, nshots, locs, seed)?;

        let (human, json_value) = super::format_measurement(&outcomes, nshots, locs, nbits);

        out.emit(&human, &json_value)
    } else if let Some(op_str) = op {
        let operator = crate::operator_parser::parse_operator_for_qubits(op_str, nbits)?;
        let value = result.expectation(&operator);

        let (human, json_value) = super::format_expectation(op_str, value);

        out.emit(&human, &json_value)
    } else {
        if let Some(ref path) = out.output {
            state_io::write_state(&result, path)?;
            out.info(&format!("State written to {}", path.display()));
        } else if std::io::stdout().is_terminal() {
            anyhow::bail!(
                "refusing to write binary state to terminal.\n\
                 Use --output <file>, --shots <n>, or pipe to another command."
            );
        } else {
            let stdout = std::io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            state_io::write_state_to_writer(&result, &mut writer)?;
        }
        Ok(())
    }
}
