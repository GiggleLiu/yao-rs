use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use std::io::{BufWriter, IsTerminal};
use yao_rs::{State, apply};

pub fn run(
    circuit_path: &str,
    input_path: Option<&str>,
    shots: Option<usize>,
    op: Option<&str>,
    locs: Option<&[usize]>,
    out: &OutputConfig,
) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    let input_state = if let Some(path) = input_path {
        state_io::read_state(path)?
    } else {
        State::zero_state(&circuit.dims)
    };

    let result = apply(&circuit, &input_state);

    if let Some(nshots) = shots {
        let mut rng = rand::thread_rng();
        let outcomes = yao_rs::measure(&result, locs, nshots, &mut rng);

        let (human, json_value) =
            super::format_measurement(&outcomes, nshots, locs, result.dims.len());

        out.emit(&human, &json_value)
    } else if let Some(op_str) = op {
        let operator = crate::operator_parser::parse_operator(op_str)?;
        let value = crate::commands::expect::compute_expectation(&result, &operator);

        let (human, json_value) = super::format_expectation(op_str, value);

        out.emit(&human, &json_value)
    } else {
        if let Some(ref path) = out.output {
            state_io::write_state(&result, path)?;
            out.info(&format!("State written to {}", path.display()));
        } else {
            if std::io::stdout().is_terminal() {
                out.info("hint: writing binary state to stdout; pipe to another command or use --output <file>");
            }
            let stdout = std::io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            state_io::write_state_to_writer(&result, &mut writer)?;
        }
        Ok(())
    }
}
