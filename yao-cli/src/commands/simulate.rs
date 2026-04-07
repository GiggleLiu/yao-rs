use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use std::io::{BufWriter, IsTerminal};
use yao_rs::{ArrayReg, apply};

pub fn simulate(circuit_path: &str, input_path: Option<&str>, out: &OutputConfig) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    let input_state = if let Some(path) = input_path {
        state_io::read_state(path)?
    } else {
        ArrayReg::zero_state(circuit.nbits)
    };

    let result = apply(&circuit, &input_state);

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
