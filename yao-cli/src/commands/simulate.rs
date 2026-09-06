use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use std::io::{BufWriter, IsTerminal};

pub fn simulate(circuit_path: &str, input_path: Option<&str>, out: &OutputConfig) -> Result<()> {
    let circuit = super::load_circuit(circuit_path)?;

    anyhow::ensure!(
        circuit_path != "-" || input_path != Some("-"),
        "Circuit and input state cannot both read from stdin"
    );
    let mut result = super::simulation_input(&circuit, input_path)?;

    result.apply(&circuit)?;

    if let Some(ref path) = out.output {
        state_io::write_state(&result, path)?;
        out.info(&format!("State written to {}", path.display()));
    } else if std::io::stdout().is_terminal() {
        anyhow::bail!(
            "refusing to write binary state to terminal.\n\
             Use --output <file> or pipe to another command (e.g. yao probs -)."
        );
    } else {
        let stdout = std::io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        state_io::write_state_to_writer(&result, &mut writer)?;
    }

    Ok(())
}
