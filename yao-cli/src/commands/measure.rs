use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;

pub fn measure(
    input: &str,
    shots: usize,
    locs: Option<&[usize]>,
    seed: Option<u64>,
    out: &OutputConfig,
) -> Result<()> {
    let state = state_io::read_state(input)?;
    let outcomes = super::sample_state(&state, shots, locs, seed)?;
    let (human, json) = super::format_measurement(&outcomes, shots, locs, state.nqubits());
    out.emit(&human, &json)
}
