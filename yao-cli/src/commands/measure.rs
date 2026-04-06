use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;

pub fn measure(
    input: &str,
    shots: usize,
    locs: Option<&[usize]>,
    out: &OutputConfig,
) -> Result<()> {
    let state = state_io::read_state(input)?;
    let mut rng = rand::thread_rng();
    let outcomes = yao_rs::measure(&state, locs, shots, &mut rng);

    let (human, json_value) = super::format_measurement(&outcomes, shots, locs, state.dims.len());

    out.emit(&human, &json_value)
}
