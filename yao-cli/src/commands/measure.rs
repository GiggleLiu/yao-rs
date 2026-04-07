use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use yao_rs::measure::{MeasureResult, PostProcess, measure_with_postprocess};

pub fn measure(
    input: &str,
    shots: usize,
    locs: Option<&[usize]>,
    out: &OutputConfig,
) -> Result<()> {
    let mut reg = state_io::read_state(input)?;
    let nbits = reg.nqubits();
    let measure_locs: Vec<usize> = locs.map_or_else(|| (0..nbits).collect(), |l| l.to_vec());
    let mut rng = rand::thread_rng();

    let mut outcomes = Vec::with_capacity(shots);
    for _ in 0..shots {
        match measure_with_postprocess(
            &mut reg,
            &measure_locs,
            PostProcess::NoPostProcess,
            &mut rng,
        ) {
            MeasureResult::Value(bits) => outcomes.push(bits),
            MeasureResult::Removed(_, _) => unreachable!(),
        }
    }

    let (human, json_value) = super::format_measurement(&outcomes, shots, locs, nbits);

    out.emit(&human, &json_value)
}
