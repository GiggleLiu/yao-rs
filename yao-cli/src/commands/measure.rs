use crate::output::OutputConfig;
use anyhow::Result;

pub fn measure(
    _input: &str,
    _shots: usize,
    _locs: Option<&[usize]>,
    _out: &OutputConfig,
) -> Result<()> {
    anyhow::bail!("measure: not yet implemented")
}
