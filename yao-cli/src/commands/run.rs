use crate::output::OutputConfig;
use anyhow::Result;

pub fn run(
    _circuit: &str,
    _input: Option<&str>,
    _shots: Option<usize>,
    _op: Option<&str>,
    _locs: Option<&[usize]>,
    _out: &OutputConfig,
) -> Result<()> {
    anyhow::bail!("run: not yet implemented")
}
