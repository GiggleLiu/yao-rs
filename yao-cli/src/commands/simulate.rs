use crate::output::OutputConfig;
use anyhow::Result;

pub fn simulate(_circuit: &str, _input: Option<&str>, _out: &OutputConfig) -> Result<()> {
    anyhow::bail!("simulate: not yet implemented")
}
