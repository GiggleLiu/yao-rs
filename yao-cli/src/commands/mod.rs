pub mod expect;
pub mod inspect;
pub mod measure;
pub mod probs;
pub mod run;
pub mod simulate;
pub mod toeinsum;
#[cfg(feature = "typst")]
pub mod visualize;

use anyhow::{Context, anyhow};
use std::io::Read;
use yao_rs::Circuit;

pub fn load_circuit(path: &str) -> anyhow::Result<Circuit> {
    let json = if path == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read circuit from stdin")?;
        buf
    } else {
        std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read circuit from '{path}'"))?
    };

    yao_rs::circuit_from_json(&json).map_err(|e| anyhow!("Failed to parse circuit: {e}"))
}
