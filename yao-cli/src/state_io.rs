use anyhow::{Context, bail};
use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use yao_rs::State;

const FORMAT_VERSION: &str = "yao-state-v1";

#[derive(Serialize, Deserialize)]
struct StateHeader {
    format: String,
    num_qubits: usize,
    dims: Vec<usize>,
    num_elements: usize,
    dtype: String,
}

pub fn write_state(state: &State, path: &Path) -> anyhow::Result<()> {
    let file =
        std::fs::File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    write_state_to_writer(state, &mut writer)
}

pub fn write_state_to_writer(state: &State, writer: &mut impl Write) -> anyhow::Result<()> {
    let header = StateHeader {
        format: FORMAT_VERSION.to_string(),
        num_qubits: state.dims.len(),
        dims: state.dims.clone(),
        num_elements: state.data.len(),
        dtype: "complex128".to_string(),
    };
    let header_json = serde_json::to_string(&header).context("Failed to serialize header")?;
    writer.write_all(header_json.as_bytes())?;
    writer.write_all(b"\n")?;

    for &amplitude in &state.data {
        writer.write_all(&amplitude.re.to_le_bytes())?;
        writer.write_all(&amplitude.im.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_state_from_file(path: &Path) -> anyhow::Result<State> {
    let file = std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    read_state_from_reader(&mut reader)
}

pub fn read_state_from_reader(reader: &mut impl BufRead) -> anyhow::Result<State> {
    let mut header_line = String::new();
    reader
        .read_line(&mut header_line)
        .context("Failed to read state header")?;

    let header: StateHeader =
        serde_json::from_str(&header_line).context("Failed to parse state header")?;

    if header.format != FORMAT_VERSION {
        bail!(
            "Unknown state format: {} (expected {})",
            header.format,
            FORMAT_VERSION
        );
    }
    if header.dtype != "complex128" {
        bail!("Unsupported dtype: {} (expected complex128)", header.dtype);
    }

    let expected_bytes = header.num_elements * 16;
    let mut buf = vec![0u8; expected_bytes];
    reader
        .read_exact(&mut buf)
        .context("Failed to read state data")?;

    let data: Vec<Complex64> = buf
        .chunks_exact(16)
        .map(|chunk| {
            let re = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let im = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
            Complex64::new(re, im)
        })
        .collect();

    Ok(State::new(header.dims, Array1::from(data)))
}

pub fn read_state(path: &str) -> anyhow::Result<State> {
    if path == "-" {
        let stdin = std::io::stdin();
        let mut reader = BufReader::new(stdin.lock());
        read_state_from_reader(&mut reader)
    } else {
        read_state_from_file(Path::new(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_round_trip_file() {
        let dims = vec![2, 2];
        let data = Array1::from(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.5),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, -0.5),
        ]);
        let state = State::new(dims.clone(), data.clone());

        let dir = std::env::temp_dir().join("yao_test_state_io");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_state.bin");

        write_state(&state, &path).unwrap();
        let loaded = read_state_from_file(&path).unwrap();

        assert_eq!(loaded.dims, dims);
        assert_eq!(loaded.data.len(), data.len());
        for (a, b) in loaded.data.iter().zip(data.iter()) {
            assert!((a - b).norm() < 1e-12);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_state_round_trip_bytes() {
        let dims = vec![2, 2, 2];
        let state = State::zero_state(&dims);

        let mut buf = Vec::new();
        write_state_to_writer(&state, &mut buf).unwrap();

        let loaded = read_state_from_reader(&mut &buf[..]).unwrap();
        assert_eq!(loaded.dims, dims);
        assert_eq!(loaded.data.len(), 8);
        assert!((loaded.data[0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }
}
