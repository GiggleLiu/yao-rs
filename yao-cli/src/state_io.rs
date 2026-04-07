use anyhow::{Context, bail};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use yao_rs::ArrayReg;

const FORMAT_VERSION: &str = "yao-state-v1";

#[derive(Serialize, Deserialize)]
struct StateHeader {
    format: String,
    num_qubits: usize,
    dims: Vec<usize>,
    num_elements: usize,
    dtype: String,
}

pub fn write_state(reg: &ArrayReg, path: &Path) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    write_state_to_writer(reg, &mut writer)
}

pub fn write_state_to_writer(reg: &ArrayReg, writer: &mut impl Write) -> anyhow::Result<()> {
    let nbits = reg.nqubits();
    let header = StateHeader {
        format: FORMAT_VERSION.to_string(),
        num_qubits: nbits,
        dims: vec![2; nbits],
        num_elements: reg.state_vec().len(),
        dtype: "complex128".to_string(),
    };
    let header_json = serde_json::to_string(&header).context("Failed to serialize header")?;
    writer.write_all(header_json.as_bytes())?;
    writer.write_all(b"\n")?;

    for &amplitude in reg.state_vec() {
        writer.write_all(&amplitude.re.to_le_bytes())?;
        writer.write_all(&amplitude.im.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_state_from_file(path: &Path) -> anyhow::Result<ArrayReg> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    read_state_from_reader(&mut reader)
}

pub fn read_state_from_reader(reader: &mut impl BufRead) -> anyhow::Result<ArrayReg> {
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

    let expected_from_dims: usize = header.dims.iter().product();
    if header.num_elements != expected_from_dims {
        bail!(
            "Header mismatch: num_elements={} but dims {:?} imply {} elements",
            header.num_elements,
            header.dims,
            expected_from_dims
        );
    }

    if !header.dims.iter().all(|&d| d == 2) {
        bail!(
            "Only qubit (d=2) states are supported, got dims {:?}",
            header.dims
        );
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

    Ok(ArrayReg::from_vec(header.num_qubits, data))
}

pub fn read_state(path: &str) -> anyhow::Result<ArrayReg> {
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
        let reg = ArrayReg::from_vec(
            2,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, 0.5),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, -0.5),
            ],
        );

        let dir = std::env::temp_dir().join("yao_test_state_io");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_state.bin");

        write_state(&reg, &path).unwrap();
        let loaded = read_state_from_file(&path).unwrap();

        assert_eq!(loaded.nqubits(), 2);
        assert_eq!(loaded.state_vec().len(), 4);
        for (a, b) in loaded.state_vec().iter().zip(reg.state_vec().iter()) {
            assert!((a - b).norm() < 1e-12);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_state_round_trip_bytes() {
        let reg = ArrayReg::zero_state(3);

        let mut buf = Vec::new();
        write_state_to_writer(&reg, &mut buf).unwrap();

        let loaded = read_state_from_reader(&mut &buf[..]).unwrap();
        assert_eq!(loaded.nqubits(), 3);
        assert_eq!(loaded.state_vec().len(), 8);
        for (a, b) in loaded.state_vec().iter().zip(reg.state_vec().iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_rejects_mismatched_dims_and_num_elements() {
        let header_json = serde_json::json!({
            "format": "yao-state-v1",
            "num_qubits": 2,
            "dims": [2, 2],
            "num_elements": 999,
            "dtype": "complex128",
        });
        let mut buf = Vec::new();
        buf.extend_from_slice(header_json.to_string().as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(&[0u8; 64]);

        let result = read_state_from_reader(&mut &buf[..]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("num_elements=999"));
    }
}
