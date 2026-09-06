use anyhow::{Context, bail};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use yao_rs::{ArrayReg, Circuit, CircuitElement, DensityMatrix, OperatorPolynomial, Register};

const FORMAT_VERSION: &str = "yao-state-v1";
const DENSITY_FORMAT_VERSION: &str = "yao-density-v1";

#[derive(Debug)]
pub enum State {
    Pure(ArrayReg),
    Density(DensityMatrix),
}

impl State {
    pub fn nqubits(&self) -> usize {
        match self {
            Self::Pure(reg) => reg.nqubits(),
            Self::Density(dm) => dm.nbits(),
        }
    }

    pub fn state_vec(&self) -> &[Complex64] {
        match self {
            Self::Pure(reg) => reg.state_vec(),
            Self::Density(dm) => dm.state_data(),
        }
    }

    pub fn probs(&self, locs: Option<&[usize]>) -> Vec<f64> {
        match self {
            Self::Pure(reg) => yao_rs::probs(reg, locs),
            Self::Density(dm) => yao_rs::probs(dm, locs),
        }
    }

    pub fn expectation(&self, operator: &OperatorPolynomial) -> Complex64 {
        match self {
            Self::Pure(reg) => yao_rs::expect_arrayreg(reg, operator),
            Self::Density(dm) => yao_rs::expect_dm(dm, operator),
        }
    }

    pub fn apply(&mut self, circuit: &Circuit) -> anyhow::Result<()> {
        if circuit
            .elements
            .iter()
            .any(|e| matches!(e, CircuitElement::Channel(_)))
        {
            checked_elements(self.nqubits(), true)?;
            if let Self::Pure(reg) = self {
                *self = Self::Density(DensityMatrix::from_reg(reg));
            }
        }
        match self {
            Self::Pure(reg) => yao_rs::apply_inplace(circuit, reg),
            Self::Density(dm) => dm.apply(circuit),
        }
        Ok(())
    }
}

pub fn checked_elements(nqubits: usize, density: bool) -> anyhow::Result<usize> {
    let exponent = nqubits
        .checked_mul(if density { 2 } else { 1 })
        .and_then(|n| u32::try_from(n).ok());
    let elements = exponent
        .and_then(|n| 1usize.checked_shl(n))
        .ok_or_else(|| {
            anyhow::anyhow!("Too many qubits for this state representation: {nqubits}")
        })?;
    anyhow::ensure!(
        elements <= isize::MAX as usize / 16,
        "State for {nqubits} qubits exceeds the supported allocation size"
    );
    Ok(elements)
}

#[derive(Serialize, Deserialize)]
struct StateHeader {
    format: String,
    num_qubits: usize,
    dims: Vec<usize>,
    num_elements: usize,
    dtype: String,
}

pub fn write_state(reg: &State, path: &Path) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    write_state_to_writer(reg, &mut writer)
}

pub fn write_state_to_writer(reg: &State, writer: &mut impl Write) -> anyhow::Result<()> {
    let nbits = reg.nqubits();
    let header = StateHeader {
        format: if matches!(reg, State::Density(_)) {
            DENSITY_FORMAT_VERSION
        } else {
            FORMAT_VERSION
        }
        .to_string(),
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

pub fn read_state_from_file(path: &Path) -> anyhow::Result<State> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
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

    let density = match header.format.as_str() {
        FORMAT_VERSION => false,
        DENSITY_FORMAT_VERSION => true,
        other => bail!("Unknown state format: {other}"),
    };
    anyhow::ensure!(
        header.dtype == "complex128",
        "Unsupported dtype: {} (expected complex128)",
        header.dtype
    );
    anyhow::ensure!(
        header.dims.len() == header.num_qubits,
        "Header mismatch: num_qubits={} but dims has {} sites",
        header.num_qubits,
        header.dims.len()
    );
    anyhow::ensure!(
        header.dims.iter().all(|&d| d == 2),
        "Only qubit (d=2) states are supported"
    );
    let expected = checked_elements(header.num_qubits, density)?;
    anyhow::ensure!(
        header.num_elements == expected,
        "Header mismatch: num_elements={} but state requires {expected} elements",
        header.num_elements
    );

    // Read only data actually supplied, so a forged header cannot cause a huge
    // up-front allocation. The bounded reader also works with piped input.
    let expected_bytes = expected * 16;
    let mut buf = Vec::new();
    reader
        .take(expected_bytes as u64)
        .read_to_end(&mut buf)
        .context("Failed to read state data")?;
    anyhow::ensure!(
        buf.len() == expected_bytes,
        "Truncated state data: expected {expected_bytes} bytes, got {}",
        buf.len()
    );

    let data: Vec<Complex64> = buf
        .chunks_exact(16)
        .map(|chunk| {
            let re = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let im = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
            Complex64::new(re, im)
        })
        .collect();

    anyhow::ensure!(
        data.iter().all(|c| c.re.is_finite() && c.im.is_finite()),
        "State amplitudes must be finite"
    );
    Ok(if density {
        State::Density(DensityMatrix::from_vec(header.num_qubits, data))
    } else {
        State::Pure(ArrayReg::from_vec(header.num_qubits, data))
    })
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

        write_state(&State::Pure(reg.clone()), &path).unwrap();
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
        write_state_to_writer(&State::Pure(reg.clone()), &mut buf).unwrap();

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
