//! Generate full output oracles outside the timed region for the Julia runner.
use num_complex::Complex64 as C;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
use yao_rs::{DensityMatrix, Op, OperatorPolynomial, Register, apply, expect_grad};
use yao_tenferro_probe::{Result, cases};
fn main() -> Result<()> {
    let directory = std::env::args()
        .nth(1)
        .ok_or("usage: reference OUTPUT_DIRECTORY")?;
    fs::create_dir_all(&directory)?;
    for case in cases()? {
        let circuit = case.circuit()?;
        let state = case.initial(circuit.nbits);
        let output: Vec<C> = match case.mode.as_str() {
            "state" => apply(&circuit, &state).state,
            "density" => {
                let mut dm = DensityMatrix::from_reg(&state);
                dm.apply(&circuit);
                dm.state
            }
            "gradient" => {
                let op = OperatorPolynomial::single(0, Op::Z, 1.0.into());
                let (value, grad) = expect_grad(&op, &circuit, &state);
                std::iter::once(value).chain(grad).map(C::from).collect()
            }
            _ => return Err("unknown mode".into()),
        };
        let mut file = BufWriter::new(File::create(
            Path::new(&directory).join(format!("{}.bin", case.id)),
        )?);
        for x in output {
            file.write_all(&x.re.to_le_bytes())?;
            file.write_all(&x.im.to_le_bytes())?;
        }
    }
    Ok(())
}
