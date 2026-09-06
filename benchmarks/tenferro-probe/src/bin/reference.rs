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
            "krylov" => {
                let spec = case.krylov.as_ref().ok_or("missing Krylov specification")?;
                let h = spec.model(circuit.nbits)?;
                let tight = h.evolve_krylov(
                    &state,
                    spec.time,
                    yao_rs::evolution::EvolutionOptions {
                        atol: 0.,
                        rtol: 1e-13,
                        krylov_dim: 40,
                        ..Default::default()
                    },
                )?;
                write_output(&directory, &format!("{}.tight", case.id), &tight.state)?;
                spec.execute(&h, &state)?.state
            }
            "state" => apply(&circuit, &state).state,
            "density" => {
                let mut dm = DensityMatrix::from_reg(&state);
                dm.apply(&circuit);
                dm.state
            }
            "expectation" => vec![yao_rs::expect::expect_arrayreg(
                &apply(&circuit, &state),
                case.operator.as_ref().ok_or("missing observable")?,
            )],
            "expectation_dm" => {
                let mut dm = DensityMatrix::from_reg(&state);
                dm.apply(&circuit);
                vec![yao_rs::expect::expect_dm(
                    &dm,
                    case.operator.as_ref().ok_or("missing observable")?,
                )]
            }
            "gradient" => {
                let op = OperatorPolynomial::single(0, Op::Z, 1.0.into());
                let (value, grad) = expect_grad(&op, &circuit, &state);
                std::iter::once(value).chain(grad).map(C::from).collect()
            }
            "custom_gradient" => {
                let c = yao_rs::differentiable::DifferentiableCircuit::from_circuit(circuit)?;
                yao_tenferro_probe::circuit_ad::native(
                    &c,
                    &state,
                    &yao_tenferro_probe::circuit_ad::target(state.nqubits()),
                )?
            }
            _ => return Err("unknown mode".into()),
        };
        write_output(&directory, &case.id, &output)?;
    }
    Ok(())
}

fn write_output(directory: &str, id: &str, values: &[C]) -> Result<()> {
    let mut file = BufWriter::new(File::create(
        Path::new(directory).join(format!("{id}.bin")),
    )?);
    for x in values {
        file.write_all(&x.re.to_le_bytes())?;
        file.write_all(&x.im.to_le_bytes())?;
    }
    Ok(())
}
