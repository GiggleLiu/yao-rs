//! Shared trajectory workloads and raw Monte Carlo diagnostics.
use crate::Result;
use num_complex::Complex64 as C;
use std::io::Write;
use yao_rs::trajectories::{TrajectoryCircuit, TrajectoryOptions};
use yao_rs::{
    Circuit, Gate, NoiseChannel, Op, OperatorPolynomial, OperatorString, channel, control, put,
};

pub const COUNTS: [usize; 3] = [128, 512, 2048];
pub const LARGE_COUNTS: [usize; 2] = [64, 256];
pub fn circuit(n: usize, entangled: bool) -> Result<Circuit> {
    let mut elements = Vec::new();
    for layer in 0..if entangled { 2 } else { 1 } {
        for q in 0..n {
            elements.push(put(vec![q], Gate::Ry(0.31 + q as f64 * 0.07)));
            if entangled {
                elements.push(put(vec![q], Gate::Rz(0.23 + layer as f64 * 0.11)));
            }
        }
        if entangled {
            for q in 0..n - 1 {
                elements.push(control(vec![q], vec![q + 1], Gate::X));
            }
        }
        for q in 0..n {
            if entangled {
                elements.push(channel(
                    vec![q],
                    NoiseChannel::AmplitudeDamping {
                        gamma: 0.15,
                        excited_population: 0.1,
                    },
                ));
                elements.push(channel(
                    vec![q],
                    NoiseChannel::Depolarizing { n: 1, p: 0.08 },
                ));
            } else {
                elements.push(channel(vec![q], NoiseChannel::BitFlip { p: 0.17 }));
            }
        }
    }
    Ok(Circuit::qubits(n, elements)?)
}
pub fn observable(n: usize, entangled: bool) -> OperatorPolynomial {
    if !entangled {
        return OperatorPolynomial::single(0, Op::Z, 1.0.into());
    }
    OperatorPolynomial::new(
        vec![C::new(0.6, 0.1), C::new(0.2, 0.), C::new(0., 0.1)],
        vec![
            OperatorString::new(vec![(0, Op::Z), (n - 1, Op::Z)]),
            OperatorString::new(vec![(n - 1, Op::Y)]),
            OperatorString::new(vec![(1, Op::Pu)]),
        ],
    )
}
pub fn product_expected() -> C {
    C::new(0.31_f64.cos() * 0.66, 0.)
}
pub fn record(
    id: &str,
    sim: &TrajectoryCircuit,
    input: &yao_rs::ArrayReg,
    op: &OperatorPolynomial,
    expected: C,
    options: TrajectoryOptions,
) -> Result<()> {
    let stats = sim.expectation(input, op, options)?;
    let err = stats.mean - expected;
    let se = stats
        .standard_error
        .ok_or("diagnostic needs at least two trajectories")?;
    if err.re.abs() > 8. * se.re + 1e-10 || err.im.abs() > 8. * se.im + 1e-10 {
        return Err("trajectory correctness check exceeds eight standard errors".into());
    }
    if let Ok(path) = std::env::var("YAO_BENCH_STATS_LOG") {
        let record = serde_json::json!({"id":id,"expected":expected,"error":err,"statistics":stats,"status":"complete"});
        writeln!(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?,
            "{record}"
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use yao_rs::{ArrayReg, DensityMatrix, Register};
    #[test]
    fn product_reference_is_independent_of_register_size() -> Result<()> {
        for n in [2, 4, 6] {
            let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(n));
            dm.apply(&circuit(n, false)?);
            assert!(
                (yao_rs::expect_dm(&dm, &observable(n, false)) - product_expected()).norm() < 1e-12
            );
        }
        Ok(())
    }
}
