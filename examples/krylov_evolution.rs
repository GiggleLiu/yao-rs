//! Adaptive Hamiltonian evolution without constructing the full matrix.
use yao_rs::{
    ArrayReg, Op, OperatorPolynomial,
    evolution::EvolutionOptions,
    expect_arrayreg,
    hamiltonian::{Boundary, ising},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let h = ising(12, -0.7, 0.4, Boundary::Open)?;
    let initial = ArrayReg::zero_state(12);
    let result = h.evolve_krylov(&initial, 0.8, EvolutionOptions::default())?;
    println!("{:?}", result.info);
    let state = ArrayReg::from_vec(12, result.state);
    let z = OperatorPolynomial::single(0, Op::Z, 1.0.into());
    println!("<Z(0)> = {}", expect_arrayreg(&state, &z));
    let reversed = h.evolve_krylov(&state, -0.8, EvolutionOptions::default())?;
    let error = initial
        .state
        .iter()
        .zip(&reversed.state)
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(error < 1e-9, "forward/backward error {error}");
    println!("Forward/backward state error: {error:.3e}");
    Ok(())
}
