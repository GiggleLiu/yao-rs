use yao_rs::trajectories::{TrajectoryCircuit, TrajectoryOptions};
use yao_rs::{
    ArrayReg, Circuit, Gate, NoiseChannel, Op, OperatorPolynomial, channel, control, put,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let circuit = Circuit::qubits(
        2,
        vec![
            put(vec![0], Gate::H),
            control(vec![0], vec![1], Gate::X),
            channel(
                vec![1],
                NoiseChannel::AmplitudeDamping {
                    gamma: 0.3,
                    excited_population: 0.,
                },
            ),
        ],
    )?;
    let simulator = TrajectoryCircuit::new(circuit)?;
    let op = OperatorPolynomial::single(1, Op::Z, 1.0.into());
    let stats = simulator.expectation(
        &ArrayReg::zero_state(2),
        &op,
        TrajectoryOptions {
            trajectories: 8192,
            seed: 7,
            threads: 1,
        },
    )?;
    println!(
        "Z(1): {} (exact 0.3), standard error {:?}, {} trajectories, seed {}",
        stats.mean, stats.standard_error, stats.trajectories, stats.seed
    );
    Ok(())
}
