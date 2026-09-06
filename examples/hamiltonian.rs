//! cargo run --release --example hamiltonian --features tenferro
use yao_rs::hamiltonian::{Boundary, ProductFormula, ising};
use yao_rs::{ArrayReg, Op, OperatorPolynomial, expect_grad};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Pauli convention: H = -0.7 Σ ZZ + 0.4 Σ X on an open chain.
    let model = ising(3, -0.7, 0.4, Boundary::Open)?;
    let mut evolution = model.evolve(0.8, 8, ProductFormula::Suzuki2)?;
    let input = ArrayReg::zero_state(3);
    let observable = OperatorPolynomial::single(0, Op::Z, 1.0.into());
    let (value, angle_gradient) = expect_grad(&observable, evolution.circuit(), &input);
    println!("<Z(0)> = {value:.10}");
    println!("[time, J, h] = {:?}", evolution.parameters());
    println!(
        "d<Z(0)>/d[time, J, h] = {:?}",
        evolution.pullback(&angle_gradient)?
    );

    // One update changes every occurrence of the shared time and couplings.
    evolution.dispatch(&[0.9, -0.7, 0.45])?;
    let output = evolution.apply(&input)?;
    println!(
        "State norm² = {:.12}",
        output.state.iter().map(|x| x.norm_sqr()).sum::<f64>()
    );

    #[cfg(feature = "tenferro")]
    {
        let tn = yao_rs::circuit_to_einsum_with_boundary(evolution.circuit(), &[]);
        let cpu = yao_rs::tenferro::CpuContractor::new(1)?;
        let plan = cpu.prepare(&tn.code, &tn.size_dict, None)?;
        let contracted = cpu.execute(&plan, &tn.tensors)?;
        assert!(
            contracted
                .iter()
                .zip(&output.state)
                .all(|(a, b)| (a - b).norm() < 1e-11)
        );
        println!("Tenferro contraction agrees with native evolution.");
    }
    Ok(())
}
