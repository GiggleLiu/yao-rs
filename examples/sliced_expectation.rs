use num_complex::Complex64 as C;
use yao_rs::contraction_plan::optimize_code;
use yao_rs::slicing::{SliceBudget, SlicedPlan};
use yao_rs::tenferro::CpuContractor;
use yao_rs::{
    ArrayReg, Circuit, Gate, Op, OperatorPolynomial, OperatorString, apply, circuit_to_expectation,
    put,
};

fn main() -> Result<(), String> {
    let circuit = Circuit::qubits(
        2,
        vec![put(vec![0], Gate::Ry(0.4)), put(vec![1], Gate::Rx(-0.2))],
    )
    .map_err(|e| e.to_string())?;
    let operator = OperatorPolynomial::new(
        vec![C::new(0.4, 0.2), C::new(0.7, 0.), C::new(-0.3, 0.1)],
        vec![
            OperatorString::identity(),
            OperatorString::new(vec![(0, Op::Z)]),
            OperatorString::new(vec![(1, Op::Y)]),
        ],
    );
    let tn = circuit_to_expectation(&circuit, &operator);
    let tree = optimize_code(&tn.code, &tn.size_dict, &omeco::GreedyMethod::default())
        .ok_or("planning failed")?;
    // This qubit network's only dimension-three index selects polynomial terms.
    let selector = *tn
        .size_dict
        .iter()
        .find(|(_, d)| **d == operator.len())
        .ok_or("selector missing")?
        .0;
    let plan = SlicedPlan::new(
        &tn.code,
        &tn.size_dict,
        &tree,
        &[selector],
        SliceBudget::default(),
    )?;
    let cpu = CpuContractor::new(1)?;
    let prepared = cpu.prepare_sliced(&plan)?;
    let result = cpu.execute_sliced(&prepared, &tn.tensors)?[ndarray::IxDyn(&[])];
    let reference =
        yao_rs::expect::expect_arrayreg(&apply(&circuit, &ArrayReg::zero_state(2)), &operator);
    if (result - reference).norm() > 1e-12 {
        return Err("sliced expectation disagrees with simulation".into());
    }
    println!(
        "Expectation: {result}; slices: {}; estimated bytes: {}",
        plan.estimate().slices,
        plan.estimate().estimated_total_bytes
    );
    Ok(())
}
