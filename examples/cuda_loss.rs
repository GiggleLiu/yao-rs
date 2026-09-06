//! GPU-resident circuit, smooth custom loss and parameter updates.
//! cargo run --release --features cuda --example cuda_loss
use tenferro_ad::EagerTensor;
use tenferro_tensor::{DType, Tensor};
use yao_rs::cuda::CudaSimulator;
use yao_rs::differentiable::DifferentiableCircuit;
use yao_rs::{ArrayReg, Circuit, Gate, control, put};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let circuit = DifferentiableCircuit::from_circuit(Circuit::qubits(
        2,
        vec![
            put(vec![0], Gate::Ry(0.2)),
            control(vec![0], vec![1], Gate::X),
            put(vec![1], Gate::Rz(0.1)),
        ],
    )?)?;
    let gpu = CudaSimulator::new(0)?;
    let prepared = gpu.prepare(&circuit)?;
    let input = ArrayReg::zero_state(2);
    let target = circuit.forward(&[0.9, -0.4], &input)?;
    let input = gpu.upload(&Tensor::from_vec_col_major(vec![4], input.state)?, false)?;
    let target = gpu.upload(&Tensor::from_vec_col_major(vec![4], target.state)?, false)?;
    let step = gpu.upload(&Tensor::from_vec_col_major(vec![2], vec![0.1; 2])?, false)?;
    let mut parameters =
        gpu.upload(&Tensor::from_vec_col_major(vec![2], vec![0.2, 0.1])?, false)?;
    for _ in 0..20 {
        // A new leaf starts an independent first-order AD graph, retaining the
        // updated values on the device. No transfer is hidden in to_tensor().
        let p = EagerTensor::requires_grad_in(parameters.to_tensor()?, gpu.runtime().clone())?;
        let delta = prepared.apply(&p, &input)?.sub(&target)?;
        // Re(conj(delta)*delta) is smooth at zero and supported by CUDA AD.
        let loss = delta
            .conj()?
            .mul(&delta)?
            .cast(DType::F64)?
            .reduce_sum(None)?;
        let gradient = gpu.runtime().grad(&loss, &p)?;
        let updated = p.sub(&gradient.mul(&step)?)?;
        parameters = EagerTensor::from_tensor_in(updated.to_tensor()?, gpu.runtime().clone())?;
    }
    let delta = prepared.apply(&parameters, &input)?.sub(&target)?;
    let loss = delta
        .conj()?
        .mul(&delta)?
        .cast(DType::F64)?
        .reduce_sum(None)?;
    println!(
        "{}: squared state error {:.6e}",
        gpu.device_name(),
        gpu.download(&loss)?.as_slice::<f64>()?[0]
    );
    println!(
        "[Ry, Rz] = {:?}",
        gpu.download(&parameters)?.as_slice::<f64>()?
    );
    Ok(())
}
