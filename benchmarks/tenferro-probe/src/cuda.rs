//! The public CUDA adapter measured with explicit transfer boundaries.
use crate::{Case, Result, circuit_ad, convert};
use num_complex::Complex64 as C;
use tenferro_ad::EagerTensor;
use tenferro_tensor::{DType, Tensor};
use yao_rs::Register;
use yao_rs::cuda::{CudaCircuit, CudaSimulator, PreparedCudaContraction};
use yao_rs::differentiable::DifferentiableCircuit;

enum Operation {
    Circuit(CudaCircuit),
    Gradient(CudaCircuit),
    Density(PreparedCudaContraction),
}

pub struct PreparedCase {
    operation: Operation,
    host: Vec<Tensor>,
    resident: Vec<EagerTensor>,
    // Density tensor logical axes use row-major ArrayReg order; output storage
    // itself is column-major. Only explicit host collection changes the layout.
    density: bool,
}
impl PreparedCase {
    pub fn new(case: &Case, gpu: &CudaSimulator) -> Result<Self> {
        let c = case.circuit()?;
        let x = case.initial(c.nbits);
        let (operation, host) = if case.mode == "density" {
            let tn = yao_rs::circuit_to_einsum_dm(&c);
            (
                Operation::Density(PreparedCudaContraction::new(&tn.code, &tn.size_dict, None)?),
                convert(&tn.tensors)?,
            )
        } else {
            let c = DifferentiableCircuit::from_circuit(c)?;
            let host = vec![
                Tensor::from_vec_col_major(
                    vec![c.num_parameters()],
                    c.template().parameters().to_vec(),
                )?,
                Tensor::from_vec_col_major(vec![x.state.len()], x.state)?,
            ];
            let plan = gpu.prepare(&c)?;
            if case.mode == "custom_gradient" {
                let mut host = host;
                host.push(Tensor::from_vec_col_major(
                    vec![c.state_len()],
                    circuit_ad::target(c.template().circuit().nbits),
                )?);
                (Operation::Gradient(plan), host)
            } else if case.mode == "state" {
                (Operation::Circuit(plan), host)
            } else {
                return Err("unsupported CUDA benchmark mode".into());
            }
        };
        let resident = host
            .iter()
            .map(|x| gpu.upload(x, false).map_err(Into::into))
            .collect::<Result<_>>()?;
        Ok(Self {
            operation,
            host,
            resident,
            density: case.mode == "density",
        })
    }

    pub fn input_bytes(&self) -> usize {
        self.host
            .iter()
            .map(|x| {
                x.shape().iter().product::<usize>() * if x.dtype() == DType::F64 { 8 } else { 16 }
            })
            .sum()
    }

    pub fn resident(&self, gpu: &CudaSimulator) -> Result<Vec<EagerTensor>> {
        self.execute(gpu, &self.resident)
    }

    pub fn end_to_end(&self, gpu: &CudaSimulator) -> Result<Vec<C>> {
        let inputs = self
            .host
            .iter()
            .map(|x| gpu.upload(x, false).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        self.download(gpu, &self.execute(gpu, &inputs)?)
    }

    fn execute(&self, gpu: &CudaSimulator, inputs: &[EagerTensor]) -> Result<Vec<EagerTensor>> {
        match &self.operation {
            Operation::Density(plan) => Ok(vec![gpu.contract(plan, inputs)?]),
            Operation::Circuit(plan) => Ok(vec![plan.apply(&inputs[0], &inputs[1])?]),
            Operation::Gradient(plan) => {
                // Fresh leaves and their device copies are included in resident
                // execution; uploading host values is not. Do not reuse tapes.
                let p =
                    EagerTensor::requires_grad_in(inputs[0].to_tensor()?, gpu.runtime().clone())?;
                let x =
                    EagerTensor::requires_grad_in(inputs[1].to_tensor()?, gpu.runtime().clone())?;
                let delta = plan.apply(&p, &x)?.sub(&inputs[2])?;
                let loss = delta
                    .conj()?
                    .mul(&delta)?
                    .cast(DType::F64)?
                    .reduce_sum(None)?;
                // backward() in tenferro 0.4.0 computes a separate VJP for
                // every retained tracked intermediate. Request only the two
                // outputs this workload needs, using the public targeted API.
                let dp = gpu.runtime().grad(&loss, &p)?;
                let dx = gpu.runtime().grad(&loss, &x)?;
                Ok(vec![loss, dp, dx])
            }
        }
    }

    pub fn download(&self, gpu: &CudaSimulator, output: &[EagerTensor]) -> Result<Vec<C>> {
        let mut values = Vec::new();
        for tensor in output {
            if tensor
                .value()?
                .as_tensor_view()
                .placement()
                .device
                .is_none()
            {
                return Err("CUDA result lost device placement".into());
            }
            let host = gpu.download(tensor)?;
            if self.density {
                values.extend(crate::output_array(&host)?.iter().copied());
            } else if host.dtype() == DType::F64 {
                values.extend(host.as_slice::<f64>()?.iter().copied().map(C::from));
            } else {
                values.extend_from_slice(host.as_slice::<C>()?);
            }
        }
        Ok(values)
    }
}

pub fn reference(case: &Case) -> Result<Vec<C>> {
    let c = case.circuit()?;
    let x = case.initial(c.nbits);
    Ok(match case.mode.as_str() {
        "state" => yao_rs::apply(&c, &x).state,
        "density" => {
            let mut dm = yao_rs::DensityMatrix::from_reg(&x);
            dm.apply(&c);
            dm.state
        }
        "custom_gradient" => circuit_ad::native(
            &DifferentiableCircuit::from_circuit(c)?,
            &x,
            &circuit_ad::target(x.nqubits()),
        )?,
        _ => return Err("unsupported CUDA reference mode".into()),
    })
}

pub fn check(got: &[C], expected: &[C]) -> Result<f64> {
    if got.len() != expected.len() {
        return Err("CUDA output length mismatch".into());
    }
    let error = got
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    if !error.is_finite()
        || got.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
        || error > 1e-9
    {
        return Err(format!("CUDA output error {error}").into());
    }
    Ok(error)
}
