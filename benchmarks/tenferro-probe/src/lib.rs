//! Published-tenferro feasibility adapter. Not a supported yao-rs backend yet.
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use num_complex::Complex64 as C;
use omeco::{EinCode, Label};
use std::collections::HashMap;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumSubscripts};
use tenferro_tensor::{BackendSessionHost, Tensor};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub mod circuit_ad;
pub mod extension;

/// Materialize logical ndarray axes in Fortran order, including strided arrays.
pub fn convert(arrays: &[ArrayD<C>]) -> Result<Vec<Tensor>> {
    arrays
        .iter()
        .map(|a| {
            Ok(Tensor::from_vec_col_major(
                a.shape().to_vec(),
                a.t().iter().copied().collect(),
            )?)
        })
        .collect()
}

/// Remap arbitrary labels without casting negative density-matrix labels.
pub fn subscripts<L: Label>(code: &EinCode<L>) -> Result<EinsumSubscripts> {
    let mut labels = HashMap::new();
    let mut remap = |xs: &[L]| -> Result<Vec<u32>> {
        xs.iter()
            .map(|x| {
                let next = u32::try_from(labels.len())?;
                Ok(*labels.entry(x.clone()).or_insert(next))
            })
            .collect()
    };
    Ok(EinsumSubscripts {
        inputs: code.ixs.iter().map(|xs| remap(xs)).collect::<Result<_>>()?,
        output: remap(&code.iy)?,
    })
}

pub fn prepare(tensors: &[Tensor], code: &EinsumSubscripts) -> Result<ConcreteEinsumPlan> {
    Ok(ConcreteEinsumPlan::prepare_subscripts(
        tensors.iter().collect::<Vec<_>>(),
        code,
    )?)
}

pub fn execute(
    plan: &ConcreteEinsumPlan,
    tensors: &[Tensor],
    backend: &mut CpuBackend,
) -> Result<Tensor> {
    Ok(backend.with_backend_session(|session| {
        plan.execute(tensors.iter().collect::<Vec<_>>(), session)
    })?)
}

pub fn output_array(output: &Tensor) -> Result<ArrayD<C>> {
    Ok(ArrayD::from_shape_vec(
        IxDyn(output.shape()).f(),
        output.as_slice::<C>()?.to_vec(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_ad::{EagerRuntime, EagerTensor};
    use tenferro_einsum::{EinsumOptimize, TraceContextEinsumExt};
    use tenferro_runtime::{GraphCompiler, Runtime, TraceContext, program::ProgramInputSpec};
    use yao_rs::einsum::{
        circuit_to_einsum_dm, circuit_to_einsum_with_boundary, circuit_to_overlap,
    };
    use yao_rs::{
        ArrayReg, Circuit, DensityMatrix, Gate, NoiseChannel, Register, apply, channel, control,
        put,
    };

    fn close(got: &[C], want: &[C]) {
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).norm() < 1e-10, "{a} != {b}");
        }
    }
    fn contract<L: Label>(arrays: &[ArrayD<C>], code: &EinCode<L>) -> Result<ArrayD<C>> {
        let tensors = convert(arrays)?;
        let plan = prepare(&tensors, &subscripts(code)?)?;
        output_array(&execute(
            &plan,
            &tensors,
            &mut CpuBackend::with_threads(1)?,
        )?)
    }

    #[test]
    fn asymmetric_complex_state_and_scalar() -> Result<()> {
        let circuit = Circuit::qubits(
            3,
            vec![
                put(vec![0], Gate::Ry(0.31)),
                put(vec![2], Gate::Rx(-0.72)),
                control(vec![0], vec![2], Gate::Y),
                put(vec![1], Gate::Phase(0.8)),
                put(vec![2, 0], Gate::FSim(0.2, 0.4)),
            ],
        )?;
        let state = apply(&circuit, &ArrayReg::zero_state(3));
        let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
        close(
            &contract(&tn.tensors, &tn.code)?
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            state.state_vec(),
        );
        let tn = circuit_to_overlap(&circuit);
        let scalar = contract(&tn.tensors, &tn.code)?;
        assert_eq!(scalar.ndim(), 0);
        close(
            &scalar.iter().copied().collect::<Vec<_>>(),
            &state.state_vec()[..1],
        );
        Ok(())
    }

    #[test]
    fn density_negative_labels_and_channels() -> Result<()> {
        let circuit = Circuit::qubits(
            2,
            vec![
                put(vec![0], Gate::Ry(0.7)),
                put(vec![1], Gate::Rx(-0.3)),
                control(vec![0], vec![1], Gate::X),
                channel(
                    vec![1],
                    NoiseChannel::AmplitudeDamping {
                        gamma: 0.23,
                        excited_population: 0.0,
                    },
                ),
            ],
        )?;
        let mut dm = DensityMatrix::zero_state(2);
        dm.apply(&circuit);
        let tn = circuit_to_einsum_dm(&circuit);
        let result = contract(&tn.tensors, &tn.code)?;
        close(&result.iter().copied().collect::<Vec<_>>(), &dm.state);
        Ok(())
    }

    #[test]
    fn qudit_and_identity_export() -> Result<()> {
        for circuit in [
            Circuit::new(vec![3], vec![])?,
            Circuit::new(
                vec![3],
                vec![put(
                    vec![0],
                    Gate::Custom {
                        matrix: ndarray::array![
                            [C::new(0., 0.), C::new(0., 0.), C::new(1., 0.)],
                            [C::new(0., 1.), C::new(0., 0.), C::new(0., 0.)],
                            [C::new(0., 0.), C::new(1., 0.), C::new(0., 0.)]
                        ],
                        is_diagonal: false,
                        label: "Q".into(),
                    },
                )],
            )?,
        ] {
            let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
            let want = yao_rs::contractor::contract(&tn);
            let got = contract(&tn.tensors, &tn.code)?;
            close(
                &got.iter().copied().collect::<Vec<_>>(),
                &want.iter().copied().collect::<Vec<_>>(),
            );
        }
        Ok(())
    }

    #[test]
    fn layout_reordering_and_reuse() -> Result<()> {
        let a = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            (0..6).map(|x| C::new(x as f64, 1. - x as f64)).collect(),
        )?
        .reversed_axes();
        let code = EinCode::new(vec![vec![u64::MAX, 7]], vec![7, u64::MAX]);
        let mut tensors = convert(std::slice::from_ref(&a))?;
        let plan = prepare(&tensors, &subscripts(&code)?)?;
        let mut backend = CpuBackend::with_threads(1)?;
        for factor in [1., 2.] {
            tensors = convert(&[a.mapv(|x| x * factor)])?;
            let got = output_array(&execute(&plan, &tensors, &mut backend)?)?;
            close(
                &got.iter().copied().collect::<Vec<_>>(),
                &a.t().iter().map(|x| x * factor).collect::<Vec<_>>(),
            );
        }
        let wrong = vec![Tensor::from_vec_col_major(
            vec![6],
            vec![C::new(0., 0.); 6],
        )?];
        assert!(execute(&plan, &wrong, &mut backend).is_err());
        assert!(execute(&plan, &[], &mut backend).is_err());
        Ok(())
    }

    #[test]
    fn complex_loss_and_nonreal_vjp() -> Result<()> {
        let context = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1)?)?;
        let values = vec![C::new(1., 2.), C::new(-0.3, 0.7)];
        let x = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![2], values.clone())?,
            context.clone(),
        )?;
        let magnitude = x.abs()?;
        let loss = magnitude.mul(&magnitude)?.reduce_sum(Some(&[0]))?;
        let grad = context.grad(&loss, &x)?;
        close(
            grad.value()?.as_slice::<C>()?,
            &values.iter().map(|x| 2. * x).collect::<Vec<_>>(),
        );
        let seeds = vec![C::new(2., -1.), C::new(-1., 3.)];
        let seed = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![2], seeds.clone())?,
            context.clone(),
        )?;
        let grad = context.vjp(&x.mul(&x)?, &x, &seed)?;
        close(
            grad.value()?.as_slice::<C>()?,
            &values
                .iter()
                .zip(seeds)
                .map(|(x, s)| s * (2. * x).conj())
                .collect::<Vec<_>>(),
        );
        Ok(())
    }

    #[test]
    fn explicit_path_and_compiled_input_reuse() -> Result<()> {
        let mut trace = TraceContext::new();
        let inputs = (0..3)
            .map(|_| {
                trace.input(ProgramInputSpec::new(
                    tenferro_runtime::DType::C64,
                    [2.into(), 2.into()],
                ))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let out = trace.einsum_with(
            &inputs,
            "ij,jk,kl->il",
            EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
        )?;
        let program = GraphCompiler::new().compile_traced_graph(&trace.finish(&[out])?)?;
        let backend = CpuBackend::with_threads(1)?;
        let mut builder = Runtime::builder();
        builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
        builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
            tenferro_cpu::runtime_engine_id()?,
        )?)?;
        let runtime = builder.build()?;
        for factor in [1., 2.] {
            let a = Tensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    C::new(factor, 0.),
                    C::new(0., 1.),
                    C::new(0., 0.),
                    C::new(1., 0.),
                ],
            )?;
            let b = Tensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    C::new(1., 0.),
                    C::new(0., 0.),
                    C::new(0., 0.),
                    C::new(1., 0.),
                ],
            )?;
            let got = runtime.run_compiled(&program, &[&a, &b, &b])?.remove(0);
            close(got.as_slice::<C>()?, a.as_slice::<C>()?);
        }
        Ok(())
    }
}

/// Shared workload identity; the serialized circuit is also consumed by Julia.
#[derive(serde::Deserialize)]
pub struct Case {
    #[serde(default)]
    pub operator: Option<yao_rs::OperatorPolynomial>,
    pub id: String,
    pub mode: String,
    pub tensor: bool,
    pub initial: String,
    pub circuit: serde_json::Value,
}
impl Case {
    pub fn initial(&self, n: usize) -> yao_rs::ArrayReg {
        match self.initial.as_str() {
            "zero" => yao_rs::ArrayReg::zero_state(n),
            "deterministic" => yao_rs::ArrayReg::deterministic_state(n),
            _ => panic!("unknown initial state"),
        }
    }
    pub fn circuit(&self) -> Result<yao_rs::Circuit> {
        Ok(yao_rs::json::circuit_from_json(&self.circuit.to_string())?)
    }
}
pub fn cases() -> Result<Vec<Case>> {
    let path = std::env::var("YAO_BENCH_CASES")?;
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

pub mod tensor_memory;
