//! Prototype of a specialized CPU circuit operation inside a tenferro graph.
use crate::Result;
use num_complex::Complex64 as C;
use std::{any::Any, sync::Arc};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::{
    ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionExecutionContext, ExtensionOp,
    ExtensionShapeContext, SymDim, apply, define_extension_runtime,
};
use tenferro_runtime::{DType, GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{Tensor, TensorBackend, TensorRead};
use yao_rs::{ArrayReg, Circuit, Gate, apply_inplace, put};

#[derive(Clone, Debug)]
struct XFirst;
const FAMILY: &str = "yao.probe.x_first.v1";
impl ExtensionOp for XFirst {
    fn family_id(&self) -> &'static str {
        "yao.probe.x_first.v1"
    }
    fn payload_hash(&self, h: &mut dyn std::hash::Hasher) {
        h.write_u8(0);
    }
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        1
    }
    fn output_count(&self) -> usize {
        1
    }
    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }
    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }
    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}
fn execute<B: TensorBackend + 'static>(
    _: &XFirst,
    inputs: &[TensorRead<'_>],
    _: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // The prototype graph only accepts C64[2, batch]. Host access rejects devices.
    let data = inputs[0].as_slice::<C>()?;
    let batch = inputs[0].shape()[1];
    let n = 1 + batch.ilog2() as usize;
    // Column-major [2,batch] makes the first axis the least-significant bit.
    let mut reg = ArrayReg::from_vec(n, data.to_vec());
    apply_inplace(
        &Circuit::qubits(n, vec![put(vec![n - 1], Gate::X)]).unwrap(),
        &mut reg,
    );
    Ok(vec![Tensor::from_vec_col_major(vec![2, batch], reg.state)?])
}
define_extension_runtime! {
    runtime = XRuntime,
    family_id = FAMILY,
    op_type = XFirst,
    execute_reads = execute,
}

/// Prepared custom operation; validates shape before the kernel can be invoked.
pub fn prepared_x(
    batch: usize,
    threads: usize,
) -> Result<(Runtime, tenferro_runtime::CompiledGraph)> {
    if !batch.is_power_of_two() {
        return Err("batch must be a nonzero power of two".into());
    }
    let input = TracedTensor::input_concrete_shape(DType::C64, &[2, batch])?;
    let output = apply(Arc::new(XFirst), &[&input])?.remove(0);
    let program = GraphCompiler::new()
        .compile_with_input_specs(&output, &[(&input, DType::C64, &[2, batch])])?;
    let backend = CpuBackend::with_threads(threads)?;
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    builder.install_extension_module(extension_module::<CpuBackend>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    Ok((builder.build()?, program))
}

#[test]
fn registered_x_matches_tensor_composition() -> Result<()> {
    use tenferro_einsum::TensorEinsumExt;
    use tenferro_tensor::BackendSessionHost;
    let (runtime, program) = prepared_x(4, 1)?;
    let a = Tensor::from_vec_col_major(
        vec![2, 4],
        (0..8).map(|i| C::new(i as f64, 0.2 * i as f64)).collect(),
    )?;
    let matrix = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            C::new(0., 0.),
            C::new(1., 0.),
            C::new(1., 0.),
            C::new(0., 0.),
        ],
    )?;
    let composed = CpuBackend::with_threads(1)?
        .with_backend_session(|s| [&matrix, &a].einsum("ij,jk->ik", s))?;
    let custom = runtime.run_compiled(&program, &[&a])?.remove(0);
    assert_eq!(custom.as_slice::<C>()?, composed.as_slice::<C>()?);
    let wrong = Tensor::from_vec_col_major(vec![2, 3], vec![C::new(0., 0.); 6])?;
    assert!(runtime.run_compiled(&program, &[&wrong]).is_err());
    assert!(prepared_x(3, 1).is_err());
    Ok(())
}

#[test]
fn missing_ad_rule_is_reported() -> Result<()> {
    use tenferro_ad::TracedTensorAdExt;
    let input = TracedTensor::input_concrete_shape(DType::C64, &[2, 4])?;
    let output = apply(Arc::new(XFirst), &[&input])?.remove(0);
    assert!(output.grad(&input).is_err());
    Ok(())
}
