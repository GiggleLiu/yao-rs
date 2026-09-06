//! CPU circuit primitives and first-order AD rules for tenferro custom losses.
//!
//! Real parameters and complex state amplitudes are graph inputs. The reversible
//! VJP retains parameters and the final state, not every intermediate state.
//! Derivatives of these derivative primitives (higher-order AD) are unsupported.

use crate::{ArrayReg, differentiable::DifferentiableCircuit};
use ::tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRuleSet,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
    SemanticPrimalVjpRequest, SemanticPrimalVjpRule,
};
pub use ::tenferro_ad::{AdContext, EagerRuntime, EagerTensor};
use num_complex::Complex64 as C;
use std::{any::Any, sync::Arc};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::{
    ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionExecutionContext, ExtensionOp,
    ExtensionShapeContext, SymDim, apply, define_extension_runtime,
};
use tenferro_runtime::program::SemanticProgramBuilder;
pub use tenferro_runtime::{DType, Runtime, TracedTensor};
pub use tenferro_tensor::Tensor;
use tenferro_tensor::{TensorBackend, TensorRead};

const FAMILY: &str = "yao.unitary_circuit.v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Forward,
    Vjp,
    JvpParameters,
    JvpInput,
    JvpBoth,
}
#[derive(Debug, Clone)]
struct CircuitOp {
    circuit: Arc<DifferentiableCircuit>,
    kind: Kind,
}
impl CircuitOp {
    fn input_meta(&self, index: usize) -> (DType, usize) {
        if index == 0 || (index == 2 && matches!(self.kind, Kind::JvpParameters | Kind::JvpBoth)) {
            (DType::F64, self.circuit.num_parameters())
        } else {
            (DType::C64, self.circuit.state_len())
        }
    }
}
fn invalid(message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(FAMILY, "circuit", message)
}
impl ExtensionOp for CircuitOp {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, h: &mut dyn std::hash::Hasher) {
        h.write_usize(Arc::as_ptr(&self.circuit) as usize);
        h.write_u8(self.kind as u8);
    }
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| self.kind == o.kind && Arc::ptr_eq(&self.circuit, &o.circuit))
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        match self.kind {
            Kind::Forward => 2,
            Kind::JvpBoth => 4,
            _ => 3,
        }
    }
    fn output_count(&self) -> usize {
        if self.kind == Kind::Vjp { 2 } else { 1 }
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
        let p = vec![SymDim::from(self.circuit.num_parameters())];
        let s = vec![SymDim::from(self.circuit.state_len())];
        for i in 0..self.input_count() {
            let (dtype, len) = self.input_meta(i);
            if ctx.input_dtype(i)? != dtype || ctx.input_shape(i)?.len() != 1 {
                return Err(invalid(
                    "Expected F64[parameters] and C64[state] inputs with fixed shapes",
                ));
            }
            ctx.require_equal(ctx.input_axis(i, 0)?, SymDim::from(len))?;
        }
        Ok(if self.kind == Kind::Vjp {
            vec![(DType::F64, p), (DType::C64, s)]
        } else {
            vec![(DType::C64, s)]
        })
    }
}
fn execute<B: TensorBackend + 'static>(
    op: &CircuitOp,
    inputs: &[TensorRead<'_>],
    _: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != op.input_count() {
        return Err(invalid("Circuit input count mismatch"));
    }
    for (i, input) in inputs.iter().enumerate() {
        let (dtype, len) = op.input_meta(i);
        if input.dtype() != dtype || input.shape() != [len] {
            return Err(invalid("Circuit input dtype or shape mismatch"));
        }
    }
    let p = inputs[0].as_slice::<f64>()?;
    let n = op.circuit.template().circuit().nbits;
    let state = ArrayReg::from_vec(n, inputs[1].as_slice::<C>()?.to_vec());
    match op.kind {
        Kind::Forward => {
            let out = op.circuit.forward(p, &state).map_err(invalid)?;
            Ok(vec![Tensor::from_vec_col_major(
                vec![out.state.len()],
                out.state,
            )?])
        }
        Kind::Vjp => {
            let bar = ArrayReg::from_vec(n, inputs[2].as_slice::<C>()?.to_vec());
            let out = op
                .circuit
                .vjp_from_output(p, &state, &bar)
                .map_err(invalid)?;
            Ok(vec![
                Tensor::from_vec_col_major(vec![out.parameters.len()], out.parameters)?,
                Tensor::from_vec_col_major(vec![out.input.state.len()], out.input.state)?,
            ])
        }
        kind => {
            let dp = if kind == Kind::JvpInput {
                vec![0.; p.len()]
            } else {
                inputs[2].as_slice::<f64>()?.to_vec()
            };
            let dx = if kind == Kind::JvpParameters {
                vec![C::new(0., 0.); state.state.len()]
            } else {
                inputs[op.input_count() - 1].as_slice::<C>()?.to_vec()
            };
            let (_, tangent) = op
                .circuit
                .jvp(p, &state, &dp, &ArrayReg::from_vec(n, dx))
                .map_err(invalid)?;
            Ok(vec![Tensor::from_vec_col_major(
                vec![tangent.state.len()],
                tangent.state,
            )?])
        }
    }
}
define_extension_runtime! {
    runtime = CircuitRuntime,
    family_id = FAMILY,
    op_type = CircuitOp,
    execute_reads = execute,
}

#[derive(Debug)]
struct CircuitRule;
impl SemanticLinearizeRule for CircuitRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn linearize(
        &self,
        r: SemanticLinearizeRequest<'_>,
        b: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        let op = r
            .op()
            .as_any()
            .downcast_ref::<CircuitOp>()
            .expect("registered circuit family");
        if op.kind != Kind::Forward {
            return Err(SemanticAdError::Unsupported {
                family_id: FAMILY,
                role: SemanticAdRuleRole::Linearize,
                message: "Circuit AD supports first-order derivatives only".into(),
            });
        }
        let tangent = r.tangent_inputs();
        let kind = match (tangent[0].value(), tangent[1].value()) {
            (Some(_), Some(_)) => Kind::JvpBoth,
            (Some(_), None) => Kind::JvpParameters,
            (None, Some(_)) => Kind::JvpInput,
            (None, None) => return Ok(SemanticLinearizeResult::new([AdValue::Absent], [])),
        };
        let mut inputs = r.primal_inputs().to_vec();
        inputs.extend(tangent.iter().filter_map(|v| v.value()));
        let out = b.add_extension(
            Arc::new(CircuitOp {
                circuit: op.circuit.clone(),
                kind,
            }),
            &inputs,
        )?;
        Ok(SemanticLinearizeResult::new([AdValue::Value(out[0])], []))
    }
}
impl SemanticPrimalVjpRule for CircuitRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::input(0).with_output(0)
    }
    fn primal_vjp(
        &self,
        r: SemanticPrimalVjpRequest<'_>,
        b: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let op = r
            .op()
            .as_any()
            .downcast_ref::<CircuitOp>()
            .expect("registered circuit family");
        if op.kind != Kind::Forward {
            return Err(SemanticAdError::Unsupported {
                family_id: FAMILY,
                role: SemanticAdRuleRole::PrimalVjp,
                message: "Circuit AD supports first-order derivatives only".into(),
            });
        }
        let Some(seed) = r.cotangent_outputs()[0].value() else {
            return Ok(vec![AdValue::Absent; 2].into_boxed_slice());
        };
        let inputs = [r.primal_input_value(0)?, r.primal_output_value(0)?, seed];
        let outputs = b.add_extension(
            Arc::new(CircuitOp {
                circuit: op.circuit.clone(),
                kind: Kind::Vjp,
            }),
            &inputs,
        )?;
        Ok(outputs
            .into_iter()
            .zip(r.active_inputs())
            .map(|(v, &active)| {
                if active {
                    AdValue::Value(v)
                } else {
                    AdValue::Absent
                }
            })
            .collect())
    }
}

/// Add a unitary circuit to a tenferro graph. Parameters must be real F64 and
/// the input C64, both rank-one tensors with the prepared dimensions.
pub fn circuit_apply(
    circuit: Arc<DifferentiableCircuit>,
    parameters: &TracedTensor,
    input: &TracedTensor,
) -> tenferro_runtime::Result<TracedTensor> {
    Ok(apply(
        Arc::new(CircuitOp {
            circuit,
            kind: Kind::Forward,
        }),
        &[parameters, input],
    )?
    .remove(0))
}

/// A tenferro AD context with the circuit's first-order VJP registered.
pub fn ad_context() -> Result<AdContext, String> {
    let mut rules = SemanticExtensionRuleSet::new();
    rules
        .register_linearize(Arc::new(CircuitRule))
        .map_err(|e| e.to_string())?;
    rules
        .register_primal_vjp(Arc::new(CircuitRule))
        .map_err(|e| e.to_string())?;
    AdContext::builder()
        .with_semantic_extension_rules(rules)
        .map_err(|e| e.to_string())?
        .build()
        .map_err(|e| e.to_string())
}

/// Explicit CPU runtime for circuit and core tensor operations. Device tensors
/// cannot enter the native kernels through an implicit host fallback.
pub fn cpu_runtime(threads: usize) -> Result<Runtime, String> {
    if threads == 0 {
        return Err("CPU thread count must be positive".into());
    }
    let backend = CpuBackend::with_threads(threads).map_err(|e| e.to_string())?;
    let mut b = Runtime::builder();
    b.register_engine(
        tenferro_cpu::runtime_engine_registration(&backend).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    b.install_extension_module(
        extension_module::<CpuBackend>(
            tenferro_cpu::runtime_engine_id().map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    b.build().map_err(|e| e.to_string())
}

/// Create an eager CPU context with circuit execution and reverse rules.
/// Reuse this context, then drop per-evaluation tensors/gradients to release
/// their tape. `backward` follows tenferro's gradient-accumulation semantics.
pub fn eager_cpu_runtime(threads: usize) -> Result<Arc<::tenferro_ad::EagerRuntime>, String> {
    if threads == 0 {
        return Err("CPU thread count must be positive".into());
    }
    let backend = CpuBackend::with_threads(threads).map_err(|e| e.to_string())?;
    let ctx = ::tenferro_ad::EagerRuntime::with_cpu_backend_and_ad_context(backend, &ad_context()?)
        .map_err(|e| e.to_string())?;
    ctx.install_extension_module(
        extension_module::<CpuBackend>(
            tenferro_cpu::runtime_engine_id().map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(ctx)
}

/// Execute a tracked circuit in an eager context created by [`eager_cpu_runtime`].
/// Compose a real scalar loss from the returned state using tenferro operators,
/// then call `backward()` to obtain both real parameter and complex input gradients.
pub fn circuit_apply_eager(
    circuit: Arc<DifferentiableCircuit>,
    parameters: &::tenferro_ad::EagerTensor,
    input: &::tenferro_ad::EagerTensor,
) -> tenferro_runtime::Result<::tenferro_ad::EagerTensor> {
    Ok(::tenferro_ad::extension::apply_eager(
        Arc::new(CircuitOp {
            circuit,
            kind: Kind::Forward,
        }),
        &[parameters, input],
    )?
    .remove(0))
}

#[cfg(test)]
#[path = "unit_tests/tenferro_ad.rs"]
mod tests;
