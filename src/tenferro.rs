//! Optional complex128 tensor contraction on tenferro's faer CPU provider.
//!
//! Plans own compiled graphs, not tensor values. A caller owns the CPU context
//! and controls its thread count and lifetime. Inputs may change values and
//! strides between executions, but must retain their prepared shapes. There is
//! no process-global context or implicit GPU dispatch.
//!
//! ```
//! use yao_rs::{Circuit, Gate, put, circuit_to_einsum_with_boundary};
//! use yao_rs::tenferro::CpuContractor;
//! let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::X)]).unwrap();
//! let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
//! let cpu = CpuContractor::new(1)?;
//! let plan = cpu.prepare(&tn.code, &tn.size_dict, None)?;
//! let state = cpu.execute(&plan, &tn.tensors)?;
//! assert_eq!(state[[1]].re, 1.0);
//! # Ok::<(), String>(())
//! ```

use ndarray::{ArrayD, Dimension, IxDyn, ShapeBuilder};
use num_complex::Complex64;
use omeco::{EinCode, GreedyMethod, Label, NestedEinsum};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EinsumOptimize, EinsumSubscripts, TraceContextEinsumExt};
use tenferro_runtime::{
    CompiledGraph, DType, GraphCompiler, Runtime, TraceContext, TraceValue,
    program::ProgramInputSpec,
    runtime::{OutputRef, ScopedExecutionOutcome, ScopedReadInputs},
};
use tenferro_tensor::{TensorView, TypedTensorView};

/// A CPU execution context. `threads` is explicit to avoid nested thread pools.
/// Share/reuse the context around repeated work; do not enter a tenferro backend
/// session around these calls. No external BLAS provider is enabled here.
pub struct CpuContractor {
    runtime: Runtime,
    threads: usize,
}

/// A fixed-shape contraction compiled independently of input values.
///
/// Dropping this value releases its compiled graph. Retain only the plans your
/// workload needs; this adapter maintains no global plan cache.
pub struct PreparedContraction {
    program: Option<CompiledGraph>,
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

impl CpuContractor {
    /// Create a context with a positive number of faer CPU threads.
    pub fn new(threads: usize) -> Result<Self, String> {
        if threads == 0 {
            return Err("CPU thread count must be positive".into());
        }
        let backend = CpuBackend::with_threads(threads).map_err(error)?;
        let mut builder = Runtime::builder();
        builder
            .register_engine(tenferro_cpu::runtime_engine_registration(&backend).map_err(error)?)
            .map_err(error)?;
        builder
            .install_extension_module(
                tenferro_einsum::extension_module::<CpuBackend>(
                    tenferro_cpu::runtime_engine_id().map_err(error)?,
                )
                .map_err(error)?,
            )
            .map_err(error)?;
        Ok(Self {
            runtime: builder.build().map_err(error)?,
            threads,
        })
    }

    /// Configured number of CPU threads.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Prepare an expression using an explicit tree, or deterministic omeco
    /// greedy planning when `tree` is `None`. Each supplied node is compiled
    /// separately, retaining its intermediate output axes and grouping. N-ary
    /// nodes contract left-to-right within that node.
    ///
    /// Dimensions must be positive and complex128 byte counts fit `isize`.
    /// No tensor values are read or allocated during validation/planning.
    pub fn prepare<L: Label>(
        &self,
        code: &EinCode<L>,
        sizes: &HashMap<L, usize>,
        tree: Option<&NestedEinsum<L>>,
    ) -> Result<PreparedContraction, String> {
        let labels: HashSet<_> = code.ixs.iter().flatten().collect();
        let mut outputs = HashSet::new();
        for label in &code.iy {
            if !labels.contains(label) || !outputs.insert(label) {
                return Err(format!("Invalid output label {label:?}"));
            }
        }
        let input_shapes = code
            .ixs
            .iter()
            .map(|xs| shape(xs, sizes))
            .collect::<Result<Vec<_>, _>>()?;
        let output_shape = shape(&code.iy, sizes)?;
        if let Some(tree) = tree {
            crate::contraction_plan::validate_tree(tree, code)?;
        }
        if code.ixs.is_empty() {
            return Ok(PreparedContraction {
                program: None,
                input_shapes,
                output_shape,
            });
        }
        let generated;
        let tree = match tree {
            Some(tree) => tree,
            None => {
                generated =
                    crate::contraction_plan::optimize_code(code, sizes, &GreedyMethod::default())
                        .ok_or_else(|| "Greedy contraction planning failed".to_string())?;
                &generated
            }
        };
        crate::contraction_plan::validate_tree(tree, code)?;
        let mut label_map = HashMap::new();
        for label in code.ixs.iter().flatten() {
            let next = u32::try_from(label_map.len()).map_err(error)?;
            label_map.entry(label.clone()).or_insert(next);
        }
        let mut trace = TraceContext::new();
        let inputs = input_shapes
            .iter()
            .map(|s| {
                trace
                    .input(ProgramInputSpec::new(
                        DType::C64,
                        s.iter().map(|&d| d.into()).collect::<Vec<_>>(),
                    ))
                    .map_err(error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = trace_tree(tree, &inputs, &label_map, sizes, &mut trace)?;
        let graph = trace.finish(&[output]).map_err(error)?;
        let program = GraphCompiler::new()
            .compile_traced_graph(&graph)
            .map_err(error)?;
        Ok(PreparedContraction {
            program: Some(program),
            input_shapes,
            output_shape,
        })
    }

    /// Execute a prepared graph with new values. Compatible contiguous inputs
    /// are borrowed with their actual strides; sliced/reversed layouts are
    /// copied once at this boundary. Tenferro may still pack operands internally.
    /// The returned ndarray owns its data, ordered by the expression's output
    /// axes (qubit 0 remains the most significant site).
    pub fn execute(
        &self,
        plan: &PreparedContraction,
        tensors: &[ArrayD<Complex64>],
    ) -> Result<ArrayD<Complex64>, String> {
        if tensors.len() != plan.input_shapes.len() {
            return Err("Tensor count does not match prepared plan".into());
        }
        for (i, (tensor, shape)) in tensors.iter().zip(&plan.input_shapes).enumerate() {
            if tensor.shape() != shape {
                return Err(format!("Tensor {i} shape does not match prepared plan"));
            }
        }
        let Some(program) = &plan.program else {
            return Ok(ArrayD::from_elem(IxDyn(&[]), Complex64::new(1., 0.)));
        };
        let storage = tensors.iter().map(InputStorage::new).collect::<Vec<_>>();
        let views = storage
            .iter()
            .zip(tensors)
            .map(|(s, t)| {
                TypedTensorView::from_slice(t.shape(), &s.strides, 0, &s.data)
                    .map(TensorView::C64)
                    .map_err(error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outcome = self
            .runtime
            .execute_scoped_read_only(program, ScopedReadInputs::new(views))
            .map_err(error)?;
        let bundle = match outcome {
            ScopedExecutionOutcome::Completed(bundle) => bundle,
            ScopedExecutionOutcome::RetiredFailed { error: e, .. } => return Err(error(e)),
        };
        match bundle.output(0).map_err(error)? {
            OutputRef::Tensor(TensorView::C64(view)) => {
                if view.shape() != plan.output_shape {
                    return Err("Backend returned an unexpected output shape".into());
                }
                if let Ok(data) = view.as_slice() {
                    return ArrayD::from_shape_vec(IxDyn(view.shape()).f(), data.to_vec())
                        .map_err(error);
                }
                let data = ndarray::indices(IxDyn(view.shape()))
                    .into_iter()
                    .map(|i| {
                        view.get(i.slice())
                            .copied()
                            .ok_or_else(|| "Backend output is not host-accessible".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                ArrayD::from_shape_vec(IxDyn(view.shape()), data).map_err(error)
            }
            _ => Err("Backend returned an unexpected output type".into()),
        }
    }
}

fn shape<L: Label>(labels: &[L], sizes: &HashMap<L, usize>) -> Result<Vec<usize>, String> {
    let shape = labels
        .iter()
        .map(|label| {
            sizes
                .get(label)
                .copied()
                .filter(|&d| d > 0)
                .ok_or_else(|| format!("Missing or zero dimension for label {label:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bytes = shape
        .iter()
        .try_fold(size_of::<Complex64>(), |n, &d| n.checked_mul(d));
    if bytes.is_none_or(|n| n > isize::MAX as usize) {
        return Err("Tensor shape exceeds addressable complex128 storage".into());
    }
    Ok(shape)
}

fn trace_tree<L: Label>(
    tree: &NestedEinsum<L>,
    inputs: &[TraceValue],
    labels: &HashMap<L, u32>,
    sizes: &HashMap<L, usize>,
    trace: &mut TraceContext,
) -> Result<TraceValue, String> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => Ok(inputs[*tensor_index]),
        NestedEinsum::Node { args, eins } => {
            shape(&eins.iy, sizes)?;
            let children = args
                .iter()
                .map(|arg| trace_tree(arg, inputs, labels, sizes, trace))
                .collect::<Result<Vec<_>, _>>()?;
            let remap = |xs: &[L]| xs.iter().map(|x| labels[x]).collect::<Vec<_>>();
            let subs = EinsumSubscripts {
                inputs: eins.ixs.iter().map(|xs| remap(xs)).collect(),
                output: remap(&eins.iy),
            };
            trace
                .einsum_subscripts_with(&children, &subs, EinsumOptimize::False)
                .map_err(error)
        }
    }
}

struct InputStorage<'a> {
    data: Cow<'a, [Complex64]>,
    strides: Vec<isize>,
}
impl<'a> InputStorage<'a> {
    fn new(tensor: &'a ArrayD<Complex64>) -> Self {
        if let Some(data) = tensor
            .as_slice_memory_order()
            .filter(|_| tensor.strides().iter().all(|&s| s >= 0))
        {
            return Self {
                data: Cow::Borrowed(data),
                strides: tensor.strides().to_vec(),
            };
        }
        let mut stride = 1;
        let strides = tensor
            .shape()
            .iter()
            .map(|&d| {
                let s = stride;
                stride *= d as isize;
                s
            })
            .collect();
        Self {
            data: Cow::Owned(tensor.t().iter().copied().collect()),
            strides,
        }
    }
}

#[cfg(test)]
#[path = "unit_tests/tenferro.rs"]
mod tests;
