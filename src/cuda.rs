//! Explicit complex128 CUDA circuit execution through tenferro.
//!
//! Upload states, real physical parameters and constants once; [`CudaCircuit::apply`]
//! returns a device tensor that composes with tenferro eager losses and AD.
//! Transfers occur only through [`CudaSimulator::upload`] and
//! [`CudaSimulator::download`]. Selecting CUDA never falls back to CPU execution.
//! Device operations inherit tenferro's operation coverage and numerical semantics.
//! Composed reverse mode retains intermediate states; it does not have the native
//! CPU reversible primitive's O(state size) memory guarantee.

use std::sync::Arc;

use ndarray::Array2;
use num_complex::Complex64 as C;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_gpu::cuda::{CudaBackend, CudaRuntime, cuda_devices, download_tensor, upload_tensor};
use tenferro_tensor::{DType, DeviceKind, DotGeneralConfig, GpuBackendKind, SliceConfig, Tensor};

use crate::{
    CircuitElement, Gate, differentiable::DifferentiableCircuit, parameters::ParameterBinding,
};

mod contraction;
pub use contraction::PreparedCudaContraction;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Caller-owned CUDA context. Device indices respect `CUDA_VISIBLE_DEVICES`.
/// Constructing this value requires an available CUDA device and runtime.
pub struct CudaSimulator {
    runtime: Arc<EagerRuntime>,
    cuda: CudaRuntime,
    device_name: String,
}

impl CudaSimulator {
    /// Select one enumerated CUDA device, without implicit CPU fallback.
    pub fn new(device: usize) -> Result<Self, String> {
        let devices = cuda_devices().map_err(error)?;
        let selected = devices
            .get(device)
            .ok_or("CUDA device index out of range")?;
        let backend = CudaBackend::new(selected.id()).map_err(error)?;
        let cuda = backend.runtime().clone();
        Ok(Self {
            runtime: EagerRuntime::with_cuda_backend(backend).map_err(error)?,
            cuda,
            device_name: selected.name().to_string(),
        })
    }

    /// Device model reported by CUDA.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Runtime for composing GPU losses and requesting first-order derivatives.
    /// Use fresh tracked input tensors for independent optimization iterations.
    pub fn runtime(&self) -> &Arc<EagerRuntime> {
        &self.runtime
    }

    /// Explicitly upload finite host complex128 or f64 data.
    /// Tracking enables derivatives with respect to this input.
    pub fn upload(&self, tensor: &Tensor, track: bool) -> Result<EagerTensor, String> {
        let finite = match tensor.dtype() {
            DType::C64 => tensor
                .as_slice::<C>()
                .map_err(error)?
                .iter()
                .all(|x| x.re.is_finite() && x.im.is_finite()),
            DType::F64 => tensor
                .as_slice::<f64>()
                .map_err(error)?
                .iter()
                .all(|x| x.is_finite()),
            _ => return Err("CUDA simulation uploads require complex128 or f64".into()),
        };
        if !finite {
            return Err("CUDA simulation inputs must be finite".into());
        }
        let tensor = upload_tensor(&self.cuda, tensor).map_err(error)?;
        if track {
            EagerTensor::requires_grad_in(tensor, self.runtime.clone()).map_err(error)
        } else {
            EagerTensor::from_tensor_in(tensor, self.runtime.clone()).map_err(error)
        }
    }

    /// Explicitly download a result from this context. No implicit normalization.
    pub fn download(&self, tensor: &EagerTensor) -> Result<Tensor, String> {
        validate_context(&self.runtime, tensor)?;
        download_tensor(&self.cuda, &tensor.to_tensor().map_err(error)?).map_err(error)
    }

    /// Wait for the current CUDA stream; use around resident timing boundaries.
    pub fn synchronize(&self) -> Result<(), String> {
        self.cuda.synchronize().map_err(error)
    }

    /// Prepare unitary structure and upload small local gate constants.
    /// Validation and physical parameter bindings are shared with the CPU API.
    pub fn prepare(&self, circuit: &DifferentiableCircuit) -> Result<CudaCircuit, String> {
        let template = circuit.template();
        let n = template.circuit().nbits;
        let mut slot = 0;
        let mut gates = Vec::new();
        for element in &template.circuit().elements {
            let CircuitElement::Gate(pg) = element else {
                continue;
            };
            let mut factors = Vec::new();
            if pg.gate.num_params() == 0 {
                factors.push(MatrixFactor::Fixed(self.matrix(&pg.gate.matrix())?));
            } else {
                for i in 0..pg.gate.num_params() {
                    let binding = template.bindings()[slot].clone();
                    slot += 1;
                    let scale = match pg.gate {
                        Gate::Rx(_) | Gate::Ry(_) | Gate::Rz(_) => 0.5,
                        Gate::Phase(_) | Gate::FSim(_, _) => 1.,
                        _ => return Err("Unsupported parameterized CUDA gate".into()),
                    };
                    let generator = pg.gate.generator_matrix(i);
                    let dimension = generator.nrows();
                    // For these generators G has eigenvalues 0 and/or ±i*s.
                    // exp(aG) = I-P + cos(a*s)P + sin(a*s)G/s, P=-G²/s².
                    // FSim's two generators commute; reuse their existing matrices.
                    let projector = generator.dot(&generator).mapv(|x| -x / (scale * scale));
                    let base = Array2::<C>::eye(dimension) - &projector;
                    let sine = generator.mapv(|x| x / scale);
                    if let ParameterBinding::Fixed(angle) = binding {
                        let matrix = base
                            + projector.mapv(|x| x * (angle * scale).cos())
                            + sine.mapv(|x| x * (angle * scale).sin());
                        factors.push(MatrixFactor::Fixed(self.matrix(&matrix)?));
                    } else {
                        factors.push(MatrixFactor::Rotation(Box::new(Rotation {
                            binding,
                            frequency: self.scalar(scale)?,
                            binding_scale: self.scalar(match template.bindings()[slot - 1] {
                                ParameterBinding::Scaled { scale, .. }
                                | ParameterBinding::Product { scale, .. } => scale,
                                ParameterBinding::Fixed(_) => unreachable!(),
                            })?,
                            base: if base.iter().all(|x| *x == C::default()) {
                                None
                            } else {
                                Some(self.matrix(&base)?)
                            },
                            cosine: self.matrix(&projector)?,
                            sine: self.matrix(&sine)?,
                        })));
                    }
                }
            }
            let axes: Vec<_> = pg.target_locs.iter().map(|&site| n - 1 - site).collect();
            let controls: Vec<_> = pg
                .control_locs
                .iter()
                .zip(&pg.control_configs)
                .map(|(&site, &value)| (n - 1 - site, value))
                .collect();
            // tenferro's column-major dot_general puts batch axes last.
            let mut order = axes.clone();
            order.extend(
                (0..n).filter(|axis| {
                    !axes.contains(axis) && !controls.iter().any(|&(c, _)| c == *axis)
                }),
            );
            order.extend(controls.iter().map(|&(axis, _)| axis));
            let permutation = (0..n)
                .map(|axis| order.iter().position(|&i| i == axis).unwrap())
                .collect();
            let identity = if controls.is_empty() {
                None
            } else {
                Some(self.matrix(&Array2::<C>::eye(1usize << axes.len()))?)
            };
            gates.push(DeviceGate {
                factors,
                axes,
                permutation,
                identity,
                controls,
            });
        }
        Ok(CudaCircuit {
            runtime: self.runtime.clone(),
            n,
            parameters: circuit.num_parameters(),
            gates,
            projectors: [
                self.upload(
                    &Tensor::from_vec_col_major(vec![2], vec![C::new(1., 0.), C::default()])
                        .map_err(error)?,
                    false,
                )?,
                self.upload(
                    &Tensor::from_vec_col_major(vec![2], vec![C::default(), C::new(1., 0.)])
                        .map_err(error)?,
                    false,
                )?,
            ],
        })
    }

    fn matrix(&self, matrix: &Array2<C>) -> Result<EagerTensor, String> {
        let data = matrix.t().iter().copied().collect();
        self.upload(
            &Tensor::from_vec_col_major(vec![matrix.nrows(), matrix.ncols()], data)
                .map_err(error)?,
            false,
        )
    }

    fn scalar(&self, value: f64) -> Result<EagerTensor, String> {
        self.upload(
            &Tensor::from_vec_col_major(vec![1], vec![value]).map_err(error)?,
            false,
        )
    }
}

fn validate_context(runtime: &Arc<EagerRuntime>, tensor: &EagerTensor) -> Result<(), String> {
    if !Arc::ptr_eq(runtime, tensor.runtime()) {
        return Err("CUDA tensor belongs to a different runtime".into());
    }
    let value = tensor.value().map_err(error)?;
    if value
        .as_tensor_view()
        .placement()
        .device
        .as_ref()
        .is_none_or(|d| d.kind != DeviceKind::Gpu(GpuBackendKind::Cuda))
    {
        return Err("CUDA execution requires device-resident tensors".into());
    }
    Ok(())
}

/// Prepared unitary circuit. Constants belong to the preparing CUDA context.
/// The state and physical parameters remain explicit runtime inputs.
pub struct CudaCircuit {
    runtime: Arc<EagerRuntime>,
    n: usize,
    parameters: usize,
    gates: Vec<DeviceGate>,
    projectors: [EagerTensor; 2],
}

struct DeviceGate {
    factors: Vec<MatrixFactor>,
    axes: Vec<usize>,
    permutation: Vec<usize>,
    controls: Vec<(usize, bool)>,
    identity: Option<EagerTensor>,
}

enum MatrixFactor {
    Fixed(EagerTensor),
    Rotation(Box<Rotation>),
}

struct Rotation {
    binding: ParameterBinding,
    binding_scale: EagerTensor,
    frequency: EagerTensor,
    base: Option<EagerTensor>,
    cosine: EagerTensor,
    sine: EagerTensor,
}

impl MatrixFactor {
    fn evaluate(&self, parameters: &EagerTensor) -> Result<EagerTensor, String> {
        let rotation = match self {
            Self::Fixed(matrix) => return Ok(matrix.clone()),
            Self::Rotation(rotation) => rotation,
        };
        let Rotation {
            binding,
            binding_scale,
            frequency,
            base,
            cosine,
            sine,
        } = rotation.as_ref();
        let slice = |i| {
            parameters.slice(SliceConfig {
                starts: vec![i],
                limits: vec![i + 1],
                strides: vec![1],
            })
        };
        let angle = match *binding {
            ParameterBinding::Scaled { index, .. } => slice(index).map_err(error)?,
            ParameterBinding::Product { left, right, .. } => slice(left)
                .map_err(error)?
                .mul(&slice(right).map_err(error)?)
                .map_err(error)?,
            ParameterBinding::Fixed(_) => unreachable!(),
        }
        .mul(binding_scale)
        .map_err(error)?
        .mul(frequency)
        .map_err(error)?;
        let expand = |scalar: EagerTensor| {
            scalar
                .cast(DType::C64)?
                .reshape(&[])?
                .broadcast_in_dim(cosine.shape(), &[])
        };
        let mut matrix = cosine
            .mul(&expand(angle.cos().map_err(error)?).map_err(error)?)
            .map_err(error)?
            .add(
                &sine
                    .mul(&expand(angle.sin().map_err(error)?).map_err(error)?)
                    .map_err(error)?,
            )
            .map_err(error)?;
        if let Some(base) = base {
            matrix = matrix.add(base).map_err(error)?;
        }
        Ok(matrix)
    }
}

impl CudaCircuit {
    /// Apply a flat complex128 state using a flat f64 physical parameter vector.
    /// Sites and amplitudes use the same ordering as `ArrayReg`.
    ///
    /// Metadata and context are validated without downloading tensor values.
    /// Values produced by user GPU arithmetic must remain finite; NaNs/overflow
    /// follow the underlying tensor operations. Inputs need not be normalized.
    pub fn apply(
        &self,
        parameters: &EagerTensor,
        input: &EagerTensor,
    ) -> Result<EagerTensor, String> {
        validate_context(&self.runtime, parameters)?;
        validate_context(&self.runtime, input)?;
        if parameters.dtype() != DType::F64 || parameters.shape() != [self.parameters] {
            return Err("CUDA physical parameter shape or dtype mismatch".into());
        }
        if input.dtype() != DType::C64 || input.shape() != [1usize << self.n] {
            return Err("CUDA state shape or dtype mismatch".into());
        }
        let shape = vec![2; self.n];
        let mut state = input.reshape(&shape).map_err(error)?;
        for gate in &self.gates {
            let mut factors = gate.factors.iter();
            let mut matrix = factors.next().unwrap().evaluate(parameters)?;
            for factor in factors {
                matrix = factor
                    .evaluate(parameters)?
                    .dot_general(
                        &matrix,
                        DotGeneralConfig {
                            lhs_contracting_dims: vec![1],
                            rhs_contracting_dims: vec![0],
                            lhs_batch_dims: vec![],
                            rhs_batch_dims: vec![],
                        },
                    )
                    .map_err(error)?;
            }
            let k = gate.axes.len();
            let c = gate.controls.len();
            matrix = matrix.reshape(&vec![2; 2 * k]).map_err(error)?;
            if let Some(identity) = &gate.identity {
                // Controls are batch axes: each batch selects U or I. The
                // state enters one linear contraction, avoiding branches in
                // its AD graph. Storage is 2^c * 4^k, not 4^(c+k).
                let bank_shape = vec![2; c + 2 * k];
                let matrix_axes: Vec<_> = (c..c + 2 * k).collect();
                let identity = identity.reshape(&vec![2; 2 * k]).map_err(error)?;
                let mut delta = matrix
                    .sub(&identity)
                    .map_err(error)?
                    .broadcast_in_dim(&bank_shape, &matrix_axes)
                    .map_err(error)?;
                for (axis, &(_, value)) in gate.controls.iter().enumerate() {
                    let mask = self.projectors[usize::from(value)]
                        .broadcast_in_dim(&bank_shape, &[axis])
                        .map_err(error)?;
                    delta = delta.mul(&mask).map_err(error)?;
                }
                matrix = identity
                    .broadcast_in_dim(&bank_shape, &matrix_axes)
                    .map_err(error)?
                    .add(&delta)
                    .map_err(error)?;
            }
            state = matrix
                .dot_general(
                    &state,
                    DotGeneralConfig {
                        lhs_contracting_dims: (c + k..c + 2 * k).collect(),
                        rhs_contracting_dims: gate.axes.clone(),
                        lhs_batch_dims: (0..c).collect(),
                        rhs_batch_dims: gate.controls.iter().map(|&(axis, _)| axis).collect(),
                    },
                )
                .map_err(error)?
                .transpose(&gate.permutation)
                .map_err(error)?;
        }
        state.reshape(&[1usize << self.n]).map_err(error)
    }
}

#[cfg(test)]
#[path = "unit_tests/cuda.rs"]
mod tests;
