//! Native tensor network contractor using omeinsum.
//!
//! Enable with the `omeinsum` feature flag. Requires the `omeinsum-rs`
//! git submodule to be initialized (`git submodule update --init`).

use ndarray::{ArrayD, ShapeBuilder};
use num_complex::Complex64;
use omeinsum::{Cpu, Einsum, Standard, Tensor};

use crate::einsum::TensorNetwork;

/// Contract a tensor network using omeinsum's native Rust backend.
///
/// Uses greedy optimization to find a contraction order, then
/// executes pairwise contractions on CPU.
///
/// # Arguments
/// * `tn` - The tensor network to contract
///
/// # Returns
/// The contracted result as an ndarray `ArrayD<Complex64>`.
pub fn contract(tn: &TensorNetwork) -> ArrayD<Complex64> {
    let tensors: Vec<Tensor<Complex64, Cpu>> = tn.tensors.iter().map(ndarray_to_omeinsum).collect();
    let tensor_refs: Vec<&Tensor<Complex64, Cpu>> = tensors.iter().collect();

    let ixs: Vec<Vec<usize>> = tn.code.ixs.clone();
    let iy: Vec<usize> = tn.code.iy.clone();

    let mut ein = Einsum::new(ixs, iy.clone(), tn.size_dict.clone());
    ein.optimize_greedy();
    let result = ein.execute::<Standard<Complex64>, Complex64, Cpu>(&tensor_refs);

    omeinsum_to_ndarray(&result, &iy, &tn.size_dict)
}

/// Convert an ndarray `ArrayD<Complex64>` to an omeinsum `Tensor`.
///
/// omeinsum uses column-major (Fortran) order, so we need to
/// convert from ndarray's default row-major layout.
fn ndarray_to_omeinsum(arr: &ArrayD<Complex64>) -> Tensor<Complex64, Cpu> {
    let shape: Vec<usize> = arr.shape().to_vec();
    // omeinsum Tensor::from_data expects column-major data
    // ndarray standard_layout is row-major, so we iterate in Fortran order
    let data: Vec<Complex64> = arr.t().iter().copied().collect();
    Tensor::from_data(&data, &shape)
}

/// Convert an omeinsum `Tensor` back to an ndarray `ArrayD<Complex64>`.
fn omeinsum_to_ndarray(
    tensor: &Tensor<Complex64, Cpu>,
    iy: &[usize],
    size_dict: &std::collections::HashMap<usize, usize>,
) -> ArrayD<Complex64> {
    let shape: Vec<usize> = iy.iter().map(|l| size_dict[l]).collect();
    let data = tensor.to_vec();
    if shape.is_empty() {
        // Scalar result
        ArrayD::from_shape_vec(ndarray::IxDyn(&[]), data).unwrap()
    } else {
        // omeinsum stores column-major; return column-major ndarray
        ArrayD::from_shape_vec(ndarray::IxDyn(&shape).f(), data).unwrap()
    }
}

#[cfg(test)]
#[path = "unit_tests/contractor.rs"]
mod tests;
