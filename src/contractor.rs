//! Native tensor network contractor using omeinsum.
//!
//! Enable with the `omeinsum` feature flag.

use std::collections::HashMap;

use ndarray::{ArrayD, ShapeBuilder};
use num_complex::Complex64;
use omeco::{EinCode, Label, NestedEinsum};
use omeinsum::{Cpu, Einsum, Standard, Tensor};

use crate::einsum::{TensorNetwork, TensorNetworkDM};

/// Contract a tensor network using omeinsum's native Rust backend.
pub fn contract(tn: &TensorNetwork) -> ArrayD<Complex64> {
    contract_impl(&tn.tensors, &tn.code, &tn.size_dict, None)
}

/// Contract a tensor network with a pre-computed contraction tree.
pub fn contract_with_tree(tn: &TensorNetwork, tree: NestedEinsum<usize>) -> ArrayD<Complex64> {
    contract_impl(&tn.tensors, &tn.code, &tn.size_dict, Some(tree))
}

/// Contract a density-matrix tensor network using omeinsum's native Rust backend.
pub fn contract_dm(tn: &TensorNetworkDM) -> ArrayD<Complex64> {
    contract_impl(&tn.tensors, &tn.code, &tn.size_dict, None)
}

/// Contract a density-matrix tensor network with a pre-computed contraction tree.
pub fn contract_dm_with_tree(tn: &TensorNetworkDM, tree: NestedEinsum<i32>) -> ArrayD<Complex64> {
    contract_impl(&tn.tensors, &tn.code, &tn.size_dict, Some(tree))
}

fn contract_impl<L: Label>(
    tensors_in: &[ArrayD<Complex64>],
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    tree: Option<NestedEinsum<L>>,
) -> ArrayD<Complex64> {
    let tensors: Vec<Tensor<Complex64, Cpu>> = tensors_in.iter().map(ndarray_to_omeinsum).collect();
    let tensor_refs: Vec<&Tensor<Complex64, Cpu>> = tensors.iter().collect();
    let remapped = remap_network(code, size_dict, tree.as_ref());

    let mut ein = Einsum::new(
        remapped.code.ixs.clone(),
        remapped.code.iy.clone(),
        remapped.size_dict.clone(),
    );
    if let Some(tree) = tree.as_ref() {
        ein.set_contraction_tree(remap_tree(tree, &remapped.label_map));
    } else {
        ein.optimize_greedy();
    }

    let result = ein.execute::<Standard<Complex64>, Complex64, Cpu>(&tensor_refs);
    omeinsum_to_ndarray(&result, &remapped.code.iy, &remapped.size_dict)
}

struct RemappedNetwork<L: Label> {
    code: EinCode<usize>,
    size_dict: HashMap<usize, usize>,
    label_map: HashMap<L, usize>,
}

fn remap_network<L: Label>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    tree: Option<&NestedEinsum<L>>,
) -> RemappedNetwork<L> {
    let mut label_map = HashMap::with_capacity(size_dict.len());
    let mut next_label = 0usize;

    let mut record = |label: &L| {
        label_map.entry(label.clone()).or_insert_with(|| {
            let mapped = next_label;
            next_label += 1;
            mapped
        });
    };

    for legs in &code.ixs {
        for label in legs {
            record(label);
        }
    }
    for label in &code.iy {
        record(label);
    }
    if let Some(tree) = tree {
        collect_tree_labels(tree, &mut record);
    }
    for label in size_dict.keys() {
        record(label);
    }

    let code = remap_code(code, &label_map);
    let size_dict = label_map
        .iter()
        .map(|(label, mapped)| (*mapped, size_dict[label]))
        .collect();

    RemappedNetwork {
        code,
        size_dict,
        label_map,
    }
}

fn collect_tree_labels<L: Label>(tree: &NestedEinsum<L>, record: &mut impl FnMut(&L)) {
    match tree {
        NestedEinsum::Leaf { .. } => {}
        NestedEinsum::Node { args, eins } => {
            for legs in &eins.ixs {
                for label in legs {
                    record(label);
                }
            }
            for label in &eins.iy {
                record(label);
            }
            for arg in args {
                collect_tree_labels(arg, record);
            }
        }
    }
}

fn remap_code<L: Label>(code: &EinCode<L>, label_map: &HashMap<L, usize>) -> EinCode<usize> {
    let ixs = code
        .ixs
        .iter()
        .map(|legs| legs.iter().map(|label| label_map[label]).collect())
        .collect();
    let iy = code.iy.iter().map(|label| label_map[label]).collect();

    EinCode::new(ixs, iy)
}

fn remap_tree<L: Label>(
    tree: &NestedEinsum<L>,
    label_map: &HashMap<L, usize>,
) -> NestedEinsum<usize> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => NestedEinsum::leaf(*tensor_index),
        NestedEinsum::Node { args, eins } => NestedEinsum::node(
            args.iter().map(|arg| remap_tree(arg, label_map)).collect(),
            remap_code(eins, label_map),
        ),
    }
}

/// Convert an ndarray `ArrayD<Complex64>` to an omeinsum `Tensor`.
///
/// omeinsum uses column-major (Fortran) order, so we need to
/// convert from ndarray's default row-major layout.
fn ndarray_to_omeinsum(arr: &ArrayD<Complex64>) -> Tensor<Complex64, Cpu> {
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<Complex64> = arr.t().iter().copied().collect();
    Tensor::from_data(&data, &shape)
}

/// Convert an omeinsum `Tensor` back to an ndarray `ArrayD<Complex64>`.
fn omeinsum_to_ndarray<L: Label>(
    tensor: &Tensor<Complex64, Cpu>,
    iy: &[L],
    size_dict: &HashMap<L, usize>,
) -> ArrayD<Complex64> {
    let shape: Vec<usize> = iy.iter().map(|label| size_dict[label]).collect();
    let data = tensor.to_vec();
    if shape.is_empty() {
        ArrayD::from_shape_vec(ndarray::IxDyn(&[]), data).unwrap()
    } else {
        ArrayD::from_shape_vec(ndarray::IxDyn(&shape).f(), data).unwrap()
    }
}

#[cfg(test)]
#[path = "unit_tests/contractor.rs"]
mod tests;
