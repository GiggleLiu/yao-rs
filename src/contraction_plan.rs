//! Shared validation of explicit tensor contraction trees.

use omeco::{CodeOptimizer, EinCode, Label, NestedEinsum};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "tenferro")]
pub(crate) struct TensorShapes {
    pub inputs: Vec<Vec<usize>>,
    pub output: Vec<usize>,
}

/// Shared complex128 shape/addressability validation for tenferro providers.
#[cfg(feature = "tenferro")]
pub(crate) fn tensor_shapes<L: Label>(
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
) -> Result<TensorShapes, String> {
    let labels: HashSet<_> = code.ixs.iter().flatten().collect();
    let mut outputs = HashSet::new();
    for label in &code.iy {
        if !labels.contains(label) || !outputs.insert(label) {
            return Err(format!("Invalid output label {label:?}"));
        }
    }
    Ok(TensorShapes {
        inputs: code
            .ixs
            .iter()
            .map(|xs| tensor_shape(xs, sizes))
            .collect::<Result<_, _>>()?,
        output: tensor_shape(&code.iy, sizes)?,
    })
}

#[cfg(feature = "tenferro")]
pub(crate) fn tensor_shape<L: Label>(
    labels: &[L],
    sizes: &HashMap<L, usize>,
) -> Result<Vec<usize>, String> {
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
        .try_fold(size_of::<num_complex::Complex64>(), |n, &d| {
            n.checked_mul(d)
        });
    if bytes.is_none_or(|n| n > isize::MAX as usize) {
        return Err("Tensor shape exceeds addressable complex128 storage".into());
    }
    Ok(shape)
}

/// Optimize a network, retaining unary traces, diagonals and permutations.
///
/// omeco's single-input optimizer returns a leaf without applying the output
/// expression. For zero or one input there is no ordering decision: use one
/// node carrying the complete expression. Larger networks use `optimizer`.
/// Validate dimensions before invoking an optimizer.
pub fn optimize_code<L: Label, O: CodeOptimizer>(
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
    optimizer: &O,
) -> Option<NestedEinsum<L>> {
    if code.ixs.len() <= 1 {
        Some(NestedEinsum::node(
            (0..code.ixs.len()).map(NestedEinsum::leaf).collect(),
            code.clone(),
        ))
    } else {
        omeco::optimize_code(code, sizes, optimizer)
    }
}

/// Validate a contraction tree against its complete network expression.
///
/// Every input must occur exactly once, child outputs must match parent inputs,
/// and no node may sum an index still needed elsewhere. Output order matters.
/// This validates semantics; dimension and allocation checks belong to the
/// executor. Repeated input labels (traces/diagonals) are allowed.
pub fn validate_tree<L: Label>(tree: &NestedEinsum<L>, code: &EinCode<L>) -> Result<(), String> {
    let mut visited = HashSet::new();
    let labels = visit(tree, code, &mut visited)?;
    if visited.len() != code.ixs.len() {
        return Err("Contraction order must use every input exactly once".into());
    }
    if labels != code.iy {
        return Err("Contraction order output does not match network output".into());
    }
    Ok(())
}

fn visit<L: Label>(
    tree: &NestedEinsum<L>,
    code: &EinCode<L>,
    visited: &mut HashSet<usize>,
) -> Result<Vec<L>, String> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let labels = code.ixs.get(*tensor_index).ok_or_else(|| {
                format!("Contraction tensor index {tensor_index} is out of range")
            })?;
            if !visited.insert(*tensor_index) {
                return Err(format!("Contraction order repeats tensor {tensor_index}"));
            }
            Ok(labels.clone())
        }
        NestedEinsum::Node { args, eins } => {
            if args.len() != eins.ixs.len() {
                return Err("Contraction node input count mismatch".into());
            }
            // An empty node is the scalar multiplicative identity, and is valid
            // only for the complete empty network.
            if args.is_empty() && (!code.ixs.is_empty() || !eins.iy.is_empty()) {
                return Err("Invalid empty contraction node".into());
            }
            let previous = visited.clone();
            for (arg, expected) in args.iter().zip(&eins.ixs) {
                if visit(arg, code, visited)? != *expected {
                    return Err("Contraction node indices disagree with child output".into());
                }
            }
            let labels: HashSet<_> = eins.ixs.iter().flatten().collect();
            let mut outputs = HashSet::new();
            for label in &eins.iy {
                if !labels.contains(label) || !outputs.insert(label) {
                    return Err(format!("Invalid contraction output label {label:?}"));
                }
            }
            let subtree: HashSet<_> = visited.difference(&previous).copied().collect();
            for label in code.iy.iter().chain(
                code.ixs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !subtree.contains(i))
                    .flat_map(|(_, legs)| legs),
            ) {
                if labels.contains(label) && !outputs.contains(label) {
                    return Err(format!(
                        "Contraction order eliminates required label {label:?}"
                    ));
                }
            }
            Ok(eins.iy.clone())
        }
    }
}
