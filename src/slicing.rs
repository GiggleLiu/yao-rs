//! Memory estimates and deterministic, streamed tensor-network slicing.
//!
//! All storage quantities are bytes of complex128 values. Estimates include
//! input/output storage and conservative tensor buffers, plus a caller-selected
//! workspace reserve. They are not a hard process RSS guarantee: compiled code,
//! provider scratch, allocator retention and runtime metadata are not measured
//! by omeco. Exactly one slice is active at a time; no nested slice thread pool
//! is created. A prepared tenferro contraction can still use CPU threads.

use std::collections::{HashMap, HashSet};

use ndarray::{ArrayD, Axis, IxDyn, Slice};
use num_complex::Complex64 as C;
use omeco::{EinCode, Label, NestedEinsum, SlicedEinsum, TreeSASlicer};
use serde::{Deserialize, Serialize};

use crate::contraction_plan::validate_tree;

/// Limits on estimated storage and total slice work. Zero limits are invalid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SliceBudget {
    /// Limit on the complete estimate, in bytes; `None` means no storage limit.
    pub max_bytes: Option<usize>,
    /// Additional reserve for backend workspace, in bytes. This is a user
    /// allowance, not a backend-reported measurement or guaranteed upper bound.
    pub workspace_bytes: usize,
    /// Limit on the product of sliced dimensions, including output slices.
    pub max_slices: usize,
}

impl Default for SliceBudget {
    fn default() -> Self {
        Self {
            max_bytes: None,
            workspace_bytes: 0,
            max_slices: 1_000_000,
        }
    }
}

/// Storage accounting for one active slice, independent of tensor values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub input_bytes: usize,
    pub output_bytes: usize,
    /// Largest input or intermediate tensor within one slice.
    pub largest_intermediate_bytes: usize,
    /// omeco's depth-first live tensor estimate, including slice inputs/output.
    pub omeco_peak_bytes: usize,
    /// Sum of slice input copies, backend input adaptation, all node outputs
    /// and result adaptation, including one temporary input-layout copy.
    /// Conservatively retains every intermediate; provider workspace is separate.
    pub worker_buffer_bytes: usize,
    pub workspace_reserved_bytes: usize,
    pub estimated_total_bytes: usize,
    pub slices: usize,
    pub concurrent_slices: usize,
}

/// A validated expression, caller-supplied tree, and fixed slicing semantics.
/// Internal slices sum contributions; output slices fill their corresponding
/// output coordinates. Slices follow first-label appearance order, with the
/// final slice label varying fastest. Duplicate slice labels are rejected.
#[derive(Debug, Clone)]
pub struct SlicedPlan<L: Label> {
    code: EinCode<L>,
    sizes: HashMap<L, usize>,
    sliced: SlicedEinsum<L>,
    slice_sizes: HashMap<L, usize>,
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
    budget: SliceBudget,
    estimate: MemoryEstimate,
}

fn add(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| "Tensor storage estimate overflow".into())
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| "Tensor storage or slice count overflow".into())
}
fn elements(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1, |n, &d| mul(n, d))
}
fn shape<L: Label>(labels: &[L], sizes: &HashMap<L, usize>) -> Result<Vec<usize>, String> {
    let shape = labels
        .iter()
        .map(|l| {
            sizes
                .get(l)
                .copied()
                .filter(|&d| d > 0)
                .ok_or_else(|| format!("Missing or zero dimension for label {l:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if mul(elements(&shape)?, size_of::<C>())? > isize::MAX as usize {
        return Err("Tensor exceeds addressable complex128 storage".into());
    }
    Ok(shape)
}
fn bytes<L: Label>(labels: &[L], sizes: &HashMap<L, usize>) -> Result<usize, String> {
    mul(elements(&shape(labels, sizes)?)?, size_of::<C>())
}

// Checked totals also protect omeco's unchecked usize arithmetic. Every value
// in its depth-first peak is bounded by this sum of all leaf/node storage.
fn tree_storage<L: Label>(
    tree: &NestedEinsum<L>,
    code: &EinCode<L>,
    sizes: &HashMap<L, usize>,
) -> Result<(usize, usize), String> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let n = bytes(&code.ixs[*tensor_index], sizes)?;
            Ok((n, n))
        }
        NestedEinsum::Node { args, eins } => {
            let n = bytes(&eins.iy, sizes)?;
            args.iter().try_fold((n, n), |(total, largest), arg| {
                let (child, peak) = tree_storage(arg, code, sizes)?;
                Ok((add(total, child)?, largest.max(peak)))
            })
        }
    }
}

impl<L: Label> SlicedPlan<L> {
    /// Keep the supplied contraction tree exactly, fixing the requested labels.
    /// No tensor values are allocated while constructing or checking the plan.
    pub fn new(
        code: &EinCode<L>,
        sizes: &HashMap<L, usize>,
        tree: &NestedEinsum<L>,
        slicing: &[L],
        budget: SliceBudget,
    ) -> Result<Self, String> {
        if budget.max_slices == 0 || budget.max_bytes == Some(0) {
            return Err("Slice and memory limits must be positive".into());
        }
        validate_tree(tree, code)?;
        let labels: HashSet<_> = code.ixs.iter().flatten().cloned().collect();
        let mut output_labels = HashSet::new();
        for label in &code.iy {
            if !labels.contains(label) || !output_labels.insert(label) {
                return Err(format!("Invalid output label {label:?}"));
            }
        }
        let mut selected = HashSet::new();
        for l in slicing {
            if !labels.contains(l) || !selected.insert(l.clone()) {
                return Err(format!("Unknown or duplicate slice label {l:?}"));
            }
        }
        // Canonicalize without requiring Ord or relying on HashMap iteration.
        let mut slicing = Vec::new();
        for l in code.ixs.iter().flatten() {
            if selected.remove(l) {
                slicing.push(l.clone());
            }
        }
        let input_shapes = code
            .ixs
            .iter()
            .map(|ix| shape(ix, sizes))
            .collect::<Result<Vec<_>, _>>()?;
        let output_shape = shape(&code.iy, sizes)?;
        let input_bytes = code
            .ixs
            .iter()
            .try_fold(0, |sum, ix| add(sum, bytes(ix, sizes)?))?;
        let output_bytes = bytes(&code.iy, sizes)?;
        let slices = slicing.iter().try_fold(1, |n, l| mul(n, sizes[l]))?;
        if slices > budget.max_slices {
            return Err(format!(
                "Plan requires {slices} slices, exceeding limit {}",
                budget.max_slices
            ));
        }
        let mut slice_sizes = sizes.clone();
        for l in &slicing {
            slice_sizes.insert(l.clone(), 1);
        }
        let (all_buffers, largest_intermediate_bytes) = tree_storage(tree, code, &slice_sizes)?;
        let omeco_peak_bytes = mul(
            omeco::peak_memory(tree, &slice_sizes, &code.ixs),
            size_of::<C>(),
        )?;
        let slice_input_bytes = code
            .ixs
            .iter()
            .try_fold(0, |sum, ix| add(sum, bytes(ix, &slice_sizes)?))?;
        let layout_scratch = code
            .ixs
            .iter()
            .map(|ix| bytes(ix, &slice_sizes))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let worker_buffer_bytes = add(
            add(
                add(all_buffers, slice_input_bytes)?,
                bytes(&code.iy, &slice_sizes)?,
            )?,
            layout_scratch,
        )?;
        let estimated_total_bytes = add(
            add(add(input_bytes, output_bytes)?, worker_buffer_bytes)?,
            budget.workspace_bytes,
        )?;
        let estimate = MemoryEstimate {
            input_bytes,
            output_bytes,
            largest_intermediate_bytes,
            omeco_peak_bytes,
            worker_buffer_bytes,
            workspace_reserved_bytes: budget.workspace_bytes,
            estimated_total_bytes,
            slices,
            concurrent_slices: 1,
        };
        if budget
            .max_bytes
            .is_some_and(|limit| estimated_total_bytes > limit)
        {
            return Err(format!(
                "Estimated storage {estimated_total_bytes} bytes exceeds memory budget {} bytes (inputs {input_bytes}, output {output_bytes}, worker {worker_buffer_bytes}, workspace reserve {})",
                budget.max_bytes.unwrap(),
                budget.workspace_bytes
            ));
        }
        Ok(Self {
            code: code.clone(),
            sizes: sizes.clone(),
            sliced: SlicedEinsum::new(slicing, tree.clone()),
            slice_sizes,
            input_shapes,
            output_shape,
            budget,
            estimate,
        })
    }

    /// Ask omeco TreeSA to select slices and refine the tree until the complete
    /// estimate fits the budget. This explicitly allows replanning; use `new`
    /// to preserve a fixed tree. Every returned plan is validated again.
    /// Planner heuristics can fail to find a feasible plan; that is an error,
    /// not evidence that no possible plan exists.
    pub fn auto(
        code: &EinCode<L>,
        sizes: &HashMap<L, usize>,
        tree: &NestedEinsum<L>,
        budget: SliceBudget,
        config: &TreeSASlicer,
    ) -> Result<Self, String> {
        let baseline = Self::new(
            code,
            sizes,
            tree,
            &[],
            SliceBudget {
                max_bytes: None,
                ..budget
            },
        )?;
        let limit = budget
            .max_bytes
            .ok_or("Automatic slicing requires a memory budget in bytes")?;
        if limit == 0 {
            return Err("Memory budget must be positive".into());
        }
        if baseline.estimate.estimated_total_bytes <= limit {
            return Self::new(code, sizes, tree, &[], budget);
        }
        let floor = add(
            add(
                baseline.estimate.input_bytes,
                baseline.estimate.output_bytes,
            )?,
            budget.workspace_bytes,
        )?;
        if floor >= limit {
            return Err(format!(
                "Memory budget {limit} bytes cannot hold inputs, output and workspace reserve ({floor} bytes)"
            ));
        }
        if config.ntrials == 0
            || config.betas.is_empty()
            || config.betas.iter().any(|v| !v.is_finite() || *v < 0.)
            || !config.optimization_ratio.is_finite()
            || config.optimization_ratio < 0.
            || [
                config.score.tc_weight,
                config.score.sc_weight,
                config.score.rw_weight,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
            || !config.fixed_slices.is_empty()
        {
            return Err("Invalid TreeSA slicer configuration; use new for fixed slices".into());
        }
        // All checked tensors fit isize::MAX. At most 64 decreasing targets
        // reach scalar intermediates, so planner retries are bounded.
        let start = (baseline.estimate.largest_intermediate_bytes / size_of::<C>())
            .max(1)
            .ilog2();
        let mut last_error = String::new();
        for target in (0..=start).rev() {
            let sliced = omeco::slice_code(
                tree,
                sizes,
                &config.clone().with_sc_target(target as f64),
                &code.ixs,
            )
            .ok_or("TreeSA slicing requires a binary contraction tree")?;
            match Self::new(code, sizes, &sliced.eins, &sliced.slicing, budget) {
                Ok(plan) => return Ok(plan),
                Err(error) => last_error = error,
            }
        }
        Err(format!(
            "TreeSA did not find a plan within the requested limits: {last_error}"
        ))
    }

    pub fn code(&self) -> &EinCode<L> {
        &self.code
    }
    pub fn sizes(&self) -> &HashMap<L, usize> {
        &self.sizes
    }
    pub fn tree(&self) -> &NestedEinsum<L> {
        &self.sliced.eins
    }
    pub fn slicing(&self) -> &[L] {
        &self.sliced.slicing
    }
    /// Dimensions used to compile one reusable slice contraction (fixed axes = 1).
    pub fn slice_sizes(&self) -> &HashMap<L, usize> {
        &self.slice_sizes
    }
    pub fn estimate(&self) -> MemoryEstimate {
        self.estimate
    }
    pub fn budget(&self) -> SliceBudget {
        self.budget
    }

    /// Stream slices through an executor prepared for `code`, `tree` and
    /// `slice_sizes`. The callback returns logical output axes in `code.iy`
    /// order. The callback's allocations must be accounted for separately if
    /// they exceed the documented tensor model and workspace reserve.
    pub fn execute_with(
        &self,
        tensors: &[ArrayD<C>],
        mut execute: impl FnMut(&[ArrayD<C>]) -> Result<ArrayD<C>, String>,
    ) -> Result<ArrayD<C>, String> {
        if tensors.len() != self.input_shapes.len()
            || tensors
                .iter()
                .zip(&self.input_shapes)
                .any(|(t, s)| t.shape() != s.as_slice())
        {
            return Err("Sliced contraction input shape/count mismatch".into());
        }
        let slice_output_shape = shape(&self.code.iy, &self.slice_sizes)?;
        let mut run = |inputs: &[ArrayD<C>]| {
            let result = execute(inputs)?;
            if result.shape() != slice_output_shape {
                return Err("Slice executor returned the wrong output shape".into());
            }
            Ok(result)
        };
        if self.slicing().is_empty() {
            return run(tensors);
        }
        let mut output = ArrayD::zeros(IxDyn(&self.output_shape));
        let mut sliced_inputs = self
            .code
            .ixs
            .iter()
            .map(|ix| Ok(ArrayD::zeros(IxDyn(&shape(ix, &self.slice_sizes)?))))
            .collect::<Result<Vec<_>, String>>()?;
        let axes: Vec<Vec<Option<usize>>> = self
            .code
            .ixs
            .iter()
            .map(|ix| {
                ix.iter()
                    .map(|l| self.slicing().iter().position(|s| s == l))
                    .collect()
            })
            .collect();
        let output_axes: Vec<_> = self
            .code
            .iy
            .iter()
            .map(|l| self.slicing().iter().position(|s| s == l))
            .collect();
        let mut digits = vec![0; self.slicing().len()];
        for slice in 0..self.estimate.slices {
            let mut remainder = slice;
            for (i, l) in self.slicing().iter().enumerate().rev() {
                digits[i] = remainder % self.sizes[l];
                remainder /= self.sizes[l];
            }
            for ((destination, source), axes) in sliced_inputs.iter_mut().zip(tensors).zip(&axes) {
                if slice != 0 && axes.iter().all(Option::is_none) {
                    continue;
                }
                let mut view = source.view();
                for (axis, selected) in axes.iter().enumerate() {
                    if let Some(selected) = selected {
                        let coordinate = digits[*selected] as isize;
                        view.slice_axis_inplace(
                            Axis(axis),
                            Slice::new(coordinate, Some(coordinate + 1), 1),
                        );
                    }
                }
                destination.assign(&view);
            }
            let result = run(&sliced_inputs)?;
            let mut view = output.view_mut();
            for (axis, selected) in output_axes.iter().enumerate() {
                if let Some(selected) = selected {
                    let coordinate = digits[*selected] as isize;
                    view.slice_axis_inplace(
                        Axis(axis),
                        Slice::new(coordinate, Some(coordinate + 1), 1),
                    );
                }
            }
            view.zip_mut_with(&result, |destination, &value| *destination += value);
        }
        Ok(output)
    }
}

#[cfg(test)]
#[path = "unit_tests/slicing.rs"]
mod tests;
