use ndarray::{ArrayD, IxDyn, Array2};
use num_complex::Complex64;

use crate::circuit::PositionedGate;

/// Leg descriptor for tensor network labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leg {
    /// Output leg for site at given index in all_locs.
    Out(usize),
    /// Input leg for site at given index in all_locs.
    In(usize),
    /// Shared (diagonal) leg for site at given index in all_locs.
    Diag(usize),
}

/// Convert a PositionedGate to a tensor and its leg descriptors.
///
/// For non-diagonal gates (or any gate with controls):
///   Shape: (d_ctrl0_out, ..., d_tgtN_out, d_ctrl0_in, ..., d_tgtN_in)
///   Legs: [Out(0), ..., Out(n-1), In(0), ..., In(n-1)]
///
/// For diagonal gates without controls:
///   Shape: (d_tgt0, d_tgt1, ...)
///   Legs: [Diag(0), Diag(1), ...]
pub fn gate_to_tensor(pg: &PositionedGate, dims: &[usize]) -> (ArrayD<Complex64>, Vec<Leg>) {
    let all_locs = pg.all_locs();
    let all_dims: Vec<usize> = all_locs.iter().map(|&loc| dims[loc]).collect();

    let has_controls = !pg.control_locs.is_empty();
    let is_diagonal = pg.gate.is_diagonal() && !has_controls;

    if is_diagonal {
        // Diagonal gate without controls: tensor has one leg per target site
        let target_dims: Vec<usize> = pg.target_locs.iter().map(|&loc| dims[loc]).collect();
        let d = if pg.target_locs.is_empty() { 2 } else { dims[pg.target_locs[0]] };
        let mat = pg.gate.matrix(d);

        // Extract diagonal elements
        let total_dim: usize = target_dims.iter().product();
        let mut data = Vec::with_capacity(total_dim);
        for i in 0..total_dim {
            data.push(mat[[i, i]]);
        }

        let shape = IxDyn(&target_dims);
        let tensor = ArrayD::from_shape_vec(shape, data).unwrap();
        let legs: Vec<Leg> = (0..target_dims.len()).map(Leg::Diag).collect();

        (tensor, legs)
    } else {
        // Non-diagonal gate or gate with controls: build full matrix then reshape
        let full_matrix = build_controlled_matrix(pg, dims);
        let total_dim: usize = all_dims.iter().product();
        assert_eq!(full_matrix.nrows(), total_dim);
        assert_eq!(full_matrix.ncols(), total_dim);

        // Shape: (all_dims_out..., all_dims_in...)
        let mut shape_vec: Vec<usize> = all_dims.clone();
        shape_vec.extend(&all_dims);
        let shape = IxDyn(&shape_vec);

        // Fill the tensor from the matrix in row-major order
        // Matrix element M[row, col] maps to tensor[out_indices..., in_indices...]
        // where row = multi-index over out dimensions, col = multi-index over in dimensions
        let n_sites = all_dims.len();
        let mut data = Vec::with_capacity(total_dim * total_dim);

        for row in 0..total_dim {
            for col in 0..total_dim {
                data.push(full_matrix[[row, col]]);
            }
        }

        let tensor = ArrayD::from_shape_vec(shape, data).unwrap();
        let mut legs: Vec<Leg> = (0..n_sites).map(Leg::Out).collect();
        legs.extend((0..n_sites).map(Leg::In));

        (tensor, legs)
    }
}

/// Build the full controlled gate matrix for a PositionedGate.
///
/// all_locs = control_locs ++ target_locs
/// The resulting matrix is total_dim x total_dim where total_dim = product(all_dims).
fn build_controlled_matrix(pg: &PositionedGate, dims: &[usize]) -> Array2<Complex64> {
    let all_locs = pg.all_locs();
    let all_dims: Vec<usize> = all_locs.iter().map(|&loc| dims[loc]).collect();
    let total_dim: usize = all_dims.iter().product();

    let ctrl_dims: Vec<usize> = pg.control_locs.iter().map(|&loc| dims[loc]).collect();
    let ctrl_dim: usize = if ctrl_dims.is_empty() { 1 } else { ctrl_dims.iter().product() };
    let tgt_dim: usize = pg.target_locs.iter().map(|&loc| dims[loc]).product();

    if pg.control_locs.is_empty() {
        // No controls: just return the gate matrix
        let d = if pg.target_locs.is_empty() { 2 } else { dims[pg.target_locs[0]] };
        return pg.gate.matrix(d);
    }

    // Compute the trigger index from control_configs
    // control_configs is Vec<bool>, control sites have dimension 2
    // trigger_index = sum over i of (config[i] as usize) * product of remaining ctrl dims
    let trigger_index = compute_trigger_index(&pg.control_configs, &ctrl_dims);

    // Get the gate matrix for targets
    let d = if pg.target_locs.is_empty() { 2 } else { dims[pg.target_locs[0]] };
    let gate_matrix = pg.gate.matrix(d);

    let one = Complex64::new(1.0, 0.0);

    // Build total_dim x total_dim matrix
    // Index = ctrl_index * tgt_dim + tgt_index
    let mut mat = Array2::zeros((total_dim, total_dim));

    for ctrl_idx in 0..ctrl_dim {
        for tgt_row in 0..tgt_dim {
            let row = ctrl_idx * tgt_dim + tgt_row;
            if ctrl_idx == trigger_index {
                // Apply gate matrix on target indices
                for tgt_col in 0..tgt_dim {
                    let col = ctrl_idx * tgt_dim + tgt_col;
                    mat[[row, col]] = gate_matrix[[tgt_row, tgt_col]];
                }
            } else {
                // Identity on target indices
                let col = ctrl_idx * tgt_dim + tgt_row;
                mat[[row, col]] = one;
            }
        }
    }

    mat
}

/// Compute the trigger index from control configurations and control dimensions.
/// Uses row-major indexing: config[0]*d1*d2*... + config[1]*d2*... + ...
fn compute_trigger_index(control_configs: &[bool], ctrl_dims: &[usize]) -> usize {
    let n = control_configs.len();
    let mut index = 0usize;
    let mut stride = 1usize;
    for i in (0..n).rev() {
        index += (control_configs[i] as usize) * stride;
        stride *= ctrl_dims[i];
    }
    index
}
