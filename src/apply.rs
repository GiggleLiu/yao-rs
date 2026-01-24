use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::circuit::Circuit;
use crate::state::State;

/// Decompose a flat index into a multi-index given dimensions (row-major order).
fn linear_to_multi(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut multi = vec![0usize; n];
    for i in (0..n).rev() {
        multi[i] = index % dims[i];
        index /= dims[i];
    }
    multi
}

/// Compose a multi-index into a flat index given dimensions (row-major order).
fn multi_to_linear(indices: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    for (i, &idx) in indices.iter().enumerate() {
        let stride: usize = dims[i + 1..].iter().product();
        index += idx * stride;
    }
    index
}

/// Simple matrix-vector multiplication for complex matrices and vectors.
fn matrix_vector_mul(mat: &Array2<Complex64>, vec: &Array1<Complex64>) -> Array1<Complex64> {
    let n = mat.nrows();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..mat.ncols() {
            sum += mat[[i, j]] * vec[j];
        }
        result[i] = sum;
    }
    result
}

/// Build the controlled local matrix on all involved sites (controls + targets).
///
/// all_dims: dimensions for all involved sites (control_locs ++ target_locs order)
/// gate_matrix: the gate's matrix on target sites only
/// control_configs: which control configuration triggers the gate
/// num_controls: number of control sites
fn build_controlled_matrix(
    all_dims: &[usize],
    gate_matrix: &Array2<Complex64>,
    control_configs: &[bool],
    num_controls: usize,
) -> Array2<Complex64> {
    if num_controls == 0 {
        return gate_matrix.clone();
    }

    let involved_dim: usize = all_dims.iter().product();
    let control_dims = &all_dims[..num_controls];
    let target_dims = &all_dims[num_controls..];
    let target_dim: usize = target_dims.iter().product();

    let mut mat = Array2::zeros((involved_dim, involved_dim));

    // Compute the trigger index in the control subspace
    let trigger_index: usize = control_configs
        .iter()
        .enumerate()
        .map(|(i, &cfg)| {
            let val = if cfg { 1usize } else { 0usize };
            let stride: usize = control_dims[i + 1..].iter().product();
            val * stride
        })
        .sum();

    let control_dim: usize = control_dims.iter().product();

    for ctrl_idx in 0..control_dim {
        for t_row in 0..target_dim {
            let row = ctrl_idx * target_dim + t_row;
            if ctrl_idx == trigger_index {
                // Apply the gate matrix on the target portion
                for t_col in 0..target_dim {
                    let col = ctrl_idx * target_dim + t_col;
                    mat[[row, col]] = gate_matrix[[t_row, t_col]];
                }
            } else {
                // Identity on the target portion
                let col = ctrl_idx * target_dim + t_row;
                mat[[row, col]] = Complex64::new(1.0, 0.0);
            }
        }
    }

    mat
}

/// Apply a circuit to a quantum state by building and multiplying full-space matrices.
///
/// For each gate in the circuit:
/// 1. Build the controlled local matrix on all involved sites
/// 2. Embed it into the full Hilbert space
/// 3. Multiply by the state vector
pub fn apply(circuit: &Circuit, state: &State) -> State {
    let dims = &circuit.dims;
    let total_dim = circuit.total_dim();
    let mut current_data = state.data.clone();

    for pg in &circuit.gates {
        // Get the gate's local matrix on target sites
        let d = dims[pg.target_locs[0]];
        let gate_matrix = pg.gate.matrix(d);

        // Build the controlled local matrix on all involved sites
        let all_locs = pg.all_locs(); // control_locs ++ target_locs
        let all_dims: Vec<usize> = all_locs.iter().map(|&loc| dims[loc]).collect();
        let num_controls = pg.control_locs.len();

        let local_matrix = build_controlled_matrix(
            &all_dims,
            &gate_matrix,
            &pg.control_configs,
            num_controls,
        );

        // Embed into full Hilbert space and multiply
        let mut full_matrix = Array2::zeros((total_dim, total_dim));

        for row in 0..total_dim {
            let row_multi = linear_to_multi(row, dims);
            for col in 0..total_dim {
                let col_multi = linear_to_multi(col, dims);

                // Check that non-involved sites are the same between row and col
                let mut non_involved_match = true;
                for site in 0..dims.len() {
                    if !all_locs.contains(&site)
                        && row_multi[site] != col_multi[site]
                    {
                        non_involved_match = false;
                        break;
                    }
                }

                if !non_involved_match {
                    // entry is 0 (already initialized)
                    continue;
                }

                // Extract involved-site indices for row and col
                let row_involved: Vec<usize> =
                    all_locs.iter().map(|&loc| row_multi[loc]).collect();
                let col_involved: Vec<usize> =
                    all_locs.iter().map(|&loc| col_multi[loc]).collect();

                let local_row = multi_to_linear(&row_involved, &all_dims);
                let local_col = multi_to_linear(&col_involved, &all_dims);

                full_matrix[[row, col]] = local_matrix[[local_row, local_col]];
            }
        }

        current_data = matrix_vector_mul(&full_matrix, &current_data);
    }

    State {
        dims: dims.clone(),
        data: current_data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_multi() {
        assert_eq!(linear_to_multi(0, &[2, 3]), vec![0, 0]);
        assert_eq!(linear_to_multi(1, &[2, 3]), vec![0, 1]);
        assert_eq!(linear_to_multi(3, &[2, 3]), vec![1, 0]);
        assert_eq!(linear_to_multi(5, &[2, 3]), vec![1, 2]);
    }

    #[test]
    fn test_multi_to_linear() {
        assert_eq!(multi_to_linear(&[0, 0], &[2, 3]), 0);
        assert_eq!(multi_to_linear(&[0, 1], &[2, 3]), 1);
        assert_eq!(multi_to_linear(&[1, 0], &[2, 3]), 3);
        assert_eq!(multi_to_linear(&[1, 2], &[2, 3]), 5);
    }

    #[test]
    fn test_roundtrip() {
        let dims = [2, 3, 2];
        let total: usize = dims.iter().product();
        for i in 0..total {
            let multi = linear_to_multi(i, &dims);
            assert_eq!(multi_to_linear(&multi, &dims), i);
        }
    }
}
