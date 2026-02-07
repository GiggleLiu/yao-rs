//! Mixed-radix indexing utilities for qudit support.
//!
//! This module provides utilities for converting between multi-indices (site indices)
//! and flat state vector indices using row-major ordering.

/// Convert site indices to flat state vector index using row-major ordering.
///
/// The index is computed as:
/// index = indices[0]*d_1*d_2*... + indices[1]*d_2*... + ... + indices[n-1]
///
/// # Example
/// ```
/// use yao_rs::index::mixed_radix_index;
/// // dims=[2,3,2] means qubit-qutrit-qubit system
/// // indices=[1,2,0] → 1×(3×2) + 2×2 + 0 = 6 + 4 + 0 = 10
/// assert_eq!(mixed_radix_index(&[1, 2, 0], &[2, 3, 2]), 10);
/// ```
pub fn mixed_radix_index(indices: &[usize], dims: &[usize]) -> usize {
    debug_assert_eq!(
        indices.len(),
        dims.len(),
        "indices and dims must have the same length"
    );
    let mut index = 0usize;
    for (i, &idx) in indices.iter().enumerate() {
        let stride: usize = dims[i + 1..].iter().product();
        index += idx * stride;
    }
    index
}

/// Decompose a flat index into site indices using row-major ordering.
///
/// # Example
/// ```
/// use yao_rs::index::linear_to_indices;
/// // dims=[2,3,2], index=10 → [1, 2, 0]
/// assert_eq!(linear_to_indices(10, &[2, 3, 2]), vec![1, 2, 0]);
/// ```
pub fn linear_to_indices(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut multi = vec![0usize; n];
    for i in (0..n).rev() {
        multi[i] = index % dims[i];
        index /= dims[i];
    }
    multi
}

/// Iterate over all computational basis states.
///
/// Yields (flat_index, site_indices) pairs for all basis states.
/// Total count = product of dims.
///
/// # Example
/// ```
/// use yao_rs::index::iter_basis;
/// let states: Vec<_> = iter_basis(&[2, 2]).collect();
/// assert_eq!(states.len(), 4);
/// assert_eq!(states[0], (0, vec![0, 0]));
/// assert_eq!(states[3], (3, vec![1, 1]));
/// ```
pub fn iter_basis(dims: &[usize]) -> impl Iterator<Item = (usize, Vec<usize>)> + '_ {
    let total: usize = dims.iter().product();
    (0..total).map(move |i| (i, linear_to_indices(i, dims)))
}

/// Iterate over basis states with fixed values at certain sites.
///
/// Used for controlled gates: only apply when controls have specific values.
/// Yields only the flat indices where the fixed sites have the specified values.
///
/// # Arguments
/// * `dims` - Dimensions of all sites
/// * `fixed_locs` - Indices of sites that should have fixed values
/// * `fixed_vals` - Values that the fixed sites should have
///
/// # Example
/// ```
/// use yao_rs::index::iter_basis_fixed;
/// // dims=[2,2], fix site 0 to value 1
/// let indices: Vec<_> = iter_basis_fixed(&[2, 2], &[0], &[1]).collect();
/// // Only states |10> and |11> match (flat indices 2 and 3)
/// assert_eq!(indices, vec![2, 3]);
/// ```
pub fn iter_basis_fixed<'a>(
    dims: &'a [usize],
    fixed_locs: &'a [usize],
    fixed_vals: &'a [usize],
) -> impl Iterator<Item = usize> + 'a {
    debug_assert_eq!(
        fixed_locs.len(),
        fixed_vals.len(),
        "fixed_locs and fixed_vals must have the same length"
    );

    let total: usize = dims.iter().product();
    (0..total).filter(move |&i| {
        let indices = linear_to_indices(i, dims);
        fixed_locs
            .iter()
            .zip(fixed_vals.iter())
            .all(|(&loc, &val)| indices[loc] == val)
    })
}

/// Insert a value at a location in site indices.
///
/// Creates a new vector with `val` inserted at position `loc`,
/// shifting elements at `loc` and beyond to the right.
///
/// # Example
/// ```
/// use yao_rs::index::insert_index;
/// // other_basis=[1,0], loc=1, val=2 → [1,2,0]
/// assert_eq!(insert_index(&[1, 0], 1, 2), vec![1, 2, 0]);
/// ```
pub fn insert_index(other_basis: &[usize], loc: usize, val: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(other_basis.len() + 1);
    result.extend_from_slice(&other_basis[..loc]);
    result.push(val);
    result.extend_from_slice(&other_basis[loc..]);
    result
}
