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
    debug_assert_eq!(indices.len(), dims.len(), "indices and dims must have the same length");
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
    debug_assert_eq!(fixed_locs.len(), fixed_vals.len(), "fixed_locs and fixed_vals must have the same length");

    let total: usize = dims.iter().product();
    (0..total).filter(move |&i| {
        let indices = linear_to_indices(i, dims);
        fixed_locs.iter().zip(fixed_vals.iter()).all(|(&loc, &val)| indices[loc] == val)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_radix_index_qubit_qutrit_qubit() {
        // dims=[2,3,2] means qubit-qutrit-qubit system (total 12 states)
        let dims = [2, 3, 2];

        // |0,0,0> → 0
        assert_eq!(mixed_radix_index(&[0, 0, 0], &dims), 0);

        // |0,0,1> → 1
        assert_eq!(mixed_radix_index(&[0, 0, 1], &dims), 1);

        // |0,1,0> → 2
        assert_eq!(mixed_radix_index(&[0, 1, 0], &dims), 2);

        // |0,2,0> → 4
        assert_eq!(mixed_radix_index(&[0, 2, 0], &dims), 4);

        // |1,0,0> → 6
        assert_eq!(mixed_radix_index(&[1, 0, 0], &dims), 6);

        // |1,2,0> → 1×6 + 2×2 + 0 = 10
        assert_eq!(mixed_radix_index(&[1, 2, 0], &dims), 10);

        // |1,2,1> → 1×6 + 2×2 + 1 = 11 (last state)
        assert_eq!(mixed_radix_index(&[1, 2, 1], &dims), 11);
    }

    #[test]
    fn test_linear_to_indices_qubit_qutrit_qubit() {
        let dims = [2, 3, 2];

        assert_eq!(linear_to_indices(0, &dims), vec![0, 0, 0]);
        assert_eq!(linear_to_indices(1, &dims), vec![0, 0, 1]);
        assert_eq!(linear_to_indices(2, &dims), vec![0, 1, 0]);
        assert_eq!(linear_to_indices(4, &dims), vec![0, 2, 0]);
        assert_eq!(linear_to_indices(6, &dims), vec![1, 0, 0]);
        assert_eq!(linear_to_indices(10, &dims), vec![1, 2, 0]);
        assert_eq!(linear_to_indices(11, &dims), vec![1, 2, 1]);
    }

    #[test]
    fn test_roundtrip_mixed_radix() {
        let dims = [2, 3, 2];
        let total: usize = dims.iter().product();

        for i in 0..total {
            let indices = linear_to_indices(i, &dims);
            assert_eq!(mixed_radix_index(&indices, &dims), i);
        }
    }

    #[test]
    fn test_iter_basis_count() {
        // Product of dims should equal count
        let dims = [2, 3, 2];
        let states: Vec<_> = iter_basis(&dims).collect();
        assert_eq!(states.len(), 12);

        // Two qubits
        let states2: Vec<_> = iter_basis(&[2, 2]).collect();
        assert_eq!(states2.len(), 4);

        // Three qutrits
        let states3: Vec<_> = iter_basis(&[3, 3, 3]).collect();
        assert_eq!(states3.len(), 27);
    }

    #[test]
    fn test_iter_basis_content() {
        let dims = [2, 2];
        let states: Vec<_> = iter_basis(&dims).collect();

        assert_eq!(states[0], (0, vec![0, 0]));
        assert_eq!(states[1], (1, vec![0, 1]));
        assert_eq!(states[2], (2, vec![1, 0]));
        assert_eq!(states[3], (3, vec![1, 1]));
    }

    #[test]
    fn test_iter_basis_fixed_single_site() {
        // dims=[2,2], fix site 0 to value 1
        // States: |00>=0, |01>=1, |10>=2, |11>=3
        // Only |10> and |11> should match
        let indices: Vec<_> = iter_basis_fixed(&[2, 2], &[0], &[1]).collect();
        assert_eq!(indices, vec![2, 3]);
    }

    #[test]
    fn test_iter_basis_fixed_multiple_sites() {
        // dims=[2,3,2], fix site 0=1 and site 2=0
        // Should match states with first qubit=1 and last qubit=0
        let dims = [2, 3, 2];
        let indices: Vec<_> = iter_basis_fixed(&dims, &[0, 2], &[1, 0]).collect();

        // |1,0,0>=6, |1,1,0>=8, |1,2,0>=10
        assert_eq!(indices, vec![6, 8, 10]);
    }

    #[test]
    fn test_iter_basis_fixed_no_fixed() {
        // No fixed sites should yield all states
        let dims = [2, 2];
        let indices: Vec<_> = iter_basis_fixed(&dims, &[], &[]).collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_iter_basis_fixed_all_fixed() {
        // All sites fixed should yield exactly one state
        let dims = [2, 3, 2];
        let indices: Vec<_> = iter_basis_fixed(&dims, &[0, 1, 2], &[1, 2, 0]).collect();
        assert_eq!(indices, vec![10]); // |1,2,0> = 10
    }

    #[test]
    fn test_insert_index_basic() {
        // other_basis=[1,0], loc=1, val=2 → [1,2,0]
        assert_eq!(insert_index(&[1, 0], 1, 2), vec![1, 2, 0]);
    }

    #[test]
    fn test_insert_index_at_start() {
        // Insert at the beginning
        assert_eq!(insert_index(&[1, 2], 0, 0), vec![0, 1, 2]);
    }

    #[test]
    fn test_insert_index_at_end() {
        // Insert at the end
        assert_eq!(insert_index(&[0, 1], 2, 2), vec![0, 1, 2]);
    }

    #[test]
    fn test_insert_index_empty() {
        // Insert into empty array
        assert_eq!(insert_index(&[], 0, 5), vec![5]);
    }

    #[test]
    fn test_insert_index_single() {
        // Insert into single-element array
        assert_eq!(insert_index(&[3], 0, 1), vec![1, 3]);
        assert_eq!(insert_index(&[3], 1, 1), vec![3, 1]);
    }
}
