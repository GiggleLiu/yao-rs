use yao_rs::index::{
    insert_index, iter_basis, iter_basis_fixed, linear_to_indices, mixed_radix_index,
};

#[test]
fn test_mixed_radix_index_qubit_qutrit_qubit() {
    // dims=[2,3,2] means qubit-qutrit-qubit system (total 12 states)
    let dims = [2, 3, 2];

    // |0,0,0> -> 0
    assert_eq!(mixed_radix_index(&[0, 0, 0], &dims), 0);

    // |0,0,1> -> 1
    assert_eq!(mixed_radix_index(&[0, 0, 1], &dims), 1);

    // |0,1,0> -> 2
    assert_eq!(mixed_radix_index(&[0, 1, 0], &dims), 2);

    // |0,2,0> -> 4
    assert_eq!(mixed_radix_index(&[0, 2, 0], &dims), 4);

    // |1,0,0> -> 6
    assert_eq!(mixed_radix_index(&[1, 0, 0], &dims), 6);

    // |1,2,0> -> 1x6 + 2x2 + 0 = 10
    assert_eq!(mixed_radix_index(&[1, 2, 0], &dims), 10);

    // |1,2,1> -> 1x6 + 2x2 + 1 = 11 (last state)
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
    // other_basis=[1,0], loc=1, val=2 -> [1,2,0]
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
