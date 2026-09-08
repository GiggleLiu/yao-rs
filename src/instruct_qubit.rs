//! Qubit (d=2) instruct functions using bit-manipulation for zero-alloc hot loops.
//!
//! Ported from Julia YaoArrayRegister:
//! `~/.julia/dev/Yao/lib/YaoArrayRegister/src/instruct.jl`
//!
//! **Location convention:** All `loc` parameters use yao-rs convention where
//! loc 0 = most significant qubit. Internally converted to bit positions
//! (LSB=0) for stride calculations.

use bitbasis::{controller, indicator, itercontrol};
use num_complex::Complex64;

/// Convert yao-rs qubit location (MSB-first, 0-indexed) to bit position (LSB-first).
#[inline]
fn loc_to_bit(nbits: usize, loc: usize) -> usize {
    nbits - 1 - loc
}

/// Apply 2x2 unitary `[[a,b],[c,d]]` to amplitudes at indices i and j.
///
/// Julia: `u1rows!(state, i, j, a, b, c, d)` from utils.jl:110-116
#[inline]
fn u1rows(
    state: &mut [Complex64],
    i: usize,
    j: usize,
    a: Complex64,
    b: Complex64,
    c: Complex64,
    d: Complex64,
) {
    let w = state[i];
    let v = state[j];
    state[i] = a * w + b * v;
    state[j] = c * w + d * v;
}

#[inline(always)]
fn for_each_single_control_base(
    total: usize,
    target_bit: usize,
    ctrl_bit: usize,
    ctrl_val: usize,
    mut f: impl FnMut(usize),
) {
    debug_assert!(ctrl_val <= 1);
    debug_assert_ne!(target_bit, ctrl_bit);

    if target_bit < ctrl_bit {
        let low_span = 1usize << target_bit;
        let middle_iters = 1usize << (ctrl_bit - target_bit - 1);
        let outer_step = 1usize << (ctrl_bit + 1);
        let ctrl_offset = ctrl_val << ctrl_bit;

        for outer_base in (0..total).step_by(outer_step) {
            let ctrl_base = outer_base + ctrl_offset;
            for middle in 0..middle_iters {
                let middle_base = ctrl_base + (middle << (target_bit + 1));
                for low in 0..low_span {
                    f(middle_base + low);
                }
            }
        }
    } else {
        let low_span = 1usize << ctrl_bit;
        let middle_iters = 1usize << (target_bit - ctrl_bit - 1);
        let outer_step = 1usize << (target_bit + 1);
        let ctrl_offset = ctrl_val << ctrl_bit;

        for outer_base in (0..total).step_by(outer_step) {
            for middle in 0..middle_iters {
                let middle_base = outer_base + (middle << (ctrl_bit + 1)) + ctrl_offset;
                for low in 0..low_span {
                    f(middle_base + low);
                }
            }
        }
    }
}

/// X gate: swap amplitudes at |...0...> and |...1...>.
pub fn instruct_x(state: &mut [Complex64], nbits: usize, loc: usize) {
    let bit = loc_to_bit(nbits, loc);
    let step = 1usize << bit;
    for block in state.chunks_exact_mut(step * 2) {
        let (low, high) = block.split_at_mut(step);
        for (low, high) in low.iter_mut().zip(high) {
            std::mem::swap(low, high);
        }
    }
}

/// Controlled X gate.
pub fn instruct_x_controlled(
    state: &mut [Complex64],
    nbits: usize,
    loc: usize,
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    let bit = loc_to_bit(nbits, loc);
    let mask = indicator(bit);
    let dim = 1usize << nbits;

    if ctrl_locs.len() == 1 {
        let ctrl_bit = loc_to_bit(nbits, ctrl_locs[0]);
        let ctrl_val = ctrl_bits[0];
        for_each_single_control_base(dim, bit, ctrl_bit, ctrl_val, |base| {
            state.swap(base, base + mask);
        });
        return;
    }

    let ctrl_bit_positions: Vec<usize> = ctrl_locs
        .iter()
        .map(|&loc| loc_to_bit(nbits, loc))
        .collect();
    let ctrl = controller(&ctrl_bit_positions, ctrl_bits);

    for basis in 0..dim {
        if basis & mask == 0 && ctrl(basis) {
            state.swap(basis, basis ^ mask);
        }
    }
}

// ========================================================================
// 1-qubit instruct (no controls)
// ========================================================================

/// Apply single-qubit gate to state vector using stride-based iteration.
///
/// `loc` is in yao-rs convention (0 = MSB).
///
/// Julia: `single_qubit_instruct!(state, U1, loc)` + `instruct_kernel`
/// from `instruct.jl:153-166`
pub fn instruct_1q(
    state: &mut [Complex64],
    loc: usize,
    a: Complex64,
    b: Complex64,
    c: Complex64,
    d: Complex64,
) {
    let nbits = state.len().trailing_zeros() as usize;
    let bit = loc_to_bit(nbits, loc);
    let step1 = 1 << bit;
    let step2 = 1 << (bit + 1);
    if [a.im, b.im, c.im, d.im].iter().all(|&v| v == 0.0) {
        // Real matrices act independently on real and imaginary components.
        // Disjoint halves let LLVM vectorize both contiguous streams.
        for block in state.chunks_exact_mut(step2) {
            let (low, high) = block.split_at_mut(step1);
            for (low, high) in low.iter_mut().zip(high) {
                let w = *low;
                let v = *high;
                *low = w * a.re + v * b.re;
                *high = w * c.re + v * d.re;
            }
        }
    } else if [a.im, d.im, b.re, c.re].iter().all(|&v| v == 0.0) {
        // Rx and Y have real diagonal and imaginary off-diagonal coefficients.
        for block in state.chunks_exact_mut(step2) {
            let (low, high) = block.split_at_mut(step1);
            for (low, high) in low.iter_mut().zip(high) {
                let w = *low;
                let v = *high;
                *low = Complex64::new(a.re * w.re - b.im * v.im, a.re * w.im + b.im * v.re);
                *high = Complex64::new(d.re * v.re - c.im * w.im, d.re * v.im + c.im * w.re);
            }
        }
    } else {
        for block in state.chunks_exact_mut(step2) {
            let (low, high) = block.split_at_mut(step1);
            for (low, high) in low.iter_mut().zip(high) {
                let w = *low;
                let v = *high;
                *low = a * w + b * v;
                *high = c * w + d * v;
            }
        }
    }
}

// ========================================================================
// 1-qubit diagonal instruct (no controls)
// ========================================================================

/// Apply single-qubit diagonal gate diag(d0, d1) using stride-based iteration.
///
/// `loc` is in yao-rs convention (0 = MSB).
///
/// Julia: `single_qubit_instruct!(state, U1::SDDiagonal, loc)`
/// from `instruct.jl:187-198`
pub fn instruct_1q_diag(state: &mut [Complex64], loc: usize, d0: Complex64, d1: Complex64) {
    let nbits = state.len().trailing_zeros() as usize;
    let bit = loc_to_bit(nbits, loc);
    let step1 = 1 << bit;
    let step2 = 1 << (bit + 1);
    if d0 == Complex64::new(1., 0.) {
        for block in state.chunks_exact_mut(step2) {
            for value in &mut block[step1..] {
                *value *= d1;
            }
        }
        return;
    }
    for block in state.chunks_exact_mut(step2) {
        let (low, high) = block.split_at_mut(step1);
        for value in low {
            *value *= d0;
        }
        for value in high {
            *value *= d1;
        }
    }
}

/// Recognize the block structure shared by exchange gates and custom matrices.
fn single_excitation_matrix(gate: &[Complex64]) -> bool {
    [1, 2, 3, 4, 7, 8, 11, 12, 13, 14]
        .iter()
        .all(|&i| gate[i] == Complex64::new(0.0, 0.0))
}

/// Apply d-dimensional unitary to amplitudes at given indices (stack-allocated for d<=4).
///
/// Julia: `unrows!(state, inds, U)` from utils.jl:176-183
fn unrows(state: &mut [Complex64], inds: &[usize], gate: &[Complex64], single_excitation: bool) {
    if single_excitation {
        state[inds[0]] *= gate[0];
        u1rows(state, inds[1], inds[2], gate[5], gate[6], gate[9], gate[10]);
        state[inds[3]] *= gate[15];
        return;
    }
    let d = inds.len();
    debug_assert!(
        d <= 4,
        "unrows: stack buffer supports at most d=4, got d={d}"
    );
    let mut old = [Complex64::new(0.0, 0.0); 4];
    for (k, &idx) in inds.iter().enumerate() {
        old[k] = state[idx];
    }
    for i in 0..d {
        let mut acc = Complex64::new(0.0, 0.0);
        for j in 0..d {
            acc += gate[i * d + j] * old[j];
        }
        state[inds[i]] = acc;
    }
}

// ========================================================================
// 2-qubit instruct (no controls)
// ========================================================================

fn two_qubit_offsets(bits: [usize; 2]) -> [usize; 4] {
    let low = 1usize << bits[0];
    let high = 1usize << bits[1];
    [0, low, high, low | high]
}

/// Permute 4x4 gate matrix when bit positions are not in ascending order.
///
/// Julia: `sort_unitary(Val(2), U, locs)` from utils.jl:16-23
fn sort_unitary_2q(gate: &[Complex64], bits: &[usize]) -> ([Complex64; 16], [usize; 2]) {
    let mut sorted = [bits[0], bits[1]];
    if sorted[0] < sorted[1] {
        return (std::array::from_fn(|i| gate[i]), sorted);
    }
    sorted.sort();
    // Swap qubit order: permute |ij⟩ → |ji⟩
    // perm = [0, 2, 1, 3] maps: 00→00, 01→10, 10→01, 11→11
    let perm = [0usize, 2, 1, 3];
    let mut new_gate = [Complex64::new(0.0, 0.0); 16];
    for i in 0..4 {
        for j in 0..4 {
            new_gate[perm[i] * 4 + perm[j]] = gate[i * 4 + j];
        }
    }
    (new_gate, sorted)
}

/// Apply a 2-qubit 4x4 gate using bit-stride iteration.
///
/// `locs` are in yao-rs convention (0 = MSB). Internally converted to bit positions.
///
/// Julia: generic `instruct!(Val(2), state, operator, locs)` from `instruct.jl:90-121`
pub fn instruct_2q(state: &mut [Complex64], nbits: usize, locs: &[usize], gate: &[Complex64]) {
    debug_assert_eq!(locs.len(), 2);
    debug_assert_eq!(gate.len(), 16);

    let bits = [loc_to_bit(nbits, locs[0]), loc_to_bit(nbits, locs[1])];
    let (gate, sorted_bits) = sort_unitary_2q(gate, &bits);
    let single_excitation = single_excitation_matrix(&gate);

    let locs_raw = two_qubit_offsets(sorted_bits);

    // Both target bits are zero in each block's base index.
    for_each_single_control_base(1usize << nbits, sorted_bits[0], sorted_bits[1], 0, |base| {
        let inds = [
            locs_raw[0] + base,
            locs_raw[1] + base,
            locs_raw[2] + base,
            locs_raw[3] + base,
        ];
        unrows(state, &inds, &gate, single_excitation);
    });
}

/// Apply 2-qubit diagonal gate diag(d0, d1, d2, d3).
///
/// `locs` are in yao-rs convention (0 = MSB).
pub fn instruct_2q_diag(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    diag: &[Complex64; 4],
) {
    debug_assert_eq!(locs.len(), 2);

    let bits = [loc_to_bit(nbits, locs[0]), loc_to_bit(nbits, locs[1])];
    let mut sorted_bits = [bits[0], bits[1]];
    let diag = if sorted_bits[0] > sorted_bits[1] {
        sorted_bits.sort();
        // Swap qubit order in diagonal: |01⟩ ↔ |10⟩
        [diag[0], diag[2], diag[1], diag[3]]
    } else {
        *diag
    };

    let locs_raw = two_qubit_offsets(sorted_bits);

    for_each_single_control_base(1usize << nbits, sorted_bits[0], sorted_bits[1], 0, |base| {
        for k in 0..4 {
            state[locs_raw[k] + base] *= diag[k];
        }
    });
}

/// SWAP gate: exchange amplitudes when the target bits differ.
pub fn instruct_swap(state: &mut [Complex64], nbits: usize, locs: &[usize]) {
    debug_assert_eq!(locs.len(), 2);

    let bit0 = loc_to_bit(nbits, locs[0]);
    let bit1 = loc_to_bit(nbits, locs[1]);
    let swap_mask = indicator(bit0) | indicator(bit1);
    for_each_single_control_base(1usize << nbits, bit0, bit1, 1, |base| {
        state.swap(base, base ^ swap_mask);
    });
}

/// Generic n-qubit gate application with optional controls.
pub fn instruct_nq(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    gate: &[Complex64],
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    let nlocs = locs.len();
    let gate_dim = 1usize << nlocs;
    debug_assert_eq!(gate.len(), gate_dim * gate_dim);

    let target_bits: Vec<usize> = locs.iter().map(|&loc| loc_to_bit(nbits, loc)).collect();
    let ctrl_bit_positions: Vec<usize> = ctrl_locs
        .iter()
        .map(|&loc| loc_to_bit(nbits, loc))
        .collect();

    let mut locked_positions = ctrl_bit_positions;
    locked_positions.extend_from_slice(&target_bits);
    let mut locked_values = ctrl_bits.to_vec();
    locked_values.extend(std::iter::repeat_n(0usize, nlocs));

    let raw_offsets: Vec<usize> = (0..gate_dim)
        .map(|config| {
            target_bits
                .iter()
                .enumerate()
                .fold(0usize, |acc, (idx, &bit)| {
                    acc | (((config >> idx) & 1) << bit)
                })
        })
        .collect();

    let mut temp = vec![Complex64::new(0.0, 0.0); gate_dim];
    let mut indices = vec![0usize; gate_dim];
    for base in itercontrol(nbits, &locked_positions, &locked_values) {
        for (index, &offset) in indices.iter_mut().zip(&raw_offsets) {
            *index = base + offset;
        }

        for row in 0..gate_dim {
            temp[row] = Complex64::new(0.0, 0.0);
            for col in 0..gate_dim {
                temp[row] += gate[row * gate_dim + col] * state[indices[col]];
            }
        }

        for (idx, value) in indices.iter().zip(temp.iter()) {
            state[*idx] = *value;
        }
    }
}

// ========================================================================
// Controlled versions
// ========================================================================

/// Apply controlled single-qubit gate.
///
/// `loc` and `ctrl_locs` are in yao-rs convention (0 = MSB).
///
/// Julia: `instruct!(Val(2), state, operator, locs, control_locs, control_bits)`
/// from `instruct.jl:90-121`
#[allow(clippy::too_many_arguments)]
pub fn instruct_1q_controlled(
    state: &mut [Complex64],
    nbits: usize,
    loc: usize,
    a: Complex64,
    b: Complex64,
    c: Complex64,
    d: Complex64,
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    let target_bit = loc_to_bit(nbits, loc);
    let step = 1 << target_bit;
    let total = 1usize << nbits;

    if ctrl_locs.len() == 1 {
        let ctrl_bit = loc_to_bit(nbits, ctrl_locs[0]);
        let ctrl_val = ctrl_bits[0];
        if [a.im, b.im, c.im, d.im].iter().all(|&v| v == 0.0) {
            for_each_single_control_base(total, target_bit, ctrl_bit, ctrl_val, |base| {
                let w = state[base];
                let v = state[base + step];
                state[base] = w * a.re + v * b.re;
                state[base + step] = w * c.re + v * d.re;
            });
        } else {
            for_each_single_control_base(total, target_bit, ctrl_bit, ctrl_val, |base| {
                u1rows(state, base, base + step, a, b, c, d);
            });
        }
        return;
    }

    // Convert control locs to bit positions
    let ctrl_bit_positions: Vec<usize> = ctrl_locs.iter().map(|&l| loc_to_bit(nbits, l)).collect();

    // Lock control bits + target bit (target locked to 0)
    let mut locked_locs: Vec<usize> = ctrl_bit_positions;
    locked_locs.push(target_bit);
    let mut locked_vals: Vec<usize> = ctrl_bits.to_vec();
    locked_vals.push(0); // target locked to 0

    let ic = itercontrol(nbits, &locked_locs, &locked_vals);

    for base in ic {
        u1rows(state, base, base + step, a, b, c, d);
    }
}

/// Apply controlled 2-qubit gate.
///
/// `locs` and `ctrl_locs` are in yao-rs convention (0 = MSB).
pub fn instruct_2q_controlled(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    gate: &[Complex64],
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    debug_assert_eq!(locs.len(), 2);
    debug_assert_eq!(gate.len(), 16);

    let bits = [loc_to_bit(nbits, locs[0]), loc_to_bit(nbits, locs[1])];
    let (gate, sorted_bits) = sort_unitary_2q(gate, &bits);
    let single_excitation = single_excitation_matrix(&gate);

    let locs_raw = two_qubit_offsets(sorted_bits);

    // Convert control locs to bit positions
    let ctrl_bit_positions: Vec<usize> = ctrl_locs.iter().map(|&l| loc_to_bit(nbits, l)).collect();

    // Lock controls + targets (targets to 0)
    let mut locked_locs: Vec<usize> = ctrl_bit_positions;
    locked_locs.extend_from_slice(&sorted_bits);
    let mut locked_vals: Vec<usize> = ctrl_bits.to_vec();
    locked_vals.extend_from_slice(&[0, 0]);

    let ic = itercontrol(nbits, &locked_locs, &locked_vals);

    for base in ic {
        let inds = [
            locs_raw[0] + base,
            locs_raw[1] + base,
            locs_raw[2] + base,
            locs_raw[3] + base,
        ];
        unrows(state, &inds, &gate, single_excitation);
    }
}

/// Apply controlled 1-qubit diagonal gate.
///
/// `loc` and `ctrl_locs` are in yao-rs convention (0 = MSB).
pub fn instruct_1q_diag_controlled(
    state: &mut [Complex64],
    nbits: usize,
    loc: usize,
    d0: Complex64,
    d1: Complex64,
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    let target_bit = loc_to_bit(nbits, loc);
    let step = 1 << target_bit;
    let total = 1usize << nbits;

    if ctrl_locs.len() == 1 {
        let ctrl_bit = loc_to_bit(nbits, ctrl_locs[0]);
        let ctrl_val = ctrl_bits[0];
        if d0 == Complex64::new(1., 0.) {
            for_each_single_control_base(total, target_bit, ctrl_bit, ctrl_val, |base| {
                state[base + step] *= d1;
            });
        } else {
            for_each_single_control_base(total, target_bit, ctrl_bit, ctrl_val, |base| {
                state[base] *= d0;
                state[base + step] *= d1;
            });
        }
        return;
    }

    let ctrl_bit_positions: Vec<usize> = ctrl_locs.iter().map(|&l| loc_to_bit(nbits, l)).collect();

    let mut locked_locs: Vec<usize> = ctrl_bit_positions;
    locked_locs.push(target_bit);
    let mut locked_vals: Vec<usize> = ctrl_bits.to_vec();
    locked_vals.push(0);

    let ic = itercontrol(nbits, &locked_locs, &locked_vals);

    if d0 == Complex64::new(1., 0.) {
        for base in ic {
            state[base + step] *= d1;
        }
    } else {
        for base in ic {
            state[base] *= d0;
            state[base + step] *= d1;
        }
    }
}

/// Apply controlled 2-qubit diagonal gate.
///
/// `locs` and `ctrl_locs` are in yao-rs convention (0 = MSB).
pub fn instruct_2q_diag_controlled(
    state: &mut [Complex64],
    nbits: usize,
    locs: &[usize],
    diag: &[Complex64; 4],
    ctrl_locs: &[usize],
    ctrl_bits: &[usize],
) {
    debug_assert_eq!(locs.len(), 2);

    let bits = [loc_to_bit(nbits, locs[0]), loc_to_bit(nbits, locs[1])];
    let mut sorted_bits = [bits[0], bits[1]];
    let diag = if sorted_bits[0] > sorted_bits[1] {
        sorted_bits.sort();
        [diag[0], diag[2], diag[1], diag[3]]
    } else {
        *diag
    };

    let locs_raw = two_qubit_offsets(sorted_bits);

    let ctrl_bit_positions: Vec<usize> = ctrl_locs.iter().map(|&l| loc_to_bit(nbits, l)).collect();

    let mut locked_locs: Vec<usize> = ctrl_bit_positions;
    locked_locs.extend_from_slice(&sorted_bits);
    let mut locked_vals: Vec<usize> = ctrl_bits.to_vec();
    locked_vals.extend_from_slice(&[0, 0]);

    let ic = itercontrol(nbits, &locked_locs, &locked_vals);

    for base in ic {
        for k in 0..4 {
            state[locs_raw[k] + base] *= diag[k];
        }
    }
}

#[cfg(test)]
#[path = "unit_tests/instruct_qubit.rs"]
mod tests;
