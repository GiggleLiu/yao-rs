//! Qubit (d=2) instruct functions using bit-manipulation for zero-alloc hot loops.
//!
//! Ported from Julia YaoArrayRegister:
//! `~/.julia/dev/Yao/lib/YaoArrayRegister/src/instruct.jl`
//!
//! **Location convention:** All `loc` parameters use yao-rs convention where
//! loc 0 = most significant qubit. Internally converted to bit positions
//! (LSB=0) for stride calculations.

use num_complex::Complex64;

/// Convert yao-rs qubit location (MSB-first, 0-indexed) to bit position (LSB-first).
#[inline]
fn loc_to_bit(nbits: usize, loc: usize) -> usize {
    nbits - 1 - loc
}

/// Apply 2x2 unitary [[a,b],[c,d]] to amplitudes at indices i and j.
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
    let total = state.len();
    let mut j = 0;
    while j < total {
        for i in j..(j + step1) {
            u1rows(state, i, i + step1, a, b, c, d);
        }
        j += step2;
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
    let total = state.len();
    let mut j = 0;
    while j < total {
        for i in j..(j + step1) {
            state[i] *= d0;
            state[i + step1] *= d1;
        }
        j += step2;
    }
}
