//! Bit manipulation utilities for qubit (d=2) simulation.
//!
//! Ported from Julia BitBasis.jl: `~/.julia/dev/BitBasis/src/bit_operations.jl`
//!
//! All bit positions are 0-indexed (unlike Julia's 1-indexed).

/// Return an integer with the k-th bit set to 1 (0-indexed).
///
/// Julia: `indicator(::Type{T}, k::Int) = one(T) << (k-1)` (1-indexed)
/// Rust: 0-indexed, so `1 << k`.
#[inline]
pub fn indicator(k: usize) -> usize {
    1 << k
}

/// Return a bitmask with bits set at the given positions (0-indexed).
///
/// Julia: `bmask(::Type{T}, positions...) = reduce(+, indicator(T, b) for b in itr)`
#[inline]
pub fn bmask(positions: &[usize]) -> usize {
    positions.iter().fold(0, |acc, &k| acc | indicator(k))
}

/// Return a bitmask for a contiguous range [start, stop) of bit positions.
///
/// Julia: `bmask(::Type{T}, range::UnitRange{Int}) = ((1 << (stop-start+1)) - 1) << (start-1)`
#[inline]
pub fn bmask_range(start: usize, stop: usize) -> usize {
    if stop <= start {
        return 0;
    }
    ((1usize << (stop - start)) - 1) << start
}

/// XOR flip bits at masked positions.
///
/// Julia: `flip(index::T, mask::T) = index ⊻ mask`
#[inline]
pub fn flip(index: usize, mask: usize) -> usize {
    index ^ mask
}

/// Return true if any bit at masked position is 1.
///
/// Julia: `anyone(index::T, mask::T) = (index & mask) != zero(T)`
#[inline]
pub fn anyone(index: usize, mask: usize) -> bool {
    (index & mask) != 0
}

/// Return true if all bits at masked positions are 1.
///
/// Julia: `allone(index::T, mask::T) = (index & mask) == mask`
#[inline]
pub fn allone(index: usize, mask: usize) -> bool {
    (index & mask) == mask
}

/// Return true if bits at masked positions equal target.
///
/// Julia: `ismatch(index::T, mask::T, target::T) = (index & mask) == target`
#[inline]
pub fn ismatch(index: usize, mask: usize, target: usize) -> bool {
    (index & mask) == target
}
