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

/// Iterator for controlled subspace of bits.
///
/// Efficiently iterates through basis states where specified bit positions
/// are locked to given values. Only "free" (unlocked) bits vary.
///
/// Ported from Julia BitBasis: `~/.julia/dev/BitBasis/src/iterate_control.jl`
///
/// # Example
/// To iterate states matching pattern `x0x10x1` (0-indexed positions 0,2,3,6):
/// ```
/// use yao_rs::bitutils::itercontrol;
/// let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
/// // Yields: 9, 11, 25, 27, 41, 43, 57, 59
/// ```
pub struct IterControl {
    n: usize,            // number of free configurations = 2^(nbits - nlocked)
    base: usize,         // base offset from locked bits set to 1
    masks: Vec<usize>,   // masks for each chunk of free bits
    factors: Vec<usize>, // shift factors for each chunk
    current: usize,      // iteration counter
}

impl IterControl {
    /// Total number of states this iterator will yield.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Random access: get the k-th element (0-indexed).
    ///
    /// Julia: `getindex(it, k) = sum((k & mask) * factor for (mask, factor)) + base`
    /// (Julia is 1-indexed, so uses k-1)
    pub fn get(&self, k: usize) -> usize {
        let mut out = 0usize;
        for (&mask, &factor) in self.masks.iter().zip(self.factors.iter()) {
            out += (k & mask) * factor;
        }
        out + self.base
    }
}

impl Iterator for IterControl {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.current >= self.n {
            return None;
        }
        let val = self.get(self.current);
        self.current += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IterControl {}

/// Create an IterControl that iterates through the controlled subspace.
///
/// `nbits`: total number of qubits
/// `positions`: 0-indexed bit positions to lock (sorted internally)
/// `bit_configs`: value (0 or 1) to lock each position to
///
/// Julia: `itercontrol(nbits, positions, bit_configs)` from
/// `~/.julia/dev/BitBasis/src/iterate_control.jl:53-59`
pub fn itercontrol(nbits: usize, positions: &[usize], bit_configs: &[usize]) -> IterControl {
    debug_assert_eq!(positions.len(), bit_configs.len());
    debug_assert!(bit_configs.iter().all(|&b| b == 0 || b == 1));

    // base = bitmask of positions where config == 1
    let base: usize = positions
        .iter()
        .zip(bit_configs.iter())
        .filter(|pair| *pair.1 == 1)
        .map(|pair| indicator(*pair.0))
        .sum();

    let mut sorted_positions: Vec<usize> = positions.to_vec();
    let (masks, factors) = group_shift(nbits, &mut sorted_positions);

    let n = 1 << (nbits - positions.len());
    IterControl {
        n,
        base,
        masks,
        factors,
        current: 0,
    }
}

/// Compute masks and shift factors for IterControl.
///
/// Groups the free (unlocked) bits into contiguous chunks and computes
/// the bitmask and multiplication factor for each chunk.
///
/// Julia: `group_shift!(nbits, positions)` from
/// `~/.julia/dev/BitBasis/src/iterate_control.jl:108-130`
///
/// NOTE: Julia uses 1-indexed positions. This function uses 0-indexed.
/// The algorithm converts to 1-indexed internally to match Julia exactly.
pub fn group_shift(nbits: usize, positions: &mut [usize]) -> (Vec<usize>, Vec<usize>) {
    positions.sort();

    let mut masks = Vec::new();
    let mut factors = Vec::new();

    // Convert to 1-indexed to match Julia algorithm exactly
    let positions_1: Vec<usize> = positions.iter().map(|&p| p + 1).collect();

    let mut k_prv: usize = 0;
    let mut i: usize = 0;

    for &k in &positions_1 {
        debug_assert!(k > k_prv, "Conflict at location: {}", k);
        if k != k_prv + 1 {
            factors.push(1 << (k_prv - i));
            let gap = k - k_prv - 1;
            // Julia: bmask(i+1:i+gap) → Rust: bmask_range(i, i+gap)
            masks.push(bmask_range(i, i + gap));
            i += gap;
        }
        k_prv = k;
    }

    // The last block: handle remaining free bits after last locked position
    let nfree = nbits - positions.len();
    if i != nfree {
        factors.push(1 << (k_prv - i));
        masks.push(bmask_range(i, nfree));
    }

    (masks, factors)
}
