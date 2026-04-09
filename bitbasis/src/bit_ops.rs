/// Return integer with bit k set. 0-indexed.
#[inline]
pub fn indicator(k: usize) -> usize {
    1usize << k
}

/// Return mask with bits at all given positions set. 0-indexed.
#[inline]
pub fn bmask(positions: &[usize]) -> usize {
    positions.iter().fold(0, |acc, &k| acc | indicator(k))
}

/// Return mask for contiguous range [start, stop). 0-indexed.
#[inline]
pub fn bmask_range(start: usize, stop: usize) -> usize {
    if stop <= start {
        return 0;
    }

    let width = stop - start;
    low_mask(width) << start
}

/// Read bit at position loc. 0-indexed. Returns 0 or 1.
#[inline]
pub fn readbit(x: usize, loc: usize) -> usize {
    (x >> loc) & 1
}

/// Set bits at masked positions to 1.
#[inline]
pub fn setbit(x: usize, mask: usize) -> usize {
    x | mask
}

/// Flip bits at masked positions.
#[inline]
pub fn flip(index: usize, mask: usize) -> usize {
    index ^ mask
}

/// True if any masked bit is 1.
#[inline]
pub fn anyone(index: usize, mask: usize) -> bool {
    (index & mask) != 0
}

/// True if all masked bits are 1.
#[inline]
pub fn allone(index: usize, mask: usize) -> bool {
    (index & mask) == mask
}

/// True if masked bits equal target.
#[inline]
pub fn ismatch(index: usize, mask: usize, target: usize) -> bool {
    (index & mask) == target
}

/// Swap bits at positions i and j. 0-indexed.
#[inline]
pub fn swapbits(b: usize, i: usize, j: usize) -> usize {
    let mask = indicator(i) | indicator(j);
    let masked = b & mask;

    if masked != 0 && masked != mask {
        b ^ mask
    } else {
        b
    }
}

/// Reverse bit order for nbits.
pub fn breflect(mut b: usize, nbits: usize) -> usize {
    for i in 0..(nbits / 2) {
        b = swapbits(b, i, nbits - 1 - i);
    }
    b
}

/// Flip all bits within nbits width.
#[inline]
pub fn neg(b: usize, nbits: usize) -> usize {
    low_mask(nbits) ^ b
}

/// Truncate to n bits.
#[inline]
pub fn btruncate(b: usize, n: usize) -> usize {
    b & low_mask(n)
}

/// Get 0-indexed positions of set bits.
pub fn baddrs(mut b: usize) -> Vec<usize> {
    let mut locs = Vec::with_capacity(b.count_ones() as usize);
    while b != 0 {
        let loc = b.trailing_zeros() as usize;
        locs.push(loc);
        b &= b - 1;
    }
    locs
}

#[inline]
fn low_mask(width: usize) -> usize {
    if width == 0 {
        0
    } else if width >= usize::BITS as usize {
        usize::MAX
    } else {
        (1usize << width) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicator() {
        assert_eq!(indicator(0), 1);
        assert_eq!(indicator(3), 8);
    }

    #[test]
    fn test_bmask() {
        assert_eq!(bmask(&[0, 2]), 0b101);
        assert_eq!(bmask(&[]), 0);
    }

    #[test]
    fn test_bmask_range() {
        assert_eq!(bmask_range(1, 4), 0b1110);
    }

    #[test]
    fn test_readbit() {
        assert_eq!(readbit(0b1010, 1), 1);
        assert_eq!(readbit(0b1010, 0), 0);
    }

    #[test]
    fn test_flip() {
        assert_eq!(flip(0b1010, 0b1100), 0b0110);
    }

    #[test]
    fn test_setbit() {
        assert_eq!(setbit(0b0010, 0b1001), 0b1011);
    }

    #[test]
    fn test_anyone() {
        assert!(anyone(0b1011, 0b1001));
        assert!(!anyone(0b1011, 0b0100));
    }

    #[test]
    fn test_allone() {
        assert!(allone(0b1011, 0b1001));
        assert!(!allone(0b1011, 0b1101));
    }

    #[test]
    fn test_ismatch() {
        assert!(ismatch(0b11001, 0b10100, 0b10000));
    }

    #[test]
    fn test_swapbits() {
        assert_eq!(swapbits(0b1011, 1, 2), 0b1101);
        assert_eq!(swapbits(0b1010, 1, 3), 0b1010);
    }

    #[test]
    fn test_breflect() {
        assert_eq!(breflect(0b1011, 4), 0b1101);
    }

    #[test]
    fn test_neg() {
        assert_eq!(neg(0b1111, 4), 0b0000);
        assert_eq!(neg(0b0111, 4), 0b1000);
    }

    #[test]
    fn test_btruncate() {
        assert_eq!(btruncate(0b101101, 3), 0b101);
        assert_eq!(btruncate(0b101101, 0), 0);
    }

    #[test]
    fn test_baddrs() {
        assert_eq!(baddrs(0b1010), vec![1, 3]);
    }
}
