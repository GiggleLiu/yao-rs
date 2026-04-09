use crate::bit_ops::{bmask, bmask_range, indicator};
use smallvec::SmallVec;

/// Inline chunk capacity for the common case.
const INLINE_CHUNKS: usize = 8;

/// Iterator over controlled subspace of bits.
/// Efficiently enumerates basis states with fixed control bits and free others.
///
/// Uses inline storage for the common case and spills to heap only for
/// unusually fragmented control layouts.
pub struct IterControl {
    n: usize,
    base: usize,
    masks: SmallVec<[usize; INLINE_CHUNKS]>,
    factors: SmallVec<[usize; INLINE_CHUNKS]>,
    current: usize,
}

impl IterControl {
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline(always)]
    pub fn get(&self, k: usize) -> usize {
        match self.masks.len() {
            0 => self.base,
            1 => (k & self.masks[0]) * self.factors[0] + self.base,
            2 => {
                (k & self.masks[0]) * self.factors[0]
                    + (k & self.masks[1]) * self.factors[1]
                    + self.base
            }
            3 => {
                (k & self.masks[0]) * self.factors[0]
                    + (k & self.masks[1]) * self.factors[1]
                    + (k & self.masks[2]) * self.factors[2]
                    + self.base
            }
            _ => {
                let mut out = self.base;
                for (&mask, &factor) in self.masks.iter().zip(self.factors.iter()) {
                    out += (k & mask) * factor;
                }
                out
            }
        }
    }
}

impl Iterator for IterControl {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.current >= self.n {
            return None;
        }

        let value = self.get(self.current);
        self.current += 1;
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IterControl {}

#[inline]
pub fn itercontrol(nbits: usize, positions: &[usize], bit_configs: &[usize]) -> IterControl {
    assert_eq!(positions.len(), bit_configs.len());
    assert!(positions.iter().all(|&position| position < nbits));
    assert!(
        bit_configs.iter().all(|&bit| bit == 0 || bit == 1),
        "Bit configurations must be 0 or 1"
    );

    let base = positions
        .iter()
        .zip(bit_configs.iter())
        .filter(|&(_, &value)| value == 1)
        .fold(0usize, |acc, (&position, _)| acc | indicator(position));

    let mut sorted_positions = positions.to_vec();
    let (masks, factors) = group_shift(nbits, &mut sorted_positions);

    IterControl {
        n: 1usize << (nbits - positions.len()),
        base,
        masks,
        factors,
        current: 0,
    }
}

pub fn group_shift(
    nbits: usize,
    positions: &mut [usize],
) -> (
    SmallVec<[usize; INLINE_CHUNKS]>,
    SmallVec<[usize; INLINE_CHUNKS]>,
) {
    positions.sort_unstable();

    let mut masks = SmallVec::<[usize; INLINE_CHUNKS]>::new();
    let mut factors = SmallVec::<[usize; INLINE_CHUNKS]>::new();

    let mut previous = 0usize;
    let mut free_index = 0usize;

    for &position in positions.iter() {
        let position = position + 1;
        assert!(position > previous, "Duplicate position");
        if position != previous + 1 {
            factors.push(1usize << (previous - free_index));
            let gap = position - previous - 1;
            masks.push(bmask_range(free_index, free_index + gap));
            free_index += gap;
        }
        previous = position;
    }

    let nfree = nbits - positions.len();
    if free_index != nfree {
        factors.push(1usize << (previous - free_index));
        masks.push(bmask_range(free_index, nfree));
    }

    (masks, factors)
}

pub fn controller(cbits: &[usize], cvals: &[usize]) -> impl Fn(usize) -> bool {
    assert_eq!(cbits.len(), cvals.len());
    assert!(
        cvals.iter().all(|&bit| bit == 0 || bit == 1),
        "Control values must be 0 or 1"
    );

    let mask = bmask(cbits);
    let target = cbits
        .iter()
        .zip(cvals.iter())
        .filter(|&(_, &value)| value == 1)
        .fold(0usize, |acc, (&position, _)| acc | indicator(position));

    move |basis| (basis & mask) == target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_itercontrol_basic() {
        let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
        let values: Vec<usize> = ic.collect();
        assert_eq!(values, vec![9, 11, 25, 27, 41, 43, 57, 59]);
    }

    #[test]
    fn test_itercontrol_len() {
        let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
        assert_eq!(ic.len(), 8);
    }

    #[test]
    fn test_itercontrol_get() {
        let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
        assert_eq!(ic.get(0), 9);
        assert_eq!(ic.get(7), 59);
    }

    #[test]
    fn test_itercontrol_all_free() {
        let ic = itercontrol(3, &[], &[]);
        let values: Vec<usize> = ic.collect();
        assert_eq!(values, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_itercontrol_all_locked() {
        let ic = itercontrol(2, &[0, 1], &[1, 0]);
        let values: Vec<usize> = ic.collect();
        assert_eq!(values, vec![1]);
    }

    #[test]
    fn test_itercontrol_more_than_eight_chunks() {
        let positions = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
        let bit_configs = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let values: Vec<usize> = itercontrol(19, &positions, &bit_configs).collect();

        assert_eq!(values.len(), 1usize << (19 - positions.len()));
        for basis in values {
            for (&position, &bit) in positions.iter().zip(bit_configs.iter()) {
                assert_eq!((basis >> position) & 1, bit);
            }
        }
    }

    #[test]
    fn test_controller() {
        let ctrl = controller(&[0, 2], &[1, 0]);
        assert!(ctrl(0b001));
        assert!(!ctrl(0b101));
        assert!(!ctrl(0b000));
    }
}
