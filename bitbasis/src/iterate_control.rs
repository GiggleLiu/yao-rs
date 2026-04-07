use crate::bit_ops::{bmask, bmask_range, indicator};

/// Iterator over controlled subspace of bits.
/// Efficiently enumerates basis states with fixed control bits and free others.
pub struct IterControl {
    n: usize,
    base: usize,
    masks: Vec<usize>,
    factors: Vec<usize>,
    current: usize,
}

impl IterControl {
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn get(&self, k: usize) -> usize {
        let mut out = 0;
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

        let value = self.get(self.current);
        self.current += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IterControl {}

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

pub fn group_shift(nbits: usize, positions: &mut [usize]) -> (Vec<usize>, Vec<usize>) {
    positions.sort_unstable();

    let mut masks = Vec::new();
    let mut factors = Vec::new();
    let positions_1: Vec<usize> = positions.iter().map(|&position| position + 1).collect();

    let mut previous = 0usize;
    let mut free_index = 0usize;

    for &position in &positions_1 {
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
    fn test_controller() {
        let ctrl = controller(&[0, 2], &[1, 0]);
        assert!(ctrl(0b001));
        assert!(!ctrl(0b101));
        assert!(!ctrl(0b000));
    }
}
