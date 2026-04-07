use num_complex::Complex64;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BitStr<const N: usize>(pub u64);

impl<const N: usize> BitStr<N> {
    pub fn new(val: u64) -> Self {
        BitStr(val & mask::<N>())
    }

    pub fn bint(self) -> u64 {
        self.0
    }

    pub fn nbits() -> usize {
        N
    }

    pub fn readbit(self, loc: usize) -> u64 {
        (self.0 >> loc) & 1
    }

    pub fn onehot(self) -> Vec<Complex64> {
        let dim = 1usize << N;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[self.0 as usize] = Complex64::new(1.0, 0.0);
        state
    }
}

impl<const N: usize> fmt::Display for BitStr<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for bit in (0..N).rev() {
            write!(f, "{}", (self.0 >> bit) & 1)?;
        }
        write!(f, " ₍₂₎")
    }
}

impl<const N: usize> fmt::Debug for BitStr<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitStr<{N}>({:#0width$b})", self.0, width = N + 2)
    }
}

impl<const N: usize> From<u64> for BitStr<N> {
    fn from(val: u64) -> Self {
        Self::new(val)
    }
}

impl<const N: usize> From<BitStr<N>> for u64 {
    fn from(bitstr: BitStr<N>) -> u64 {
        bitstr.0
    }
}

impl<const N: usize> From<BitStr<N>> for usize {
    fn from(bitstr: BitStr<N>) -> usize {
        bitstr.0 as usize
    }
}

#[inline]
const fn mask<const N: usize>() -> u64 {
    if N == 0 {
        0
    } else if N >= u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << N) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitstr_new_truncates() {
        let b = BitStr::<3>::new(0b11111);
        assert_eq!(b.0, 0b111);
    }

    #[test]
    fn test_bitstr_display() {
        let b = BitStr::<4>::new(0b1010);
        assert_eq!(format!("{b}"), "1010 ₍₂₎");
    }

    #[test]
    fn test_bitstr_readbit() {
        let b = BitStr::<4>::new(0b1010);
        assert_eq!(b.readbit(0), 0);
        assert_eq!(b.readbit(1), 1);
        assert_eq!(b.readbit(3), 1);
    }

    #[test]
    fn test_bitstr_onehot() {
        let b = BitStr::<2>::new(0b10);
        let v = b.onehot();
        assert_eq!(v.len(), 4);
        assert_eq!(v[2], Complex64::new(1.0, 0.0));
        assert_eq!(v[0], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_bitstr_metadata_and_conversions() {
        let b = BitStr::<5>::new(0b1_1010);
        assert_eq!(BitStr::<5>::nbits(), 5);
        assert_eq!(b.bint(), 0b1_1010);
        assert_eq!(u64::from(b), 0b1_1010);
        assert_eq!(usize::from(b), 0b1_1010);
    }
}
