// tests/bitutils.rs
use yao_rs::bitutils::*;

#[test]
fn test_indicator() {
    assert_eq!(indicator(0), 1); // bit 0
    assert_eq!(indicator(1), 2); // bit 1
    assert_eq!(indicator(3), 8); // bit 3
}

#[test]
fn test_bmask_single() {
    assert_eq!(bmask(&[0]), 0b1);
    assert_eq!(bmask(&[1]), 0b10);
    assert_eq!(bmask(&[0, 1]), 0b11);
    assert_eq!(bmask(&[0, 2]), 0b101);
}

#[test]
fn test_bmask_empty() {
    assert_eq!(bmask(&[]), 0);
}

#[test]
fn test_bmask_range() {
    // bmask_range(0, 3) = bits 0,1,2 = 0b111
    assert_eq!(bmask_range(0, 3), 0b111);
    assert_eq!(bmask_range(1, 4), 0b1110);
    assert_eq!(bmask_range(2, 5), 0b11100);
}

#[test]
fn test_flip() {
    assert_eq!(flip(0b1011, 0b1011), 0b0000);
    assert_eq!(flip(0b0000, 0b1010), 0b1010);
    assert_eq!(flip(0b1100, 0b0011), 0b1111);
}

#[test]
fn test_anyone() {
    assert!(anyone(0b1011, 0b1001));
    assert!(anyone(0b1011, 0b1100));
    assert!(!anyone(0b1011, 0b0100));
}

#[test]
fn test_allone() {
    assert!(allone(0b1011, 0b1011));
    assert!(allone(0b1011, 0b1001));
    assert!(!allone(0b1011, 0b0100));
}

#[test]
fn test_ismatch() {
    // ismatch(0b11001, 0b10100, 0b10000) == true
    let n = 0b11001usize;
    let mask = 0b10100usize;
    let target = 0b10000usize;
    assert!(ismatch(n, mask, target));
    assert!(!ismatch(n, mask, 0b00100));
}

#[test]
fn test_group_shift_basic() {
    // positions [1, 3, 4, 7] (Julia 1-indexed)
    // In Rust 0-indexed: [0, 2, 3, 6]
    // nbits=7
    let (masks, factors) = group_shift(7, &mut vec![0, 2, 3, 6]);
    // After sorting: [0, 2, 3, 6]
    // Gaps: before 0 (none), between 0..2 (bit 1), between 3..6 (bits 4,5)
    // After removing locked bits, free bits are: 1, 4, 5
    // Chunks: [bit 1], [bits 4,5]
    assert_eq!(masks.len(), factors.len());
    // The exact values depend on the algorithm; verify via itercontrol below
}

#[test]
fn test_itercontrol_basic() {
    // Julia example: itercontrol(7, [1,3,4,7], (1,0,1,0))
    // iterates: 0001001, 0001011, 0011001, 0011011, 0101001, 0101011, 0111001, 0111011
    //
    // Rust 0-indexed: nbits=7, positions=[0,2,3,6], bits=[1,0,1,0]
    // Pattern: bit0=1, bit2=0, bit3=1, bit6=0 => x0x10x1
    // Free bits: 1, 4, 5
    let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
    let values: Vec<usize> = ic.collect();
    assert_eq!(values.len(), 8); // 2^3 = 8 free bits
    // Expected: 0001001=9, 0001011=11, 0011001=25, 0011011=27,
    //           0101001=41, 0101011=43, 0111001=57, 0111011=59
    assert_eq!(values, vec![9, 11, 25, 27, 41, 43, 57, 59]);
}

#[test]
fn test_itercontrol_no_controls() {
    // No locked bits: iterate all 2^3 = 8 states
    let ic = itercontrol(3, &[], &[]);
    let values: Vec<usize> = ic.collect();
    assert_eq!(values.len(), 8);
    assert_eq!(values, (0..8).collect::<Vec<_>>());
}

#[test]
fn test_itercontrol_all_locked() {
    // Lock all 3 bits to 101 = 5
    let ic = itercontrol(3, &[0, 1, 2], &[1, 0, 1]);
    let values: Vec<usize> = ic.collect();
    assert_eq!(values.len(), 1);
    assert_eq!(values, vec![5]);
}

#[test]
fn test_itercontrol_single_free() {
    // Lock bit 0 to 1, free bit 1. nbits=2
    let ic = itercontrol(2, &[0], &[1]);
    let values: Vec<usize> = ic.collect();
    assert_eq!(values.len(), 2); // 2^1 free
    assert_eq!(values, vec![1, 3]); // 01, 11
}

#[test]
fn test_itercontrol_getindex() {
    let ic = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]);
    // Test random access matches iteration
    let values: Vec<usize> = (0..ic.len()).map(|k| ic.get(k)).collect();
    let iter_values: Vec<usize> = itercontrol(7, &[0, 2, 3, 6], &[1, 0, 1, 0]).collect();
    assert_eq!(values, iter_values);
}
