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
