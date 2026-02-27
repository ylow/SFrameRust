//! Dense bitset compatible with GraphLab's dense_bitset serialization.
//!
//! Wire format:
//! - len (8 bytes LE): number of bits
//! - arrlen (8 bytes LE): number of u64 words
//! - arrlen * 8 bytes: raw u64 words (little-endian)

use std::io::Read;

use crate::error::Result;
use crate::serialization::read_u64;

/// A dense bitset stored as packed u64 words.
pub struct DenseBitset {
    len: usize,
    words: Vec<u64>,
}

impl DenseBitset {
    /// Deserialize from GraphLab archive format.
    pub fn deserialize(reader: &mut impl Read) -> Result<Self> {
        let len = read_u64(reader)? as usize;
        let arrlen = read_u64(reader)? as usize;
        let mut words = Vec::with_capacity(arrlen);
        for _ in 0..arrlen {
            words.push(read_u64(reader)?);
        }
        Ok(DenseBitset { len, words })
    }

    /// Number of bits in the bitset.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get bit at the given index. Returns false for out-of-range indices.
    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.words[word_idx] >> bit_idx) & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_empty_bitset() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        let bs = DenseBitset::deserialize(&mut cursor).unwrap();
        assert_eq!(bs.len(), 0);
        assert!(bs.is_empty());
    }

    #[test]
    fn test_read_bitset_single_word() {
        let mut data = Vec::new();
        data.extend_from_slice(&8u64.to_le_bytes()); // len = 8 bits
        data.extend_from_slice(&1u64.to_le_bytes()); // arrlen = 1 word
        data.extend_from_slice(&0xAAu64.to_le_bytes()); // 0b10101010
        let mut cursor = Cursor::new(&data);
        let bs = DenseBitset::deserialize(&mut cursor).unwrap();
        assert_eq!(bs.len(), 8);
        assert!(!bs.get(0)); // bit 0 = 0
        assert!(bs.get(1)); // bit 1 = 1
        assert!(!bs.get(2)); // bit 2 = 0
        assert!(bs.get(3)); // bit 3 = 1
        assert!(!bs.get(4));
        assert!(bs.get(5));
        assert!(!bs.get(6));
        assert!(bs.get(7));
    }

    #[test]
    fn test_bitset_out_of_range() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0xFFu64.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        let bs = DenseBitset::deserialize(&mut cursor).unwrap();
        assert!(!bs.get(100));
    }

    #[test]
    fn test_bitset_multi_word() {
        let mut data = Vec::new();
        data.extend_from_slice(&128u64.to_le_bytes()); // 128 bits = 2 words
        data.extend_from_slice(&2u64.to_le_bytes()); // arrlen = 2
        data.extend_from_slice(&0u64.to_le_bytes()); // word 0: all zeros
        data.extend_from_slice(&1u64.to_le_bytes()); // word 1: bit 0 set
        let mut cursor = Cursor::new(&data);
        let bs = DenseBitset::deserialize(&mut cursor).unwrap();
        assert_eq!(bs.len(), 128);
        assert!(!bs.get(0));
        assert!(!bs.get(63));
        assert!(bs.get(64)); // first bit of second word
        assert!(!bs.get(65));
    }
}
