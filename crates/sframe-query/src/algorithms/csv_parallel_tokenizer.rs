//! Parallel CSV tokenizer using quote-parity bitset for concurrent line splitting.
//!
//! The key insight: to split a CSV buffer into lines in parallel, each thread
//! needs to know whether a given newline is "real" (a row boundary) or "quoted"
//! (inside a quoted field). The [`DenseBitset`] tracks quote parity — bit *i*
//! is set when byte position *i* is inside a quoted field — so that parallel
//! workers can classify newlines without scanning from the start of the buffer.

/// Compact bitset -- 1 bit per byte position in a CSV buffer.
/// Used to track quote parity: bit set means "inside a quoted field".
pub(crate) struct DenseBitset {
    bits: Vec<u64>,
    /// Number of addressable bit positions.
    len: usize,
}

impl DenseBitset {
    /// Allocate a zeroed bitset that can hold `len` bits.
    pub fn new(len: usize) -> Self {
        let num_words = (len + 63) / 64;
        DenseBitset {
            bits: vec![0u64; num_words],
            len,
        }
    }

    /// Set the bit at `idx`.
    #[inline(always)]
    pub fn set(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    /// Clear the bit at `idx`.
    #[inline(always)]
    pub fn clear_bit(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] &= !(1u64 << bit);
    }

    /// Test whether the bit at `idx` is set.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len);
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] >> bit) & 1 != 0
    }

    /// Set the bit at `idx` to `val`.
    pub fn set_to(&mut self, idx: usize, val: bool) {
        debug_assert!(idx < self.len);
        if val {
            self.set(idx);
        } else {
            self.clear_bit(idx);
        }
    }

    /// Zero all bits, keeping the current allocation and length.
    pub fn clear_all(&mut self) {
        self.bits.fill(0);
    }

    /// Resize to `len` bits and zero everything.
    /// Reuses the existing heap allocation when possible.
    pub fn resize_and_clear(&mut self, len: usize) {
        let num_words = (len + 63) / 64;
        self.bits.clear();
        self.bits.resize(num_words, 0);
        self.len = len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_basic() {
        let mut bs = DenseBitset::new(256);

        // All bits start as 0.
        for i in [0, 1, 63, 64, 65, 127, 128, 199, 255] {
            assert!(!bs.get(i), "bit {i} should start unset");
        }

        // Set specific positions including word boundaries.
        for i in [0, 63, 64, 199] {
            bs.set(i);
        }

        // Only the set positions should read as true.
        assert!(bs.get(0));
        assert!(bs.get(63));
        assert!(bs.get(64));
        assert!(bs.get(199));

        // Neighbors should still be unset.
        assert!(!bs.get(1));
        assert!(!bs.get(62));
        assert!(!bs.get(65));
        assert!(!bs.get(198));
        assert!(!bs.get(200));
    }

    #[test]
    fn test_bitset_clear() {
        let mut bs = DenseBitset::new(128);

        bs.set(10);
        bs.set(64);
        assert!(bs.get(10));
        assert!(bs.get(64));

        bs.clear_bit(10);
        assert!(!bs.get(10));
        assert!(bs.get(64)); // other bit unaffected

        bs.clear_bit(64);
        assert!(!bs.get(64));
    }

    #[test]
    fn test_bitset_clear_all() {
        let mut bs = DenseBitset::new(200);

        // Set a spread of bits across multiple words.
        for i in [0, 33, 64, 100, 127, 150, 199] {
            bs.set(i);
        }
        // Sanity: they are set.
        for i in [0, 33, 64, 100, 127, 150, 199] {
            assert!(bs.get(i));
        }

        bs.clear_all();

        // Every bit should be zero now.
        for i in 0..200 {
            assert!(!bs.get(i), "bit {i} should be cleared");
        }
    }

    #[test]
    fn test_bitset_set_to() {
        let mut bs = DenseBitset::new(128);

        // set_to true
        bs.set_to(42, true);
        assert!(bs.get(42));

        // set_to false
        bs.set_to(42, false);
        assert!(!bs.get(42));

        // set_to true then true again (idempotent)
        bs.set_to(99, true);
        bs.set_to(99, true);
        assert!(bs.get(99));

        // set_to false on an already-unset bit (idempotent)
        bs.set_to(10, false);
        assert!(!bs.get(10));
    }
}
