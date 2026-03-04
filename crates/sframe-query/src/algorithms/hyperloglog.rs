//! HyperLogLog sketch for approximate cardinality estimation.
//!
//! Standard HyperLogLog algorithm (Flajolet et al. 2007) with bias
//! correction for small and large cardinalities.
//!
//! Memory usage: 2^precision bytes (default precision=12 → 4 KB, ~1.6% error).

use std::hash::{Hash, Hasher};

use sframe_types::flex_type::FlexType;

/// HyperLogLog cardinality sketch.
#[derive(Clone)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    p: u8,
    m: usize,
    count: usize,
}

impl HyperLogLog {
    /// Create a new HyperLogLog with the given precision.
    ///
    /// Precision must be in [4, 18]. Higher precision = more memory but lower error.
    /// - p=10: 1 KB, ~3.25% error
    /// - p=12: 4 KB, ~1.63% error (default)
    /// - p=14: 16 KB, ~0.81% error
    /// - p=16: 64 KB, ~0.41% error
    pub fn new(precision: u8) -> Self {
        assert!(
            (4..=18).contains(&precision),
            "HyperLogLog precision must be in [4, 18], got {}",
            precision
        );
        let m = 1usize << precision;
        HyperLogLog {
            registers: vec![0u8; m],
            p: precision,
            m,
            count: 0,
        }
    }

    /// Insert a value into the sketch.
    pub fn insert(&mut self, value: &FlexType) {
        let hash = hash_value(value);
        let idx = (hash >> (64 - self.p)) as usize;
        let remaining = if self.p >= 64 {
            0u64
        } else {
            (hash << self.p) | (1u64 << (self.p - 1))
        };
        let zeros = remaining.leading_zeros() as u8 + 1;
        if zeros > self.registers[idx] {
            self.registers[idx] = zeros;
        }
        self.count += 1;
    }

    /// Return the estimated cardinality.
    pub fn estimate(&self) -> u64 {
        let m = self.m as f64;

        // Harmonic mean of 2^(-register)
        let sum: f64 = self.registers.iter().map(|&r| 2.0f64.powi(-(r as i32))).sum();
        let alpha = alpha_m(self.m);
        let raw = alpha * m * m / sum;

        if raw <= 2.5 * m {
            // Small range correction: count zero registers
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                (m * (m / zeros as f64).ln()) as u64
            } else {
                raw as u64
            }
        } else if raw > (1u64 << 32) as f64 / 30.0 {
            // Large range correction (only needed for 32-bit hash; we use 64-bit)
            raw as u64
        } else {
            raw as u64
        }
    }

    /// Merge another HyperLogLog into this one (register-wise max).
    ///
    /// Both must have the same precision.
    pub fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(
            self.p, other.p,
            "Cannot merge HyperLogLog with different precisions ({} vs {})",
            self.p, other.p
        );
        for (a, &b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if b > *a {
                *a = b;
            }
        }
        self.count += other.count;
    }

    /// Total number of items inserted.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Alpha constant for bias correction.
fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

/// Hash a FlexType value to u64 using the standard Hash trait.
fn hash_value(value: &FlexType) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_sketch() {
        let hll = HyperLogLog::new(12);
        assert_eq!(hll.estimate(), 0);
        assert_eq!(hll.count(), 0);
    }

    #[test]
    fn test_single_value() {
        let mut hll = HyperLogLog::new(12);
        hll.insert(&FlexType::Integer(42));
        assert!(hll.estimate() >= 1);
        assert_eq!(hll.count(), 1);
    }

    #[test]
    fn test_duplicate_values() {
        let mut hll = HyperLogLog::new(12);
        for _ in 0..1000 {
            hll.insert(&FlexType::Integer(42));
        }
        // Should estimate ~1 despite 1000 insertions (same value)
        let est = hll.estimate();
        assert!(est <= 3, "Expected ~1 for duplicates, got {}", est);
    }

    #[test]
    fn test_known_cardinality() {
        let mut hll = HyperLogLog::new(14); // higher precision for tighter bound
        let n = 10_000;
        for i in 0..n {
            hll.insert(&FlexType::Integer(i));
        }
        let est = hll.estimate();
        // With p=14, error should be < 2%
        let error = (est as f64 - n as f64).abs() / n as f64;
        assert!(
            error < 0.05,
            "Estimate {} too far from {} (error {:.1}%)",
            est,
            n,
            error * 100.0
        );
    }

    #[test]
    fn test_string_values() {
        let mut hll = HyperLogLog::new(12);
        for i in 0..5000 {
            hll.insert(&FlexType::String(format!("item_{}", i).into()));
        }
        let est = hll.estimate();
        let error = (est as f64 - 5000.0).abs() / 5000.0;
        assert!(
            error < 0.10,
            "Estimate {} too far from 5000 (error {:.1}%)",
            est,
            error * 100.0
        );
    }

    #[test]
    fn test_merge() {
        let mut hll1 = HyperLogLog::new(12);
        let mut hll2 = HyperLogLog::new(12);
        for i in 0..5000 {
            hll1.insert(&FlexType::Integer(i));
        }
        for i in 3000..8000 {
            hll2.insert(&FlexType::Integer(i));
        }
        hll1.merge(&hll2);
        // True cardinality: 8000 (0..8000)
        let est = hll1.estimate();
        let error = (est as f64 - 8000.0).abs() / 8000.0;
        assert!(
            error < 0.10,
            "Merged estimate {} too far from 8000 (error {:.1}%)",
            est,
            error * 100.0
        );
    }

    #[test]
    fn test_mixed_types() {
        let mut hll = HyperLogLog::new(12);
        hll.insert(&FlexType::Integer(1));
        hll.insert(&FlexType::Float(2.0));
        hll.insert(&FlexType::String("three".into()));
        hll.insert(&FlexType::Undefined);
        let est = hll.estimate();
        assert!(est >= 2 && est <= 8, "Estimate {} out of range for 4 distinct values", est);
    }
}
