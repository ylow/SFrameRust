//! Greenwald-Khanna streaming approximate quantile sketch.
//!
//! Maintains a sorted summary of observed values such that any quantile
//! query has error at most `epsilon * N` where N is the number of items
//! inserted. Memory usage is O(1/epsilon * log(epsilon * N)).
//!
//! The sketch explicitly tracks the global min and max so that
//! `query(0.0)` and `query(1.0)` are always exact.
//!
//! Reference: Greenwald & Khanna, "Space-Efficient Online Computation of
//! Quantile Summaries", SIGMOD 2001.

use sframe_types::flex_type::FlexType;

use super::sort::compare_flex_type;

/// A single tuple in the GK summary.
#[derive(Clone, Debug)]
struct GKTuple {
    /// The observed value.
    value: FlexType,
    /// g: the difference in rank between this tuple and its predecessor.
    g: usize,
    /// delta: the maximum possible error in this tuple's rank.
    delta: usize,
}

/// Greenwald-Khanna streaming approximate quantile sketch.
pub struct QuantileSketch {
    tuples: Vec<GKTuple>,
    count: usize,
    epsilon: f64,
    /// Explicitly tracked minimum value (exact).
    min_val: Option<FlexType>,
    /// Explicitly tracked maximum value (exact).
    max_val: Option<FlexType>,
    /// Insert buffer — batch inserts for efficiency.
    buffer: Vec<FlexType>,
    buffer_capacity: usize,
}

impl QuantileSketch {
    /// Create a new sketch with the given error bound.
    ///
    /// The sketch guarantees that `query(q)` returns an element whose rank
    /// is within `epsilon * count()` of the true `q`-quantile rank.
    /// `query(0.0)` and `query(1.0)` are always exact (min and max).
    pub fn new(epsilon: f64) -> Self {
        let buffer_capacity = (1.0 / (2.0 * epsilon)).ceil() as usize;
        QuantileSketch {
            tuples: Vec::new(),
            count: 0,
            epsilon,
            min_val: None,
            max_val: None,
            buffer: Vec::with_capacity(buffer_capacity),
            buffer_capacity: buffer_capacity.max(1),
        }
    }

    /// Insert a value into the sketch.
    pub fn insert(&mut self, value: FlexType) {
        // Track min/max explicitly
        match &self.min_val {
            None => {
                self.min_val = Some(value.clone());
                self.max_val = Some(value.clone());
            }
            Some(cur_min) => {
                if compare_flex_type(&value, cur_min) == std::cmp::Ordering::Less {
                    self.min_val = Some(value.clone());
                }
                if compare_flex_type(&value, self.max_val.as_ref().unwrap())
                    == std::cmp::Ordering::Greater
                {
                    self.max_val = Some(value.clone());
                }
            }
        }

        self.buffer.push(value);
        if self.buffer.len() >= self.buffer_capacity {
            self.flush_buffer();
        }
    }

    /// Flush the internal buffer, inserting all buffered values into the summary.
    fn flush_buffer(&mut self) {
        if self.buffer.is_empty() {
            return;
        }

        // Sort the buffer
        let mut buf = std::mem::take(&mut self.buffer);
        buf.sort_by(|a, b| compare_flex_type(a, b));

        // Insert each buffered value
        for value in buf {
            self.insert_one(value);
        }

        // Compress periodically
        self.compress();
    }

    /// Insert a single value into the summary (core GK algorithm).
    fn insert_one(&mut self, value: FlexType) {
        self.count += 1;

        if self.tuples.is_empty() {
            self.tuples.push(GKTuple {
                value,
                g: 1,
                delta: 0,
            });
            return;
        }

        // Find insertion position (first tuple where value <= tuple.value)
        let pos = self
            .tuples
            .iter()
            .position(|t| compare_flex_type(&value, &t.value) != std::cmp::Ordering::Greater);

        match pos {
            // value is smaller than all existing tuples — insert at front
            Some(0) => {
                self.tuples.insert(
                    0,
                    GKTuple {
                        value,
                        g: 1,
                        delta: 0,
                    },
                );
            }
            // value is larger than all existing tuples — insert at end
            None => {
                self.tuples.push(GKTuple {
                    value,
                    g: 1,
                    delta: 0,
                });
            }
            // Insert in the middle
            Some(i) => {
                let delta = self.band_width().saturating_sub(1);
                self.tuples.insert(
                    i,
                    GKTuple {
                        value,
                        g: 1,
                        delta,
                    },
                );
            }
        }
    }

    /// The maximum band width: floor(2 * epsilon * count).
    fn band_width(&self) -> usize {
        (2.0 * self.epsilon * self.count as f64).floor() as usize
    }

    /// Compress the summary by merging adjacent tuples where possible.
    fn compress(&mut self) {
        if self.tuples.len() < 3 {
            return;
        }

        let threshold = self.band_width();
        let len = self.tuples.len();

        // Walk from right to left (skip first and last)
        let mut i = len - 2;
        while i > 0 {
            let g_sum = self.tuples[i].g + self.tuples[i + 1].g;
            let delta_next = self.tuples[i + 1].delta;
            if g_sum + delta_next <= threshold {
                // Merge tuple[i] into tuple[i+1]
                self.tuples[i + 1].g = g_sum;
                self.tuples.remove(i);
            }
            i -= 1;
        }
    }

    /// Query an approximate quantile.
    ///
    /// Returns the value at approximately the `quantile`-th rank.
    /// `quantile` should be in `[0.0, 1.0]`.
    ///
    /// `query(0.0)` returns the exact minimum, `query(1.0)` returns the
    /// exact maximum. Interior quantiles have error at most `epsilon * N`.
    pub fn query(&self, quantile: f64) -> FlexType {
        let total = self.count + self.buffer.len();
        if total == 0 {
            return FlexType::Undefined;
        }

        // Exact min/max at boundaries
        if quantile <= 0.0 {
            return self.min_val.clone().unwrap();
        }
        if quantile >= 1.0 {
            return self.max_val.clone().unwrap();
        }

        // Handle unflushed buffer via fallback
        if self.tuples.is_empty() {
            // Everything is still in the buffer
            let mut sorted = self.buffer.clone();
            sorted.sort_by(|a, b| compare_flex_type(a, b));
            let idx = ((quantile * (sorted.len() as f64 - 1.0)).round() as usize)
                .min(sorted.len() - 1);
            return sorted[idx].clone();
        }

        // Standard GK query: desired rank (1-based)
        let desired = (quantile * self.count as f64).ceil() as usize;
        let desired = desired.max(1).min(self.count);
        let tolerance = (self.epsilon * self.count as f64).ceil() as usize;

        // Walk tuples, find the one whose cumulative rank is closest to
        // desired while staying within the tolerance window.
        let mut rank = 0usize;
        let mut best_idx = 0usize;
        let mut best_dist = usize::MAX;

        for (i, tuple) in self.tuples.iter().enumerate() {
            rank += tuple.g;
            let dist = rank.abs_diff(desired);
            if dist <= tolerance && dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        self.tuples[best_idx].value.clone()
    }

    /// Produce `num_quantiles - 1` cut points that divide the distribution
    /// into `num_quantiles` roughly equal partitions.
    pub fn quantiles(&self, num_quantiles: usize) -> Vec<FlexType> {
        if num_quantiles <= 1 || self.count() == 0 {
            return Vec::new();
        }
        // Need mutable access to flush
        let sketch = self.flushed_copy();
        (1..num_quantiles)
            .map(|i| {
                let q = i as f64 / num_quantiles as f64;
                sketch.query(q)
            })
            .collect()
    }

    /// Return the total number of items inserted.
    pub fn count(&self) -> usize {
        self.count + self.buffer.len()
    }

    /// Merge another sketch into this one.
    ///
    /// The resulting sketch has error bound max(self.epsilon, other.epsilon)
    /// over the combined count. For best results, use sketches with the same
    /// epsilon.
    pub fn merge(&mut self, other: &QuantileSketch) {
        // Merge min/max
        if let Some(ref other_min) = other.min_val {
            match &self.min_val {
                None => {
                    self.min_val = Some(other_min.clone());
                    self.max_val = other.max_val.clone();
                }
                Some(cur_min) => {
                    if compare_flex_type(other_min, cur_min) == std::cmp::Ordering::Less {
                        self.min_val = Some(other_min.clone());
                    }
                    let other_max = other.max_val.as_ref().unwrap();
                    if compare_flex_type(other_max, self.max_val.as_ref().unwrap())
                        == std::cmp::Ordering::Greater
                    {
                        self.max_val = Some(other_max.clone());
                    }
                }
            }
        }

        // Flush both buffers
        self.flush_buffer();

        // Insert other's buffered values
        for val in &other.buffer {
            self.insert(val.clone());
        }
        self.flush_buffer();

        if other.tuples.is_empty() {
            return;
        }

        if self.tuples.is_empty() {
            self.tuples = other.tuples.clone();
            self.count += other.count;
            return;
        }

        // Merge the two sorted tuple lists
        let mut merged = Vec::with_capacity(self.tuples.len() + other.tuples.len());
        let mut i = 0;
        let mut j = 0;
        while i < self.tuples.len() && j < other.tuples.len() {
            if compare_flex_type(&self.tuples[i].value, &other.tuples[j].value)
                != std::cmp::Ordering::Greater
            {
                merged.push(self.tuples[i].clone());
                i += 1;
            } else {
                merged.push(other.tuples[j].clone());
                j += 1;
            }
        }
        merged.extend_from_slice(&self.tuples[i..]);
        merged.extend_from_slice(&other.tuples[j..]);

        self.tuples = merged;
        self.count += other.count;
        self.compress();
    }

    /// Create a flushed copy for read-only quantile queries.
    fn flushed_copy(&self) -> QuantileSketch {
        let mut copy = QuantileSketch {
            tuples: self.tuples.clone(),
            count: self.count,
            epsilon: self.epsilon,
            min_val: self.min_val.clone(),
            max_val: self.max_val.clone(),
            buffer: Vec::new(),
            buffer_capacity: self.buffer_capacity,
        };
        // Insert buffered values
        for val in &self.buffer {
            copy.insert(val.clone());
        }
        copy.flush_buffer();
        copy
    }

    /// Flush the buffer (must be called before querying for accurate results).
    pub fn finish(&mut self) {
        self.flush_buffer();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_quantiles() {
        let mut sketch = QuantileSketch::new(0.01);
        for i in 0..10000 {
            sketch.insert(FlexType::Integer(i));
        }
        sketch.finish();

        // q=0.0 should be exactly 0 (tracked min)
        assert_eq!(sketch.query(0.0), FlexType::Integer(0));

        // q=0.5 should be near 5000
        let median = sketch.query(0.5);
        match median {
            FlexType::Integer(v) => {
                assert!(
                    (v - 5000).unsigned_abs() < 200,
                    "median off: {} vs 5000",
                    v
                );
            }
            _ => panic!("Expected Integer"),
        }

        // q=1.0 should be exactly 9999 (tracked max)
        assert_eq!(sketch.query(1.0), FlexType::Integer(9999));
    }

    #[test]
    fn test_quantiles_partitions() {
        let mut sketch = QuantileSketch::new(0.01);
        for i in 0..10000 {
            sketch.insert(FlexType::Integer(i));
        }
        sketch.finish();

        let cuts = sketch.quantiles(10);
        assert_eq!(cuts.len(), 9);

        // Each cut point should be roughly at i * 1000
        for (i, cut) in cuts.iter().enumerate() {
            let expected = ((i + 1) * 1000) as i64;
            match cut {
                FlexType::Integer(v) => {
                    assert!(
                        (*v - expected).unsigned_abs() < 200,
                        "cut {} off: {} vs {}",
                        i,
                        v,
                        expected
                    );
                }
                _ => panic!("Expected Integer at cut {}", i),
            }
        }
    }

    #[test]
    fn test_merge() {
        let mut s1 = QuantileSketch::new(0.01);
        let mut s2 = QuantileSketch::new(0.01);

        for i in 0..5000 {
            s1.insert(FlexType::Integer(i));
        }
        for i in 5000..10000 {
            s2.insert(FlexType::Integer(i));
        }
        s1.finish();
        s2.finish();

        s1.merge(&s2);

        assert_eq!(s1.count(), 10000);

        // Exact min/max after merge
        assert_eq!(s1.query(0.0), FlexType::Integer(0));
        assert_eq!(s1.query(1.0), FlexType::Integer(9999));

        let median = s1.query(0.5);
        match median {
            FlexType::Integer(v) => {
                assert!(
                    (v - 5000).unsigned_abs() < 300,
                    "merged median off: {} vs 5000",
                    v
                );
            }
            _ => panic!("Expected Integer"),
        }
    }

    #[test]
    fn test_small_input() {
        let mut sketch = QuantileSketch::new(0.1);
        sketch.insert(FlexType::Integer(5));
        sketch.insert(FlexType::Integer(10));
        sketch.insert(FlexType::Integer(15));
        sketch.finish();

        assert_eq!(sketch.count(), 3);

        let min = sketch.query(0.0);
        assert_eq!(min, FlexType::Integer(5));

        let max = sketch.query(1.0);
        assert_eq!(max, FlexType::Integer(15));
    }

    #[test]
    fn test_single_value() {
        let mut sketch = QuantileSketch::new(0.01);
        sketch.insert(FlexType::Integer(42));
        sketch.finish();

        assert_eq!(sketch.count(), 1);
        assert_eq!(sketch.query(0.0), FlexType::Integer(42));
        assert_eq!(sketch.query(0.5), FlexType::Integer(42));
        assert_eq!(sketch.query(1.0), FlexType::Integer(42));
    }

    #[test]
    fn test_string_values() {
        let mut sketch = QuantileSketch::new(0.05);
        let values = ["apple", "banana", "cherry", "date", "elderberry"];
        for v in &values {
            sketch.insert(FlexType::String((*v).into()));
        }
        sketch.finish();

        let min = sketch.query(0.0);
        assert_eq!(min, FlexType::String("apple".into()));

        let max = sketch.query(1.0);
        assert_eq!(max, FlexType::String("elderberry".into()));
    }

    #[test]
    fn test_float_values() {
        let mut sketch = QuantileSketch::new(0.01);
        for i in 0..1000 {
            sketch.insert(FlexType::Float(i as f64 * 0.1));
        }
        sketch.finish();

        assert_eq!(sketch.query(0.0), FlexType::Float(0.0));
        assert_eq!(sketch.query(1.0), FlexType::Float(99.9));

        let median = sketch.query(0.5);
        match median {
            FlexType::Float(v) => {
                assert!(
                    (v - 50.0).abs() < 3.0,
                    "float median off: {} vs 50.0",
                    v
                );
            }
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_empty_sketch() {
        let sketch = QuantileSketch::new(0.01);
        assert_eq!(sketch.count(), 0);
        assert_eq!(sketch.query(0.5), FlexType::Undefined);
        assert!(sketch.quantiles(10).is_empty());
    }

    #[test]
    fn test_quantiles_edge_cases() {
        let sketch = QuantileSketch::new(0.01);
        // 0 or 1 partitions should return empty
        assert!(sketch.quantiles(0).is_empty());
        assert!(sketch.quantiles(1).is_empty());
    }
}
