//! Space-Saving algorithm for finding frequent items (heavy hitters).
//!
//! Metwally, Agrawal, El Abbadi (2005). Maintains a bounded set of
//! counters that tracks items with frequency > N/capacity. Uses O(capacity)
//! memory regardless of the number of distinct items.

use std::collections::HashMap;

use sframe_types::flex_type::FlexType;

/// Space-Saving frequent items sketch.
#[derive(Clone)]
pub struct SpaceSaving {
    /// Tracked items and their counts, ordered by count descending.
    items: Vec<FlexType>,
    counts: Vec<u64>,
    /// Fast lookup: item → index in items/counts vecs.
    index: HashMap<FlexType, usize>,
    capacity: usize,
    total: usize,
}

impl SpaceSaving {
    /// Create a new Space-Saving sketch with the given capacity.
    ///
    /// Capacity determines accuracy: any item with true frequency > N/capacity
    /// is guaranteed to be in the result. Typical values: 100-1000.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SpaceSaving capacity must be > 0");
        SpaceSaving {
            items: Vec::with_capacity(capacity),
            counts: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            capacity,
            total: 0,
        }
    }

    /// Insert a value.
    pub fn insert(&mut self, value: &FlexType) {
        self.total += 1;

        // Case 1: item already tracked — increment its count
        if let Some(&idx) = self.index.get(value) {
            self.counts[idx] += 1;
            // Bubble up to maintain sorted order
            self.bubble_up(idx);
            return;
        }

        // Case 2: room for more items — add it
        if self.items.len() < self.capacity {
            let idx = self.items.len();
            self.items.push(value.clone());
            self.counts.push(1);
            self.index.insert(value.clone(), idx);
            self.bubble_up(idx);
            return;
        }

        // Case 3: at capacity — replace the item with the smallest count
        let last = self.items.len() - 1;
        let min_count = self.counts[last];
        self.index.remove(&self.items[last]);
        self.items[last] = value.clone();
        self.counts[last] = min_count + 1;
        self.index.insert(value.clone(), last);
        self.bubble_up(last);
    }

    /// Return the top-k items by estimated frequency.
    pub fn top_k(&self, k: usize) -> Vec<(FlexType, u64)> {
        let n = k.min(self.items.len());
        self.items[..n]
            .iter()
            .zip(self.counts[..n].iter())
            .map(|(item, &count)| (item.clone(), count))
            .collect()
    }

    /// Merge another SpaceSaving sketch into this one.
    ///
    /// For each item in `other`, adds its count to this sketch.
    pub fn merge(&mut self, other: &SpaceSaving) {
        for (item, &count) in other.items.iter().zip(other.counts.iter()) {
            for _ in 0..count {
                self.insert(item);
            }
        }
    }

    /// Total number of items inserted.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of currently tracked items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Bubble an item up to maintain descending count order.
    fn bubble_up(&mut self, mut idx: usize) {
        while idx > 0 && self.counts[idx] > self.counts[idx - 1] {
            // Swap with predecessor
            self.items.swap(idx, idx - 1);
            self.counts.swap(idx, idx - 1);
            // Update index map
            self.index.insert(self.items[idx].clone(), idx);
            self.index.insert(self.items[idx - 1].clone(), idx - 1);
            idx -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let ss = SpaceSaving::new(10);
        assert_eq!(ss.total(), 0);
        assert_eq!(ss.len(), 0);
        assert!(ss.top_k(5).is_empty());
    }

    #[test]
    fn test_single_item() {
        let mut ss = SpaceSaving::new(10);
        ss.insert(&FlexType::Integer(42));
        assert_eq!(ss.total(), 1);
        assert_eq!(ss.len(), 1);
        let top = ss.top_k(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, FlexType::Integer(42));
        assert_eq!(top[0].1, 1);
    }

    #[test]
    fn test_frequency_ordering() {
        let mut ss = SpaceSaving::new(10);
        // "a" appears 5 times, "b" appears 3 times, "c" appears 1 time
        for _ in 0..5 {
            ss.insert(&FlexType::String("a".into()));
        }
        for _ in 0..3 {
            ss.insert(&FlexType::String("b".into()));
        }
        ss.insert(&FlexType::String("c".into()));

        let top = ss.top_k(3);
        assert_eq!(top[0].0, FlexType::String("a".into()));
        assert_eq!(top[0].1, 5);
        assert_eq!(top[1].0, FlexType::String("b".into()));
        assert_eq!(top[1].1, 3);
        assert_eq!(top[2].0, FlexType::String("c".into()));
        assert_eq!(top[2].1, 1);
    }

    #[test]
    fn test_eviction() {
        let mut ss = SpaceSaving::new(3);
        // Insert 5 distinct items with capacity 3
        ss.insert(&FlexType::Integer(1));
        ss.insert(&FlexType::Integer(2));
        ss.insert(&FlexType::Integer(3));
        ss.insert(&FlexType::Integer(4)); // evicts the least frequent
        ss.insert(&FlexType::Integer(5)); // evicts again

        assert_eq!(ss.len(), 3);
        assert_eq!(ss.total(), 5);
    }

    #[test]
    fn test_heavy_hitter_detection() {
        let mut ss = SpaceSaving::new(10);
        // "hot" appears 100 times, 99 other items appear once each
        for _ in 0..100 {
            ss.insert(&FlexType::String("hot".into()));
        }
        for i in 0..99 {
            ss.insert(&FlexType::Integer(i));
        }

        let top = ss.top_k(1);
        assert_eq!(top[0].0, FlexType::String("hot".into()));
        assert_eq!(top[0].1, 100);
    }

    #[test]
    fn test_top_k_capped() {
        let mut ss = SpaceSaving::new(5);
        for i in 0..3 {
            ss.insert(&FlexType::Integer(i));
        }
        // Requesting more than available returns all
        let top = ss.top_k(10);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_merge() {
        let mut ss1 = SpaceSaving::new(10);
        let mut ss2 = SpaceSaving::new(10);

        for _ in 0..50 {
            ss1.insert(&FlexType::String("a".into()));
        }
        for _ in 0..30 {
            ss2.insert(&FlexType::String("a".into()));
        }
        for _ in 0..20 {
            ss2.insert(&FlexType::String("b".into()));
        }

        ss1.merge(&ss2);
        let top = ss1.top_k(2);
        assert_eq!(top[0].0, FlexType::String("a".into()));
        assert_eq!(top[0].1, 80); // 50 + 30
    }
}
