//! Space-Saving algorithm for finding frequent items (heavy hitters).
//!
//! Metwally, Agrawal, El Abbadi (2005). Maintains a bounded set of
//! counters that tracks items with frequency > N/capacity. Uses O(capacity)
//! memory regardless of the number of distinct items.
//!
//! Optimizations borrowed from the C++ SFrame implementation:
//! - **No sorted maintenance**: entries are unsorted; sort only at `top_k` time.
//! - **Base-level eviction**: tracks the minimum count and a candidate list
//!   of entries at that count, enabling O(1) amortized eviction.
//! - **Cached last insert**: skips hash lookup on repeated keys (temporal locality).
//! - **Batch counts**: `add(value, count)` avoids per-unit insert overhead.

use std::collections::HashMap;

use sframe_types::flex_type::FlexType;

/// Space-Saving frequent items sketch.
#[derive(Clone)]
pub struct SpaceSaving {
    /// Fixed-capacity entry storage.
    items: Vec<FlexType>,
    counts: Vec<u64>,
    errors: Vec<u64>,
    /// Fast lookup: item → index in items/counts/errors vecs.
    index: HashMap<FlexType, usize>,
    capacity: usize,
    n_entries: usize,
    total: usize,
    /// Minimum count among all tracked entries.
    base_level: u64,
    /// Indices of entries whose count == base_level (lazy, may be stale).
    base_candidates: Vec<usize>,
    /// Cached last-inserted key and its index (temporal locality optimization).
    cached_key: Option<FlexType>,
    cached_idx: usize,
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
            errors: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity * 2),
            capacity,
            n_entries: 0,
            total: 0,
            base_level: 0,
            base_candidates: Vec::with_capacity(capacity),
            cached_key: None,
            cached_idx: 0,
        }
    }

    /// Insert a value with count 1.
    #[inline]
    pub fn insert(&mut self, value: &FlexType) {
        self.add(value, 1);
    }

    /// Insert a value with an arbitrary count (batch increment).
    #[inline]
    pub fn add(&mut self, value: &FlexType, count: u64) {
        self.total += count as usize;

        // Fast path: cached last-inserted key (temporal locality).
        if let Some(ref cached) = self.cached_key {
            if cached == value {
                self.counts[self.cached_idx] += count;
                return;
            }
        }

        // Lookup in hash map.
        if let Some(&idx) = self.index.get(value) {
            self.counts[idx] += count;
            self.cached_key = Some(value.clone());
            self.cached_idx = idx;
            return;
        }

        // Not tracked yet.
        if self.n_entries < self.capacity {
            // Room available — append.
            let idx = self.n_entries;
            self.items.push(value.clone());
            self.counts.push(count);
            self.errors.push(0);
            self.index.insert(value.clone(), idx);
            self.n_entries += 1;
            // Update base_level for the new entry.
            if count < self.base_level || idx == 0 {
                self.base_level = count;
                self.base_candidates.clear();
                self.base_candidates.push(idx);
            } else if count == self.base_level {
                self.base_candidates.push(idx);
            }
            self.cached_key = Some(value.clone());
            self.cached_idx = idx;
        } else {
            // At capacity — evict the entry with the minimum count.
            self.evict_and_replace(value, count);
        }
    }

    /// Evict an entry at base_level and replace it with a new value.
    #[inline(never)]
    fn evict_and_replace(&mut self, value: &FlexType, count: u64) {
        // Lazily trim stale candidates (their counts may have increased).
        while let Some(&idx) = self.base_candidates.last() {
            if self.counts[idx] == self.base_level {
                break;
            }
            self.base_candidates.pop();
        }

        // If no candidates remain, regenerate.
        if self.base_candidates.is_empty() {
            self.regenerate_base_level();
        }

        let victim_idx = self.base_candidates.pop().unwrap();

        // Remove old item from index.
        self.index.remove(&self.items[victim_idx]);

        // Replace in-place.
        let old_count = self.counts[victim_idx];
        self.errors[victim_idx] = old_count;
        self.counts[victim_idx] = old_count + count;
        self.items[victim_idx] = value.clone();
        self.index.insert(value.clone(), victim_idx);

        self.cached_key = Some(value.clone());
        self.cached_idx = victim_idx;
    }

    /// Rebuild the base_level and candidate list by scanning all entries.
    /// O(capacity), but amortized O(1) per insert.
    #[inline(never)]
    fn regenerate_base_level(&mut self) {
        let mut min_count = u64::MAX;
        self.base_candidates.clear();

        for i in 0..self.n_entries {
            let c = self.counts[i];
            if c < min_count {
                min_count = c;
                self.base_candidates.clear();
                self.base_candidates.push(i);
            } else if c == min_count {
                self.base_candidates.push(i);
            }
        }

        self.base_level = if min_count == u64::MAX { 0 } else { min_count };
    }

    /// Return the top-k items by estimated frequency.
    pub fn top_k(&self, k: usize) -> Vec<(FlexType, u64)> {
        let n = k.min(self.n_entries);
        if n == 0 {
            return Vec::new();
        }

        // Collect and sort by count descending (only at query time).
        let mut entries: Vec<(usize, u64)> = (0..self.n_entries)
            .map(|i| (i, self.counts[i]))
            .collect();
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        entries[..n]
            .iter()
            .map(|&(i, count)| (self.items[i].clone(), count))
            .collect()
    }

    /// Merge another SpaceSaving sketch into this one.
    ///
    /// Uses batch count addition instead of per-unit insert.
    pub fn merge(&mut self, other: &SpaceSaving) {
        // Invalidate cache since we're bulk-modifying.
        self.cached_key = None;

        for i in 0..other.n_entries {
            self.add(&other.items[i], other.counts[i]);
        }
    }

    /// Total number of items inserted.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of currently tracked items.
    pub fn len(&self) -> usize {
        self.n_entries
    }

    /// Returns `true` if no items are being tracked.
    pub fn is_empty(&self) -> bool {
        self.n_entries == 0
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

    #[test]
    fn test_cached_insert_speedup() {
        // Verify that repeated inserts of the same key work correctly
        // (exercises the cached insertion path).
        let mut ss = SpaceSaving::new(5);
        for _ in 0..1000 {
            ss.insert(&FlexType::Integer(42));
        }
        for _ in 0..500 {
            ss.insert(&FlexType::Integer(7));
        }
        let top = ss.top_k(2);
        assert_eq!(top[0].0, FlexType::Integer(42));
        assert_eq!(top[0].1, 1000);
        assert_eq!(top[1].0, FlexType::Integer(7));
        assert_eq!(top[1].1, 500);
    }

    #[test]
    fn test_batch_add() {
        let mut ss = SpaceSaving::new(10);
        ss.add(&FlexType::Integer(1), 100);
        ss.add(&FlexType::Integer(2), 50);
        ss.add(&FlexType::Integer(3), 25);

        let top = ss.top_k(3);
        assert_eq!(top[0].1, 100);
        assert_eq!(top[1].1, 50);
        assert_eq!(top[2].1, 25);
    }

    #[test]
    fn test_eviction_replaces_minimum() {
        let mut ss = SpaceSaving::new(3);
        ss.add(&FlexType::Integer(1), 10);
        ss.add(&FlexType::Integer(2), 5);
        ss.add(&FlexType::Integer(3), 1);
        // Capacity full. Insert new item — should evict item with count=1 (Integer(3))
        ss.add(&FlexType::Integer(4), 1);

        assert_eq!(ss.len(), 3);
        // Integer(3) should be gone, Integer(4) should have count = 1 + 1 = 2
        let top = ss.top_k(10);
        let items: Vec<_> = top.iter().map(|(v, _)| v.clone()).collect();
        assert!(!items.contains(&FlexType::Integer(3)));
        assert!(items.contains(&FlexType::Integer(4)));
    }
}
