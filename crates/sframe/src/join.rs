//! Hash join building blocks.
//!
//! `JoinHashTable` maps composite keys (one or more columns) from a "build"
//! batch to row indices. The probe side can then look up matching rows in
//! parallel via rayon, collecting matched pairs and optionally tracking
//! unmatched rows on both sides.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use sframe_query::batch::SFrameRows;
use sframe_types::flex_type::FlexType;

/// Result of probing the hash table with a batch of probe rows.
pub struct ProbeResult {
    /// Build-side row indices of matched pairs (parallel with `matched_probe`).
    pub matched_build: Vec<usize>,
    /// Probe-side row indices of matched pairs (parallel with `matched_build`).
    pub matched_probe: Vec<usize>,
    /// Probe-side row indices that found no match in the build table.
    pub unmatched_probe: Vec<usize>,
}

/// A hash table built from the key columns of a build-side `SFrameRows` batch.
///
/// Maps composite key values to the list of build-row indices sharing that key.
/// Supports parallel probing and optional tracking of which build rows were
/// ever matched (for full/right outer joins).
pub struct JoinHashTable {
    /// key -> list of build row indices with that key.
    map: HashMap<Vec<FlexType>, Vec<usize>>,
    /// One flag per build row; set to `true` when matched during probe.
    /// Only allocated when `track_matched` is true.
    matched: Vec<AtomicBool>,
}

impl JoinHashTable {
    /// Build a hash table from `build`, hashing the columns at `key_cols`.
    ///
    /// If `track_matched` is true, an `AtomicBool` array is allocated so that
    /// `unmatched_build_indices` can later report build rows that were never
    /// matched by any probe call.
    pub fn new(build: &SFrameRows, key_cols: &[usize], track_matched: bool) -> Self {
        let n = build.num_rows();
        let mut map: HashMap<Vec<FlexType>, Vec<usize>> = HashMap::new();

        for row_idx in 0..n {
            let key: Vec<FlexType> = key_cols
                .iter()
                .map(|&c| build.column(c).get(row_idx))
                .collect();
            map.entry(key).or_default().push(row_idx);
        }

        let matched = if track_matched {
            (0..n).map(|_| AtomicBool::new(false)).collect()
        } else {
            Vec::new()
        };

        JoinHashTable { map, matched }
    }

    /// Probe the hash table in parallel with a batch of probe rows.
    ///
    /// Each rayon chunk extracts keys from `probe_key_cols`, looks them up in
    /// the shared (read-only) hash map, and collects `(build_idx, probe_idx)`
    /// matched pairs. If `collect_unmatched` is true, probe rows with no match
    /// are also collected.
    ///
    /// When `track_matched` was enabled at construction, every matched build
    /// row has its flag set (via `AtomicBool`), so `unmatched_build_indices`
    /// can later report the complement.
    pub fn probe_parallel(
        &self,
        probe: &SFrameRows,
        probe_key_cols: &[usize],
        collect_unmatched: bool,
    ) -> ProbeResult {
        let n = probe.num_rows();
        let track = !self.matched.is_empty();

        // Determine chunk size: at least 1, aim for ~rayon thread count chunks.
        let num_threads = rayon::current_num_threads().max(1);
        let chunk_size = (n / num_threads).max(64).min(n.max(1));

        // Each chunk produces local vectors to avoid contention.
        struct ChunkResult {
            matched_build: Vec<usize>,
            matched_probe: Vec<usize>,
            unmatched_probe: Vec<usize>,
        }

        let chunks: Vec<ChunkResult> = (0..n)
            .into_par_iter()
            .with_min_len(chunk_size)
            .fold(
                || ChunkResult {
                    matched_build: Vec::new(),
                    matched_probe: Vec::new(),
                    unmatched_probe: Vec::new(),
                },
                |mut acc, probe_idx| {
                    let key: Vec<FlexType> = probe_key_cols
                        .iter()
                        .map(|&c| probe.column(c).get(probe_idx))
                        .collect();

                    if let Some(build_indices) = self.map.get(&key) {
                        for &build_idx in build_indices {
                            acc.matched_build.push(build_idx);
                            acc.matched_probe.push(probe_idx);
                            if track {
                                self.matched[build_idx].store(true, Ordering::Relaxed);
                            }
                        }
                    } else if collect_unmatched {
                        acc.unmatched_probe.push(probe_idx);
                    }

                    acc
                },
            )
            .collect();

        // Merge chunk results.
        let total_matched: usize = chunks.iter().map(|c| c.matched_build.len()).sum();
        let total_unmatched: usize = chunks.iter().map(|c| c.unmatched_probe.len()).sum();

        let mut matched_build = Vec::with_capacity(total_matched);
        let mut matched_probe = Vec::with_capacity(total_matched);
        let mut unmatched_probe = Vec::with_capacity(total_unmatched);

        for chunk in chunks {
            matched_build.extend(chunk.matched_build);
            matched_probe.extend(chunk.matched_probe);
            unmatched_probe.extend(chunk.unmatched_probe);
        }

        ProbeResult {
            matched_build,
            matched_probe,
            unmatched_probe,
        }
    }

    /// Returns the indices of build rows that were never matched by any probe.
    ///
    /// Only meaningful when the table was constructed with `track_matched = true`;
    /// otherwise returns an empty vector.
    pub fn unmatched_build_indices(&self) -> Vec<usize> {
        self.matched
            .iter()
            .enumerate()
            .filter_map(|(i, flag)| {
                if !flag.load(Ordering::Relaxed) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sframe_types::flex_type::FlexTypeEnum;

    #[test]
    fn test_hash_table_build_and_probe() {
        // Build side: id = 1, 2, 3
        let build_rows = vec![
            vec![FlexType::Integer(1), FlexType::String("a".into())],
            vec![FlexType::Integer(2), FlexType::String("b".into())],
            vec![FlexType::Integer(3), FlexType::String("c".into())],
        ];
        let build_dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let build = SFrameRows::from_rows(&build_rows, &build_dtypes).unwrap();

        // Probe side: id = 2, 3, 4
        let probe_rows = vec![
            vec![FlexType::Integer(2), FlexType::String("x".into())],
            vec![FlexType::Integer(3), FlexType::String("y".into())],
            vec![FlexType::Integer(4), FlexType::String("z".into())],
        ];
        let probe_dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let probe = SFrameRows::from_rows(&probe_rows, &probe_dtypes).unwrap();

        // Build hash table on column 0 (the id column), tracking matched rows.
        let ht = JoinHashTable::new(&build, &[0], true);

        // Probe with column 0.
        let result = ht.probe_parallel(&probe, &[0], true);

        // Should have 2 matches: build id=2 <-> probe id=2, build id=3 <-> probe id=3.
        assert_eq!(result.matched_build.len(), 2);
        assert_eq!(result.matched_probe.len(), 2);

        // Verify matched pairs (order may vary due to parallelism, so collect as sets).
        let mut pairs: Vec<(usize, usize)> = result
            .matched_build
            .iter()
            .zip(result.matched_probe.iter())
            .map(|(&b, &p)| (b, p))
            .collect();
        pairs.sort();
        // build idx 1 (id=2) matched with probe idx 0 (id=2)
        // build idx 2 (id=3) matched with probe idx 1 (id=3)
        assert_eq!(pairs, vec![(1, 0), (2, 1)]);

        // 1 unmatched probe row: probe idx 2 (id=4).
        assert_eq!(result.unmatched_probe, vec![2]);

        // 1 unmatched build row: build idx 0 (id=1).
        let unmatched_build = ht.unmatched_build_indices();
        assert_eq!(unmatched_build, vec![0]);
    }
}
