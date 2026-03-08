//! Hash join building blocks.
//!
//! `JoinHashTable` maps composite keys (one or more columns) from a "build"
//! batch to row indices. The probe side can then look up matching rows in
//! parallel via rayon, collecting matched pairs and optionally tracking
//! unmatched rows on both sides.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use sframe_query::algorithms::join::{JoinOn, JoinType};
use sframe_query::batch::{ColumnData, SFrameRows};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

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

/// Create a column of `n` null values for the given type.
///
/// For typed columns (Integer, Float, etc.) this produces `vec![None; n]`.
/// For Undefined/Flexible, it produces `vec![FlexType::Undefined; n]`.
pub fn null_column(dtype: FlexTypeEnum, n: usize) -> ColumnData {
    match dtype {
        FlexTypeEnum::Integer => ColumnData::Integer(vec![None; n]),
        FlexTypeEnum::Float => ColumnData::Float(vec![None; n]),
        FlexTypeEnum::String => ColumnData::String(vec![None; n]),
        FlexTypeEnum::Vector => ColumnData::Vector(vec![None; n]),
        FlexTypeEnum::List => ColumnData::List(vec![None; n]),
        FlexTypeEnum::Dict => ColumnData::Dict(vec![None; n]),
        FlexTypeEnum::DateTime => ColumnData::DateTime(vec![None; n]),
        FlexTypeEnum::Undefined => ColumnData::Flexible(vec![FlexType::Undefined; n]),
    }
}

/// Construct output `SFrameRows` from matched and unmatched index lists.
///
/// Output column order: all left columns, then right non-key columns.
///
/// - `left_to_right_key`: mapping from left key column index to right key column index
///   (derived from `JoinOn.pairs`). For unmatched-right rows, left key columns are
///   filled from the corresponding right key column instead of nulls.
/// - `right_key_set`: the set of right-side key column indices (excluded from output).
pub fn build_join_output(
    left: &SFrameRows,
    right: &SFrameRows,
    matched_left: &[usize],
    matched_right: &[usize],
    unmatched_left: &[usize],
    unmatched_right: &[usize],
    left_to_right_key: &HashMap<usize, usize>,
    right_key_set: &HashSet<usize>,
) -> SFrameRows {
    let left_dtypes = left.dtypes();
    let right_dtypes = right.dtypes();

    let mut output_columns: Vec<ColumnData> = Vec::with_capacity(
        left.num_columns() + right.num_columns() - right_key_set.len(),
    );

    // Build left-side output columns.
    for left_col_idx in 0..left.num_columns() {
        // Matched rows: gather from left at matched_left indices.
        let mut col = left.column(left_col_idx).gather(matched_left);

        // Unmatched-left rows: gather from left.
        if !unmatched_left.is_empty() {
            let ul = left.column(left_col_idx).gather(unmatched_left);
            col.extend(&ul).expect("type mismatch in unmatched_left extend");
        }

        // Unmatched-right rows: if this is a key column, fill from corresponding
        // right key column; otherwise fill with nulls.
        if !unmatched_right.is_empty() {
            if let Some(&right_col_idx) = left_to_right_key.get(&left_col_idx) {
                let ur = right.column(right_col_idx).gather(unmatched_right);
                col.extend(&ur).expect("type mismatch in unmatched_right key extend");
            } else {
                let nulls = null_column(left_dtypes[left_col_idx], unmatched_right.len());
                col.extend(&nulls).expect("type mismatch in null extend");
            }
        }

        output_columns.push(col);
    }

    // Build right-side output columns (excluding key columns).
    for right_col_idx in 0..right.num_columns() {
        if right_key_set.contains(&right_col_idx) {
            continue;
        }

        // Matched rows: gather from right at matched_right indices.
        let mut col = right.column(right_col_idx).gather(matched_right);

        // Unmatched-left rows: nulls.
        if !unmatched_left.is_empty() {
            let nulls = null_column(right_dtypes[right_col_idx], unmatched_left.len());
            col.extend(&nulls).expect("type mismatch in unmatched_left null extend");
        }

        // Unmatched-right rows: gather from right.
        if !unmatched_right.is_empty() {
            let ur = right.column(right_col_idx).gather(unmatched_right);
            col.extend(&ur).expect("type mismatch in unmatched_right extend");
        }

        output_columns.push(col);
    }

    SFrameRows::new(output_columns).expect("column length mismatch in build_join_output")
}

/// Create an empty output batch with the correct schema for a join.
///
/// Schema: all left dtypes followed by right non-key dtypes.
pub fn make_empty_output(
    left_dtypes: &[FlexTypeEnum],
    right_dtypes: &[FlexTypeEnum],
    on: &JoinOn,
) -> SFrameRows {
    let right_key_set: HashSet<usize> = on.pairs.iter().map(|&(_, r)| r).collect();
    let mut dtypes: Vec<FlexTypeEnum> = left_dtypes.to_vec();
    for (i, &dt) in right_dtypes.iter().enumerate() {
        if !right_key_set.contains(&i) {
            dtypes.push(dt);
        }
    }
    SFrameRows::empty(&dtypes)
}

/// Join two batches using the hash-table approach.
///
/// `build_is_left` controls which side is used to build the hash table.
/// When true, the left side is built and the right side probes; when false,
/// the right side is built and the left probes.
///
/// The output schema is always: all left columns + right non-key columns.
pub fn join_batches(
    left: &SFrameRows,
    right: &SFrameRows,
    on: &JoinOn,
    join_type: JoinType,
    build_is_left: bool,
) -> SFrameRows {
    let left_key_cols: Vec<usize> = on.pairs.iter().map(|&(l, _)| l).collect();
    let right_key_cols: Vec<usize> = on.pairs.iter().map(|&(_, r)| r).collect();

    // Do we need to track unmatched build rows?
    let need_unmatched_build = match (join_type, build_is_left) {
        // If build is left, we need unmatched build for Left/Full joins.
        (JoinType::Left, true) | (JoinType::Full, true) => true,
        // If build is right, we need unmatched build for Right/Full joins.
        (JoinType::Right, false) | (JoinType::Full, false) => true,
        _ => false,
    };

    // Do we need to collect unmatched probe rows?
    let need_unmatched_probe = match (join_type, build_is_left) {
        // If build is left, probe is right. Need unmatched probe for Right/Full.
        (JoinType::Right, true) | (JoinType::Full, true) => true,
        // If build is right, probe is left. Need unmatched probe for Left/Full.
        (JoinType::Left, false) | (JoinType::Full, false) => true,
        _ => false,
    };

    let (build, probe) = if build_is_left {
        (left, right)
    } else {
        (right, left)
    };
    let (build_key_cols, probe_key_cols) = if build_is_left {
        (&left_key_cols, &right_key_cols)
    } else {
        (&right_key_cols, &left_key_cols)
    };

    // Build hash table.
    let ht = JoinHashTable::new(build, build_key_cols, need_unmatched_build);

    // Probe.
    let probe_result = ht.probe_parallel(probe, probe_key_cols, need_unmatched_probe);

    // Map build/probe indices back to left/right.
    let (matched_left, matched_right) = if build_is_left {
        (probe_result.matched_build, probe_result.matched_probe)
    } else {
        (probe_result.matched_probe, probe_result.matched_build)
    };

    // Determine unmatched indices.
    let (unmatched_left, unmatched_right) = if build_is_left {
        let ul = if need_unmatched_build {
            ht.unmatched_build_indices()
        } else {
            Vec::new()
        };
        let ur = if need_unmatched_probe {
            probe_result.unmatched_probe
        } else {
            Vec::new()
        };
        (ul, ur)
    } else {
        let ur = if need_unmatched_build {
            ht.unmatched_build_indices()
        } else {
            Vec::new()
        };
        let ul = if need_unmatched_probe {
            probe_result.unmatched_probe
        } else {
            Vec::new()
        };
        (ul, ur)
    };

    // Build the key mapping and key set for output construction.
    let left_to_right_key: HashMap<usize, usize> = on.pairs.iter().copied().collect();
    let right_key_set: HashSet<usize> = right_key_cols.into_iter().collect();

    build_join_output(
        left,
        right,
        &matched_left,
        &matched_right,
        &unmatched_left,
        &unmatched_right,
        &left_to_right_key,
        &right_key_set,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_null_column_integer() {
        let col = null_column(FlexTypeEnum::Integer, 3);
        assert_eq!(col.len(), 3);
        for i in 0..3 {
            assert_eq!(col.get(i), FlexType::Undefined);
        }
    }

    #[test]
    fn test_null_column_flexible() {
        let col = null_column(FlexTypeEnum::Undefined, 2);
        assert_eq!(col.len(), 2);
        for i in 0..2 {
            assert_eq!(col.get(i), FlexType::Undefined);
        }
    }

    #[test]
    fn test_build_join_output_inner() {
        // Left: id(int), name(str)  |  Right: id(int), score(float)
        // Join on left col 0 = right col 0.
        // 2 matched pairs, no unmatched.
        let left = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(1), FlexType::String("alice".into())],
                vec![FlexType::Integer(2), FlexType::String("bob".into())],
                vec![FlexType::Integer(3), FlexType::String("charlie".into())],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        let right = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(2), FlexType::Float(80.0)],
                vec![FlexType::Integer(3), FlexType::Float(90.0)],
                vec![FlexType::Integer(4), FlexType::Float(70.0)],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let left_to_right_key: HashMap<usize, usize> = [(0, 0)].into_iter().collect();
        let right_key_set: HashSet<usize> = [0].into_iter().collect();

        // matched_left=[1,2], matched_right=[0,1] means left row 1 <-> right row 0, etc.
        let result = build_join_output(
            &left,
            &right,
            &[1, 2],    // matched_left
            &[0, 1],    // matched_right
            &[],         // unmatched_left
            &[],         // unmatched_right
            &left_to_right_key,
            &right_key_set,
        );

        // Output: left.id, left.name, right.score  (right.id excluded)
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 3);

        // Row 0: left row 1 (id=2, "bob") + right row 0 (score=80.0)
        assert_eq!(result.row(0), vec![
            FlexType::Integer(2),
            FlexType::String("bob".into()),
            FlexType::Float(80.0),
        ]);
        // Row 1: left row 2 (id=3, "charlie") + right row 1 (score=90.0)
        assert_eq!(result.row(1), vec![
            FlexType::Integer(3),
            FlexType::String("charlie".into()),
            FlexType::Float(90.0),
        ]);
    }

    #[test]
    fn test_build_join_output_full() {
        // Left: id(int), name(str)  |  Right: id(int), score(float)
        // 2 matched + 1 unmatched left + 1 unmatched right => 4 rows.
        let left = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(1), FlexType::String("alice".into())],
                vec![FlexType::Integer(2), FlexType::String("bob".into())],
                vec![FlexType::Integer(3), FlexType::String("charlie".into())],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        let right = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(2), FlexType::Float(80.0)],
                vec![FlexType::Integer(3), FlexType::Float(90.0)],
                vec![FlexType::Integer(4), FlexType::Float(70.0)],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let left_to_right_key: HashMap<usize, usize> = [(0, 0)].into_iter().collect();
        let right_key_set: HashSet<usize> = [0].into_iter().collect();

        let result = build_join_output(
            &left,
            &right,
            &[1, 2],    // matched_left: left rows 1,2
            &[0, 1],    // matched_right: right rows 0,1
            &[0],        // unmatched_left: left row 0 (id=1)
            &[2],        // unmatched_right: right row 2 (id=4)
            &left_to_right_key,
            &right_key_set,
        );

        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 3);

        // Row 0: matched left=1 right=0 => (2, "bob", 80.0)
        assert_eq!(result.row(0), vec![
            FlexType::Integer(2),
            FlexType::String("bob".into()),
            FlexType::Float(80.0),
        ]);
        // Row 1: matched left=2 right=1 => (3, "charlie", 90.0)
        assert_eq!(result.row(1), vec![
            FlexType::Integer(3),
            FlexType::String("charlie".into()),
            FlexType::Float(90.0),
        ]);
        // Row 2: unmatched left=0 => (1, "alice", NULL)
        assert_eq!(result.row(2), vec![
            FlexType::Integer(1),
            FlexType::String("alice".into()),
            FlexType::Undefined,
        ]);
        // Row 3: unmatched right=2 => (4, NULL, 70.0)
        // Key column (left col 0) is filled from right col 0 = 4.
        // Non-key left column (name) is null.
        assert_eq!(result.row(3), vec![
            FlexType::Integer(4),
            FlexType::Undefined,
            FlexType::Float(70.0),
        ]);
    }

    #[test]
    fn test_join_batches_inner() {
        // Left: id(int), name(str)
        let left = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(1), FlexType::String("alice".into())],
                vec![FlexType::Integer(2), FlexType::String("bob".into())],
                vec![FlexType::Integer(3), FlexType::String("charlie".into())],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        // Right: id(int), score(float)
        let right = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(2), FlexType::Float(80.0)],
                vec![FlexType::Integer(3), FlexType::Float(90.0)],
                vec![FlexType::Integer(4), FlexType::Float(70.0)],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let on = JoinOn::new(0, 0);
        let result = join_batches(&left, &right, &on, JoinType::Inner, false);

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 3); // left.id, left.name, right.score

        // Collect rows and sort by id for deterministic comparison.
        let mut rows: Vec<Vec<FlexType>> = (0..result.num_rows())
            .map(|i| result.row(i))
            .collect();
        rows.sort_by_key(|r| match &r[0] {
            FlexType::Integer(i) => *i,
            _ => panic!("expected integer"),
        });

        assert_eq!(rows[0], vec![
            FlexType::Integer(2),
            FlexType::String("bob".into()),
            FlexType::Float(80.0),
        ]);
        assert_eq!(rows[1], vec![
            FlexType::Integer(3),
            FlexType::String("charlie".into()),
            FlexType::Float(90.0),
        ]);
    }

    #[test]
    fn test_join_batches_left() {
        let left = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(1), FlexType::String("alice".into())],
                vec![FlexType::Integer(2), FlexType::String("bob".into())],
                vec![FlexType::Integer(3), FlexType::String("charlie".into())],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        let right = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(2), FlexType::Float(80.0)],
                vec![FlexType::Integer(3), FlexType::Float(90.0)],
                vec![FlexType::Integer(4), FlexType::Float(70.0)],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let on = JoinOn::new(0, 0);
        let result = join_batches(&left, &right, &on, JoinType::Left, false);

        // All 3 left rows should be present.
        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.num_columns(), 3);

        let mut rows: Vec<Vec<FlexType>> = (0..result.num_rows())
            .map(|i| result.row(i))
            .collect();
        rows.sort_by_key(|r| match &r[0] {
            FlexType::Integer(i) => *i,
            _ => panic!("expected integer"),
        });

        // id=1 has no match, score should be NULL.
        assert_eq!(rows[0], vec![
            FlexType::Integer(1),
            FlexType::String("alice".into()),
            FlexType::Undefined,
        ]);
        assert_eq!(rows[1], vec![
            FlexType::Integer(2),
            FlexType::String("bob".into()),
            FlexType::Float(80.0),
        ]);
        assert_eq!(rows[2], vec![
            FlexType::Integer(3),
            FlexType::String("charlie".into()),
            FlexType::Float(90.0),
        ]);
    }

    #[test]
    fn test_join_batches_right() {
        let left = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(1), FlexType::String("alice".into())],
                vec![FlexType::Integer(2), FlexType::String("bob".into())],
                vec![FlexType::Integer(3), FlexType::String("charlie".into())],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        let right = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(2), FlexType::Float(80.0)],
                vec![FlexType::Integer(3), FlexType::Float(90.0)],
                vec![FlexType::Integer(4), FlexType::Float(70.0)],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let on = JoinOn::new(0, 0);
        let result = join_batches(&left, &right, &on, JoinType::Right, false);

        // All 3 right rows should be present.
        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.num_columns(), 3);

        let mut rows: Vec<Vec<FlexType>> = (0..result.num_rows())
            .map(|i| result.row(i))
            .collect();
        rows.sort_by_key(|r| match &r[0] {
            FlexType::Integer(i) => *i,
            _ => panic!("expected integer"),
        });

        assert_eq!(rows[0], vec![
            FlexType::Integer(2),
            FlexType::String("bob".into()),
            FlexType::Float(80.0),
        ]);
        assert_eq!(rows[1], vec![
            FlexType::Integer(3),
            FlexType::String("charlie".into()),
            FlexType::Float(90.0),
        ]);
        // id=4 is right-only: key column filled from right, name is NULL.
        assert_eq!(rows[2], vec![
            FlexType::Integer(4),
            FlexType::Undefined,
            FlexType::Float(70.0),
        ]);
    }

    #[test]
    fn test_join_batches_full() {
        let left = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(1), FlexType::String("alice".into())],
                vec![FlexType::Integer(2), FlexType::String("bob".into())],
                vec![FlexType::Integer(3), FlexType::String("charlie".into())],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        let right = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(2), FlexType::Float(80.0)],
                vec![FlexType::Integer(3), FlexType::Float(90.0)],
                vec![FlexType::Integer(4), FlexType::Float(70.0)],
            ],
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let on = JoinOn::new(0, 0);
        let result = join_batches(&left, &right, &on, JoinType::Full, false);

        // 2 matched + 1 left-only (id=1) + 1 right-only (id=4) = 4 rows.
        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 3);

        let mut rows: Vec<Vec<FlexType>> = (0..result.num_rows())
            .map(|i| result.row(i))
            .collect();
        rows.sort_by_key(|r| match &r[0] {
            FlexType::Integer(i) => *i,
            _ => panic!("expected integer"),
        });

        // id=1: left-only, score is NULL
        assert_eq!(rows[0], vec![
            FlexType::Integer(1),
            FlexType::String("alice".into()),
            FlexType::Undefined,
        ]);
        // id=2: matched
        assert_eq!(rows[1], vec![
            FlexType::Integer(2),
            FlexType::String("bob".into()),
            FlexType::Float(80.0),
        ]);
        // id=3: matched
        assert_eq!(rows[2], vec![
            FlexType::Integer(3),
            FlexType::String("charlie".into()),
            FlexType::Float(90.0),
        ]);
        // id=4: right-only, key filled from right, name is NULL
        assert_eq!(rows[3], vec![
            FlexType::Integer(4),
            FlexType::Undefined,
            FlexType::Float(70.0),
        ]);
    }

    #[test]
    fn test_make_empty_output() {
        let left_dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let right_dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::Float];
        let on = JoinOn::new(0, 0);

        let result = make_empty_output(&left_dtypes, &right_dtypes, &on);
        assert_eq!(result.num_rows(), 0);
        assert_eq!(result.num_columns(), 3);
        assert_eq!(
            result.dtypes(),
            vec![FlexTypeEnum::Integer, FlexTypeEnum::String, FlexTypeEnum::Float]
        );
    }
}
