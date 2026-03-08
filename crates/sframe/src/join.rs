//! Hash join building blocks.
//!
//! `JoinHashTable` maps composite keys (one or more columns) from a "build"
//! batch to row indices. The probe side can then look up matching rows in
//! parallel via rayon, collecting matched pairs and optionally tracking
//! unmatched rows on both sides.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rayon::prelude::*;
use sframe_io::cache_fs::{global_cache_fs, CacheFs};
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_query::algorithms::join::{JoinOn, JoinType};
use sframe_query::batch::{ColumnData, SFrameRows};
use sframe_query::execute::materialize_sync;
use sframe_storage::scatter_writer::ScatterWriter;
use sframe_storage::segment_reader::SegmentReader;
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::sframe::{SFrame, SFrameBuilder};

/// Result of probing the hash table with a batch of probe rows.
struct ProbeResult {
    /// Build-side row indices of matched pairs (parallel with `matched_probe`).
    matched_build: Vec<usize>,
    /// Probe-side row indices of matched pairs (parallel with `matched_build`).
    matched_probe: Vec<usize>,
    /// Probe-side row indices that found no match in the build table.
    unmatched_probe: Vec<usize>,
}

/// A hash table built from the key columns of a build-side `SFrameRows` batch.
///
/// Maps composite key values to the list of build-row indices sharing that key.
/// Supports parallel probing and optional tracking of which build rows were
/// ever matched (for full/right outer joins).
struct JoinHashTable {
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
    fn new(build: &SFrameRows, key_cols: &[usize], track_matched: bool) -> Self {
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
    fn probe_parallel(
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
    fn unmatched_build_indices(&self) -> Vec<usize> {
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
fn null_column(dtype: FlexTypeEnum, n: usize) -> ColumnData {
    match dtype {
        FlexTypeEnum::Integer => ColumnData::Integer(std::iter::repeat_n(None, n).collect()),
        FlexTypeEnum::Float => ColumnData::Float(std::iter::repeat_n(None, n).collect()),
        FlexTypeEnum::String => ColumnData::String(std::iter::repeat_n(None, n).collect()),
        FlexTypeEnum::Vector => ColumnData::Vector(std::iter::repeat_n(None, n).collect()),
        FlexTypeEnum::List => ColumnData::List(std::iter::repeat_n(None, n).collect()),
        FlexTypeEnum::Dict => ColumnData::Dict(std::iter::repeat_n(None, n).collect()),
        FlexTypeEnum::DateTime => ColumnData::DateTime(std::iter::repeat_n(None, n).collect()),
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
fn build_join_output(
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
fn make_empty_output(
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
fn join_batches(
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

// ---------------------------------------------------------------------------
// Streaming & GRACE hash join pipeline
// ---------------------------------------------------------------------------

const DEFAULT_CHUNK_SIZE: usize = 8192;


/// Consume a `BatchStream` fully, merging all batches into a single `SFrameRows`.
fn materialize_batch_stream(sf: &SFrame) -> Result<SFrameRows> {
    let stream = sf.compile_stream()?;
    materialize_sync(stream)
}

/// Estimate the number of cells (rows * cols) in an SFrame.
///
/// First tries `known_len()` (free). If unavailable, falls back to
/// `num_rows()` which may materialize the plan.
fn estimate_cells(sf: &SFrame) -> Result<u64> {
    let ncols = sf.num_columns() as u64;
    if ncols == 0 {
        return Ok(0);
    }
    let nrows = match sf.known_len() {
        Some(n) => n,
        None => sf.num_rows()?,
    };
    Ok(nrows * ncols)
}

/// Hash key columns of a row to compute a partition index.
fn compute_partition(
    batch: &SFrameRows,
    row_idx: usize,
    key_cols: &[usize],
    num_partitions: usize,
) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for &c in key_cols {
        batch.column(c).get(row_idx).hash(&mut hasher);
    }
    let h = hasher.finish();
    (h % num_partitions as u64) as usize
}

/// On-disk partition set written by `partition_to_disk`.
struct PartitionSet {
    /// Cache directory holding all segment files.
    dir: String,
    /// Segment file paths (one per partition).
    segment_files: Vec<String>,
    /// `sizes[partition][column]` — row count per column per partition.
    sizes: Vec<Vec<u64>>,
    /// Column types matching the source SFrame.
    dtypes: Vec<FlexTypeEnum>,
}

/// Stream an SFrame batch-by-batch, hash-partition each row by `key_cols`,
/// and write to a `ScatterWriter` on CacheFs.
fn partition_to_disk(
    sf: &SFrame,
    key_cols: &[usize],
    num_partitions: usize,
    cache_fs: &Arc<CacheFs>,
) -> Result<PartitionSet> {
    let dtypes = sf.column_types();
    let dir = cache_fs.alloc_dir();
    let vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
    vfs.mkdir_p(&dir)?;

    let mut scatter = ScatterWriter::new(&*vfs, &dir, "part", &dtypes, num_partitions)?;

    let mut stream = sf.compile_stream()?;
    while let Some(batch_result) = stream.next_batch() {
        let batch = batch_result?;
        let nrows = batch.num_rows();
        let ncols = batch.num_columns();
        for row_idx in 0..nrows {
            let part = compute_partition(&batch, row_idx, key_cols, num_partitions);
            for col in 0..ncols {
                let val = batch.column(col).get(row_idx);
                scatter.write_to_segment(col, part, val)?;
            }
        }
    }

    let result = scatter.finish()?;
    Ok(PartitionSet {
        dir,
        segment_files: result.segment_files,
        sizes: result.all_segment_sizes,
        dtypes,
    })
}

/// Read one partition from a `PartitionSet` into `SFrameRows`.
fn read_partition(parts: &PartitionSet, partition_idx: usize, cache_fs: &Arc<CacheFs>) -> Result<SFrameRows> {
    // Check if partition is empty (all column sizes are 0).
    let col_sizes = &parts.sizes[partition_idx];
    if col_sizes.iter().all(|&s| s == 0) {
        return Ok(SFrameRows::empty(&parts.dtypes));
    }

    let seg_file = &parts.segment_files[partition_idx];
    let path = format!("{}/{}", parts.dir, seg_file);
    let vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
    let file = vfs.open_read(&path)?;
    let file_size = file.size()?;
    let mut reader = SegmentReader::open(Box::new(file), file_size, parts.dtypes.clone())?;

    let mut columns = Vec::with_capacity(parts.dtypes.len());
    for col_idx in 0..parts.dtypes.len() {
        let values = reader.read_column(col_idx)?;
        let col = ColumnData::from_flex_vec(values, parts.dtypes[col_idx]);
        columns.push(col);
    }

    SFrameRows::new(columns)
}

/// Streaming hash join: materialize one side into a hash table,
/// stream the other side batch-by-batch, probing and emitting output.
fn streaming_hash_join(
    left: &SFrame,
    right: &SFrame,
    on: &JoinOn,
    join_type: JoinType,
    build_is_left: bool,
    output_names: &[String],
    output_dtypes: &[FlexTypeEnum],
) -> Result<SFrame> {
    let left_key_cols: Vec<usize> = on.pairs.iter().map(|&(l, _)| l).collect();
    let right_key_cols: Vec<usize> = on.pairs.iter().map(|&(_, r)| r).collect();
    let left_to_right_key: HashMap<usize, usize> = on.pairs.iter().copied().collect();
    let right_key_set: HashSet<usize> = right_key_cols.iter().copied().collect();

    // Determine what we need to track.
    let need_unmatched_build = match (join_type, build_is_left) {
        (JoinType::Left, true) | (JoinType::Full, true) => true,
        (JoinType::Right, false) | (JoinType::Full, false) => true,
        _ => false,
    };
    let need_unmatched_probe = match (join_type, build_is_left) {
        (JoinType::Right, true) | (JoinType::Full, true) => true,
        (JoinType::Left, false) | (JoinType::Full, false) => true,
        _ => false,
    };

    // Step 1: Materialize the build side.
    let (build_sf, probe_sf) = if build_is_left {
        (left, right)
    } else {
        (right, left)
    };
    let build_data = materialize_batch_stream(build_sf)?;

    // Step 2: Build hash table.
    let (build_key_cols, probe_key_cols) = if build_is_left {
        (&left_key_cols, &right_key_cols)
    } else {
        (&right_key_cols, &left_key_cols)
    };
    let ht = JoinHashTable::new(&build_data, build_key_cols, need_unmatched_build);

    // Step 3: Stream probe side, probe hash table, emit output.
    let mut builder = SFrameBuilder::anonymous(output_names.to_vec(), output_dtypes.to_vec())?;
    let mut probe_stream = probe_sf.compile_stream()?;

    while let Some(batch_result) = probe_stream.next_batch() {
        let probe_batch = batch_result?;
        let probe_result = ht.probe_parallel(&probe_batch, probe_key_cols, need_unmatched_probe);

        // Map build/probe indices to left/right.
        let (matched_left, matched_right) = if build_is_left {
            (probe_result.matched_build, probe_result.matched_probe)
        } else {
            (probe_result.matched_probe, probe_result.matched_build)
        };

        // Per-batch unmatched probe rows (emit immediately for RIGHT/FULL when
        // build is left, or LEFT/FULL when build is right).
        let unmatched_probe_rows = if need_unmatched_probe {
            probe_result.unmatched_probe
        } else {
            Vec::new()
        };

        let (unmatched_left, unmatched_right) = if build_is_left {
            // Build is left; unmatched build handled after all batches.
            // Unmatched probe rows are right-side unmatched.
            (Vec::new(), unmatched_probe_rows)
        } else {
            // Build is right; unmatched build handled after all batches.
            // Unmatched probe rows are left-side unmatched.
            (unmatched_probe_rows, Vec::new())
        };

        // Determine left/right references for this batch.
        let (left_ref, right_ref) = if build_is_left {
            (&build_data, &probe_batch)
        } else {
            (&probe_batch, &build_data)
        };

        let output_batch = build_join_output(
            left_ref,
            right_ref,
            &matched_left,
            &matched_right,
            &unmatched_left,
            &unmatched_right,
            &left_to_right_key,
            &right_key_set,
        );

        if output_batch.num_rows() > 0 {
            builder.write_batch_chunked(&output_batch, DEFAULT_CHUNK_SIZE)?;
        }
    }

    // Step 4: Emit unmatched build rows (LEFT/FULL when build is left, etc.)
    if need_unmatched_build {
        let unmatched_build_indices = ht.unmatched_build_indices();
        if !unmatched_build_indices.is_empty() {
            let empty_probe = SFrameRows::empty(
                &if build_is_left {
                    right.column_types()
                } else {
                    left.column_types()
                },
            );

            let (unmatched_left, unmatched_right) = if build_is_left {
                (unmatched_build_indices, Vec::new())
            } else {
                (Vec::new(), unmatched_build_indices)
            };

            let (left_ref, right_ref) = if build_is_left {
                (&build_data, &empty_probe)
            } else {
                (&empty_probe, &build_data)
            };

            let tail_batch = build_join_output(
                left_ref,
                right_ref,
                &[],  // no matched rows
                &[],
                &unmatched_left,
                &unmatched_right,
                &left_to_right_key,
                &right_key_set,
            );

            if tail_batch.num_rows() > 0 {
                builder.write_batch_chunked(&tail_batch, DEFAULT_CHUNK_SIZE)?;
            }
        }
    }

    builder.finish()
}

/// GRACE hash join: partition both sides to disk, then join each
/// partition pair in-memory.
fn grace_hash_join(
    left: &SFrame,
    right: &SFrame,
    on: &JoinOn,
    join_type: JoinType,
    output_names: &[String],
    output_dtypes: &[FlexTypeEnum],
    budget: usize,
) -> Result<SFrame> {
    let left_key_cols: Vec<usize> = on.pairs.iter().map(|&(l, _)| l).collect();
    let right_key_cols: Vec<usize> = on.pairs.iter().map(|&(_, r)| r).collect();

    // Estimate partition count.
    let left_cells = estimate_cells(left)?;
    let right_cells = estimate_cells(right)?;
    let smaller_cells = left_cells.min(right_cells).max(1);
    let num_partitions = ((smaller_cells as usize / budget.max(1)) + 1).max(2);

    let cache_fs = global_cache_fs();

    // Partition both sides to disk.
    let left_parts = partition_to_disk(left, &left_key_cols, num_partitions, cache_fs)?;
    let right_parts = partition_to_disk(right, &right_key_cols, num_partitions, cache_fs)?;

    let mut builder = SFrameBuilder::anonymous(output_names.to_vec(), output_dtypes.to_vec())?;

    // Join each partition pair.
    for p in 0..num_partitions {
        let left_batch = read_partition(&left_parts, p, cache_fs)?;
        let right_batch = read_partition(&right_parts, p, cache_fs)?;

        if left_batch.num_rows() == 0 && right_batch.num_rows() == 0 {
            continue;
        }

        // Pick smaller side as build.
        let build_is_left = left_batch.num_rows() <= right_batch.num_rows();
        let output = join_batches(&left_batch, &right_batch, on, join_type, build_is_left);

        if output.num_rows() > 0 {
            builder.write_batch_chunked(&output, DEFAULT_CHUNK_SIZE)?;
        }
    }

    // Cleanup temp dirs.
    cache_fs.remove_dir(&left_parts.dir).ok();
    cache_fs.remove_dir(&right_parts.dir).ok();

    builder.finish()
}

/// Top-level hash join entry point.
///
/// Routes to `streaming_hash_join` when one side fits in memory,
/// or `grace_hash_join` for large-by-large joins.
pub(crate) fn hash_join(
    left: &SFrame,
    right: &SFrame,
    on: &JoinOn,
    join_type: JoinType,
    output_names: &[String],
    output_dtypes: &[FlexTypeEnum],
) -> Result<SFrame> {
    let budget = sframe_config::global().join_buffer_num_cells();

    // Try known_len first (free, no materialization).
    let left_known = left.known_len().map(|n| n * left.num_columns() as u64);
    let right_known = right.known_len().map(|n| n * right.num_columns() as u64);

    match (left_known, right_known) {
        (Some(lc), Some(rc)) => {
            if lc <= budget as u64 && (rc > budget as u64 || lc <= rc) {
                // Left fits and is smaller (or right doesn't fit) — build on left.
                streaming_hash_join(left, right, on, join_type, true, output_names, output_dtypes)
            } else if rc <= budget as u64 {
                // Right fits in memory — use it as build side.
                streaming_hash_join(left, right, on, join_type, false, output_names, output_dtypes)
            } else {
                // Neither fits — GRACE.
                grace_hash_join(left, right, on, join_type, output_names, output_dtypes, budget)
            }
        }
        (Some(lc), None) if lc <= budget as u64 => {
            streaming_hash_join(left, right, on, join_type, true, output_names, output_dtypes)
        }
        (None, Some(rc)) if rc <= budget as u64 => {
            streaming_hash_join(left, right, on, join_type, false, output_names, output_dtypes)
        }
        _ => {
            // Sizes unknown or known but neither fits — estimate to decide.
            let lc = estimate_cells(left)?;
            let rc = estimate_cells(right)?;
            if lc <= budget as u64 && (rc > budget as u64 || lc <= rc) {
                streaming_hash_join(left, right, on, join_type, true, output_names, output_dtypes)
            } else if rc <= budget as u64 {
                streaming_hash_join(left, right, on, join_type, false, output_names, output_dtypes)
            } else {
                grace_hash_join(left, right, on, join_type, output_names, output_dtypes, budget)
            }
        }
    }
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

    #[test]
    fn test_streaming_hash_join() {
        let left = SFrame::from_columns(vec![
            (
                "id",
                crate::sarray::SArray::from_vec(
                    (1..=3).map(FlexType::Integer).collect(),
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "name",
                crate::sarray::SArray::from_vec(
                    vec![
                        FlexType::String("a".into()),
                        FlexType::String("b".into()),
                        FlexType::String("c".into()),
                    ],
                    FlexTypeEnum::String,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let right = SFrame::from_columns(vec![
            (
                "id",
                crate::sarray::SArray::from_vec(
                    vec![
                        FlexType::Integer(2),
                        FlexType::Integer(3),
                        FlexType::Integer(4),
                    ],
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "score",
                crate::sarray::SArray::from_vec(
                    vec![
                        FlexType::Float(20.0),
                        FlexType::Float(30.0),
                        FlexType::Float(40.0),
                    ],
                    FlexTypeEnum::Float,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let on = JoinOn::new(0, 0);
        let names = vec!["id".into(), "name".into(), "score".into()];
        let dtypes = vec![
            FlexTypeEnum::Integer,
            FlexTypeEnum::String,
            FlexTypeEnum::Float,
        ];

        let result =
            streaming_hash_join(&left, &right, &on, JoinType::Inner, true, &names, &dtypes)
                .unwrap();
        assert_eq!(result.num_rows().unwrap(), 2);
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_streaming_hash_join_left() {
        let left = SFrame::from_columns(vec![
            (
                "id",
                crate::sarray::SArray::from_vec(
                    (1..=3).map(FlexType::Integer).collect(),
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "name",
                crate::sarray::SArray::from_vec(
                    vec![
                        FlexType::String("a".into()),
                        FlexType::String("b".into()),
                        FlexType::String("c".into()),
                    ],
                    FlexTypeEnum::String,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let right = SFrame::from_columns(vec![
            (
                "id",
                crate::sarray::SArray::from_vec(
                    vec![FlexType::Integer(2), FlexType::Integer(3)],
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "score",
                crate::sarray::SArray::from_vec(
                    vec![FlexType::Float(20.0), FlexType::Float(30.0)],
                    FlexTypeEnum::Float,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let on = JoinOn::new(0, 0);
        let names = vec!["id".into(), "name".into(), "score".into()];
        let dtypes = vec![
            FlexTypeEnum::Integer,
            FlexTypeEnum::String,
            FlexTypeEnum::Float,
        ];

        // Left join with build=left: all 3 left rows should appear.
        let result =
            streaming_hash_join(&left, &right, &on, JoinType::Left, true, &names, &dtypes)
                .unwrap();
        assert_eq!(result.num_rows().unwrap(), 3);
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_grace_hash_join() {
        let n = 200i64;
        let left = SFrame::from_columns(vec![
            (
                "id",
                crate::sarray::SArray::from_vec(
                    (0..n).map(FlexType::Integer).collect(),
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "val",
                crate::sarray::SArray::from_vec(
                    (0..n)
                        .map(|i| FlexType::String(format!("L{i}").into()))
                        .collect(),
                    FlexTypeEnum::String,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let right = SFrame::from_columns(vec![
            (
                "id",
                crate::sarray::SArray::from_vec(
                    (0..n).map(FlexType::Integer).collect(),
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "score",
                crate::sarray::SArray::from_vec(
                    (0..n).map(|i| FlexType::Float(i as f64)).collect(),
                    FlexTypeEnum::Float,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
        let on = JoinOn::new(0, 0);
        let names = vec!["id".into(), "val".into(), "score".into()];
        let dtypes = vec![
            FlexTypeEnum::Integer,
            FlexTypeEnum::String,
            FlexTypeEnum::Float,
        ];

        // Force GRACE with tiny budget
        let result =
            grace_hash_join(&left, &right, &on, JoinType::Inner, &names, &dtypes, 100).unwrap();
        assert_eq!(result.num_rows().unwrap(), n as u64);
        assert_eq!(result.num_columns(), 3);
    }
}
