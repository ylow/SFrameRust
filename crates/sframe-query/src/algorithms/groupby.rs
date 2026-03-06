//! Hash-partitioned external memory groupby algorithm.
//!
//! Two-phase algorithm with bounded memory:
//! - Phase 1: Ingest rows in parallel, hash-partition into segments, spill to disk when full
//! - Phase 2: Per-segment k-way merge of spilled chunks, write output directly to CacheFS
//!
//! Input is read in parallel from a PlannerNode (when the plan is parallel-sliceable).
//! Output is materialized as an SFrame on CacheFS, returned as a `cache://` path.
//!
//! Spill files are stored in `cache://` via the CacheFs VFS backend, which
//! provides automatic cleanup when the SpillState is dropped.

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use sframe_io::cache_fs::{global_cache_fs, CacheFs};
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_storage::segment_writer::SegmentWriter;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};
use sframe_types::serialization::{read_flex_type, read_u64, write_flex_type, write_u64};

use crate::algorithms::aggregators::AggSpec;
use crate::execute::{compile, for_each_batch_sync, parallel_slice_row_count};
use crate::planner::{clone_plan_with_row_range, Aggregator, PlannerNode};

/// (hash, keys, aggregators) triple used during flush and merge.
type GroupEntry = (u64, Vec<FlexType>, Vec<Box<dyn Aggregator>>);

/// Default number of hash-partition segments.
const DEFAULT_NUM_SEGMENTS: usize = 16;

/// Default total buffer capacity in distinct keys across all segments.
const DEFAULT_BUFFER_NUM_ROWS: usize = 1_048_576; // ~1M keys

/// Number of rows to buffer before writing a column block to the output segment.
const OUTPUT_BATCH_SIZE: usize = 65_536;

/// Perform a groupby operation on a plan.
///
/// Groups by `key_columns` and applies `agg_specs` to produce the aggregated result.
/// Reads input in parallel when the plan is parallel-sliceable.
/// Writes output directly to CacheFS.
///
/// Returns a `cache://` SFrame path. The caller is responsible for cleanup
/// via `global_cache_fs().remove_dir()` when the result is no longer needed.
pub fn groupby(
    plan: &Arc<PlannerNode>,
    key_columns: &[usize],
    agg_specs: &[AggSpec],
    column_names: &[String],
    column_types: &[FlexTypeEnum],
) -> Result<String> {
    groupby_with_config(
        plan,
        key_columns,
        agg_specs,
        column_names,
        column_types,
        DEFAULT_NUM_SEGMENTS,
        DEFAULT_BUFFER_NUM_ROWS / DEFAULT_NUM_SEGMENTS,
    )
}

/// Groupby with explicit configuration for testing.
pub(crate) fn groupby_with_config(
    plan: &Arc<PlannerNode>,
    key_columns: &[usize],
    agg_specs: &[AggSpec],
    column_names: &[String],
    column_types: &[FlexTypeEnum],
    num_segments: usize,
    per_segment_buffer: usize,
) -> Result<String> {
    let num_key_cols = key_columns.len();

    // Initialize shared segments with per-segment locks
    let segments: Arc<Vec<Mutex<GroupBySegment>>> = Arc::new(
        (0..num_segments)
            .map(|_| {
                Mutex::new(GroupBySegment {
                    groups: HashMap::new(),
                })
            })
            .collect(),
    );
    let spill_state: Arc<Mutex<Option<SpillState>>> = Arc::new(Mutex::new(None));

    // Phase 1: Ingest rows (parallel when possible)
    ingest_parallel(
        plan,
        key_columns,
        agg_specs,
        num_segments,
        per_segment_buffer,
        &segments,
        &spill_state,
    )?;

    // Phase 2: Produce output directly to CacheFS
    let mut spill_opt = spill_state.lock().unwrap().take();
    let mut owned_segments: Vec<GroupBySegment> = Arc::try_unwrap(segments)
        .map_err(|_| SFrameError::Format("Segments still referenced".into()))?
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect();

    write_output_to_cache(
        &mut owned_segments,
        &mut spill_opt,
        num_key_cols,
        agg_specs,
        column_names,
        column_types,
    )
}

// ---------------------------------------------------------------------------
// Phase 1: Parallel ingestion
// ---------------------------------------------------------------------------

fn ingest_parallel(
    plan: &Arc<PlannerNode>,
    key_columns: &[usize],
    agg_specs: &[AggSpec],
    num_segments: usize,
    per_segment_buffer: usize,
    segments: &Arc<Vec<Mutex<GroupBySegment>>>,
    spill_state: &Arc<Mutex<Option<SpillState>>>,
) -> Result<()> {
    let total_rows = parallel_slice_row_count(plan);

    if let Some(total_rows) = total_rows {
        // Parallel path: split into worker plans by row range
        let n_workers = rayon::current_num_threads().max(1);
        let worker_plans: Vec<Arc<PlannerNode>> = (0..n_workers)
            .filter_map(|i| {
                let begin = (i as u64 * total_rows) / n_workers as u64;
                let end = ((i as u64 + 1) * total_rows) / n_workers as u64;
                if begin >= end {
                    return None;
                }
                Some(clone_plan_with_row_range(plan, begin, end))
            })
            .collect();

        let results: Vec<Result<()>> = worker_plans
            .into_par_iter()
            .map(|worker_plan| {
                let stream = compile(&worker_plan)?;
                ingest_stream(
                    stream,
                    key_columns,
                    agg_specs,
                    num_segments,
                    per_segment_buffer,
                    segments,
                    spill_state,
                )
            })
            .collect();

        for r in results {
            r?;
        }
    } else {
        // Single-threaded fallback
        let stream = compile(plan)?;
        ingest_stream(
            stream,
            key_columns,
            agg_specs,
            num_segments,
            per_segment_buffer,
            segments,
            spill_state,
        )?;
    }

    Ok(())
}

fn ingest_stream(
    stream: crate::execute::BatchIterator,
    key_columns: &[usize],
    agg_specs: &[AggSpec],
    num_segments: usize,
    per_segment_buffer: usize,
    segments: &Arc<Vec<Mutex<GroupBySegment>>>,
    spill_state: &Arc<Mutex<Option<SpillState>>>,
) -> Result<()> {
    for_each_batch_sync(stream, |batch| {
        for row_idx in 0..batch.num_rows() {
            // Extract group key
            let key: Vec<FlexType> = key_columns
                .iter()
                .map(|&col| batch.column(col).get(row_idx))
                .collect();

            // Hash the key to determine segment
            let key_hash = compute_key_hash(&key);
            let seg_id = (key_hash as usize) % num_segments;

            let mut segment = segments[seg_id].lock().unwrap();

            // Get or create aggregators for this group
            let aggs = segment.groups.entry(key).or_insert_with(|| {
                agg_specs
                    .iter()
                    .map(|spec| spec.aggregator.box_clone())
                    .collect()
            });

            // Feed each aggregator its column value
            for (i, spec) in agg_specs.iter().enumerate() {
                let val = batch.column(spec.column).get(row_idx);
                aggs[i].add(&[val]);
            }

            // Check if this segment needs to spill
            if segment.groups.len() >= per_segment_buffer {
                let mut spill_guard = spill_state.lock().unwrap();
                let state = spill_guard.get_or_insert_with(|| {
                    SpillState::new(num_segments).expect("Failed to create CacheFs for spill")
                });
                flush_segment(&mut segment.groups, seg_id, state)?;
            }
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Phase 2: Output to CacheFS
// ---------------------------------------------------------------------------

fn write_output_to_cache(
    segments: &mut [GroupBySegment],
    spill_state: &mut Option<SpillState>,
    num_key_cols: usize,
    agg_specs: &[AggSpec],
    column_names: &[String],
    column_types: &[FlexTypeEnum],
) -> Result<String> {
    let num_cols = column_types.len();
    let cache_fs = global_cache_fs();
    let base_path = cache_fs.alloc_dir();
    let vfs = Arc::new(ArcCacheFsVfs(cache_fs.clone()));

    VirtualFileSystem::mkdir_p(&*vfs, &base_path)?;

    let seg_name = "seg.0000".to_string();
    let seg_path = format!("{base_path}/{seg_name}");
    let file = VirtualFileSystem::open_write(&*vfs, &seg_path)?;
    let mut seg_writer = SegmentWriter::new(file, num_cols);

    let mut row_buffer: Vec<Vec<FlexType>> = (0..num_cols).map(|_| Vec::new()).collect();
    let mut total_rows: u64 = 0;

    // Helper closure: flush buffer to segment writer
    let flush_buffer =
        |buf: &mut Vec<Vec<FlexType>>,
         writer: &mut SegmentWriter<Box<dyn sframe_io::vfs::WritableFile>>,
         dtypes: &[FlexTypeEnum],
         total: &mut u64| -> Result<()> {
            if buf[0].is_empty() {
                return Ok(());
            }
            let n = buf[0].len() as u64;
            for (col_idx, col_data) in buf.iter_mut().enumerate() {
                writer.write_column_block(col_idx, col_data, dtypes[col_idx])?;
                col_data.clear();
            }
            *total += n;
            Ok(())
        };

    let num_segments = segments.len();

    for (seg_id, segment) in segments.iter_mut().enumerate().take(num_segments) {

        if let Some(state) = spill_state.as_mut() {
            if !state.chunks[seg_id].is_empty() {
                // This segment had spills — flush remaining in-memory entries as final chunk
                if !segment.groups.is_empty() {
                    flush_segment(&mut segment.groups, seg_id, state)?;
                }

                // K-way merge of all chunks, writing directly to buffer
                merge_segment_chunks_buffered(
                    state,
                    seg_id,
                    num_key_cols,
                    agg_specs,
                    &mut row_buffer,
                    &mut seg_writer,
                    column_types,
                    &mut total_rows,
                )?;
                continue;
            }
        }

        // Fast path: no spills for this segment, finalize in-memory
        for (keys, mut aggs) in segment.groups.drain() {
            for (i, val) in keys.into_iter().enumerate() {
                row_buffer[i].push(val);
            }
            for (i, agg) in aggs.iter_mut().enumerate() {
                row_buffer[num_key_cols + i].push(agg.finalize());
            }

            if row_buffer[0].len() >= OUTPUT_BATCH_SIZE {
                flush_buffer(
                    &mut row_buffer,
                    &mut seg_writer,
                    column_types,
                    &mut total_rows,
                )?;
            }
        }
    }

    // Flush any remaining rows
    let flush_buffer_final =
        |buf: &mut Vec<Vec<FlexType>>,
         writer: &mut SegmentWriter<Box<dyn sframe_io::vfs::WritableFile>>,
         dtypes: &[FlexTypeEnum],
         total: &mut u64| -> Result<()> {
            if buf[0].is_empty() {
                return Ok(());
            }
            let n = buf[0].len() as u64;
            for (col_idx, col_data) in buf.iter_mut().enumerate() {
                writer.write_column_block(col_idx, col_data, dtypes[col_idx])?;
                col_data.clear();
            }
            *total += n;
            Ok(())
        };
    flush_buffer_final(
        &mut row_buffer,
        &mut seg_writer,
        column_types,
        &mut total_rows,
    )?;

    let segment_sizes = seg_writer.finish()?;

    // Assemble SFrame metadata
    let col_name_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
    sframe_storage::sframe_writer::assemble_sframe_from_segments(
        &*vfs,
        &base_path,
        &col_name_refs,
        column_types,
        &[seg_name],
        &[segment_sizes],
        total_rows,
        &std::collections::HashMap::new(),
    )?;

    Ok(base_path)
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Per-segment in-memory hash table.
struct GroupBySegment {
    groups: HashMap<Vec<FlexType>, Vec<Box<dyn Aggregator>>>,
}

/// Metadata for a single spilled chunk: which cache:// file and how many entries.
struct ChunkInfo {
    path: String,
    entry_count: u64,
}

/// Manages spill files via CacheFs.
///
/// Each flush creates a new `cache://` file containing sorted entries.
/// Files are automatically cleaned up when the CacheFs is dropped.
struct SpillState {
    /// CacheFs instance for ephemeral spill storage.
    cache_fs: Arc<CacheFs>,
    /// Per-segment list of spilled chunks.
    chunks: Vec<Vec<ChunkInfo>>,
}

impl SpillState {
    fn new(num_segments: usize) -> Result<Self> {
        let cache_fs = Arc::new(CacheFs::new()?);
        let chunks = (0..num_segments).map(|_| Vec::new()).collect();
        Ok(SpillState {
            cache_fs,
            chunks,
        })
    }
}

// ---------------------------------------------------------------------------
// Spill / flush
// ---------------------------------------------------------------------------

/// Compute a u64 hash of a key vector.
fn compute_key_hash(key: &[FlexType]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for k in key {
        k.hash(&mut hasher);
    }
    hasher.finish()
}

/// Flush a segment's in-memory groups to a new cache:// spill file.
fn flush_segment(
    groups: &mut HashMap<Vec<FlexType>, Vec<Box<dyn Aggregator>>>,
    seg_id: usize,
    state: &mut SpillState,
) -> Result<()> {
    // Collect and sort entries by (hash, keys) for merge-friendly ordering
    let mut entries: Vec<GroupEntry> = groups
        .drain()
        .map(|(key, mut aggs)| {
            let h = compute_key_hash(&key);
            for agg in aggs.iter_mut() {
                agg.partial_finalize();
            }
            (h, key, aggs)
        })
        .collect();

    entries.sort_by(|a, b| {
        a.0.cmp(&b.0).then_with(|| {
            for (x, y) in a.1.iter().zip(b.1.iter()) {
                let c = x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal);
                if c != std::cmp::Ordering::Equal {
                    return c;
                }
            }
            std::cmp::Ordering::Equal
        })
    });

    let entry_count = entries.len() as u64;

    // Allocate a new cache:// path and write the chunk
    let path = state.cache_fs.alloc_path();
    {
        let mut writer = state.cache_fs.open_cache_write(&path)?;
        for (hash, key, aggs) in &entries {
            write_u64(&mut writer, *hash)?;
            for k in key {
                write_flex_type(&mut writer, k)?;
            }
            for agg in aggs {
                agg.save(&mut writer)?;
            }
        }
        writer.flush_all()?;
    }

    state.chunks[seg_id].push(ChunkInfo { path, entry_count });

    Ok(())
}

// ---------------------------------------------------------------------------
// Merge phase
// ---------------------------------------------------------------------------

/// A chunk reader that deserializes entries from a spill file.
struct ChunkReader {
    reader: Box<dyn sframe_io::vfs::ReadableFile>,
    remaining: u64,
    num_key_cols: usize,
}

impl ChunkReader {
    fn new(
        cache_fs: &CacheFs,
        path: &str,
        entry_count: u64,
        num_key_cols: usize,
    ) -> Result<Self> {
        let reader = cache_fs.open_read(path)?;
        Ok(ChunkReader {
            reader,
            remaining: entry_count,
            num_key_cols,
        })
    }

    /// Read the next entry. Returns None when the chunk is exhausted.
    fn next_entry(&mut self, agg_specs: &[AggSpec]) -> Result<Option<MergeEntry>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;

        let hash = read_u64(&mut self.reader)?;
        let mut keys = Vec::with_capacity(self.num_key_cols);
        for _ in 0..self.num_key_cols {
            keys.push(read_flex_type(&mut self.reader)?);
        }

        let mut aggs: Vec<Box<dyn Aggregator>> = agg_specs
            .iter()
            .map(|spec| spec.aggregator.box_clone())
            .collect();
        for agg in aggs.iter_mut() {
            agg.load(&mut self.reader)?;
        }

        Ok(Some(MergeEntry { hash, keys, aggs }))
    }
}

/// Entry in the k-way merge heap.
struct MergeEntry {
    hash: u64,
    keys: Vec<FlexType>,
    aggs: Vec<Box<dyn Aggregator>>,
}

/// Wrapper for BinaryHeap (min-heap via Reverse ordering).
struct HeapEntry {
    entry: MergeEntry,
    chunk_idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.entry.hash == other.entry.hash && self.entry.keys == other.entry.keys
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap
        other
            .entry
            .hash
            .cmp(&self.entry.hash)
            .then_with(|| {
                for (a, b) in other.entry.keys.iter().zip(self.entry.keys.iter()) {
                    let c = a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
                    if c != std::cmp::Ordering::Equal {
                        return c;
                    }
                }
                std::cmp::Ordering::Equal
            })
    }
}

/// K-way merge of all chunks for a segment, writing directly to buffered output.
// All parameters are distinct merge state; bundling into a struct would not simplify the API.
#[allow(clippy::too_many_arguments)]
fn merge_segment_chunks_buffered(
    state: &mut SpillState,
    seg_id: usize,
    num_key_cols: usize,
    agg_specs: &[AggSpec],
    row_buffer: &mut [Vec<FlexType>],
    seg_writer: &mut SegmentWriter<Box<dyn sframe_io::vfs::WritableFile>>,
    column_types: &[FlexTypeEnum],
    total_rows: &mut u64,
) -> Result<()> {
    let chunks = &state.chunks[seg_id];
    if chunks.is_empty() {
        return Ok(());
    }

    // Open one reader per chunk
    let mut readers: Vec<ChunkReader> = chunks
        .iter()
        .map(|chunk| {
            ChunkReader::new(&state.cache_fs, &chunk.path, chunk.entry_count, num_key_cols)
        })
        .collect::<Result<Vec<_>>>()?;

    // Initialize the min-heap
    let mut heap = BinaryHeap::new();
    for (chunk_idx, reader) in readers.iter_mut().enumerate() {
        if let Some(entry) = reader.next_entry(agg_specs)? {
            heap.push(HeapEntry { entry, chunk_idx });
        }
    }

    // Merge loop
    while let Some(HeapEntry {
        entry: current,
        chunk_idx,
    }) = heap.pop()
    {
        // Refill from the chunk we just consumed
        if let Some(next) = readers[chunk_idx].next_entry(agg_specs)? {
            heap.push(HeapEntry {
                entry: next,
                chunk_idx,
            });
        }

        // Collect all entries with the same key
        let mut combined_aggs = current.aggs;
        let current_hash = current.hash;
        let current_keys = current.keys;

        while let Some(top) = heap.peek() {
            if top.entry.hash == current_hash && top.entry.keys == current_keys {
                let HeapEntry {
                    entry: dup,
                    chunk_idx: dup_chunk,
                } = heap.pop().unwrap();
                // Merge aggregators
                for (i, agg) in dup.aggs.iter().enumerate() {
                    combined_aggs[i].merge(agg.as_ref());
                }
                // Refill from the chunk we consumed
                if let Some(next) = readers[dup_chunk].next_entry(agg_specs)? {
                    heap.push(HeapEntry {
                        entry: next,
                        chunk_idx: dup_chunk,
                    });
                }
            } else {
                break;
            }
        }

        // Finalize and emit to buffer
        for (i, val) in current_keys.into_iter().enumerate() {
            row_buffer[i].push(val);
        }
        for (i, agg) in combined_aggs.iter_mut().enumerate() {
            row_buffer[num_key_cols + i].push(agg.finalize());
        }

        // Flush buffer if full
        if row_buffer[0].len() >= OUTPUT_BATCH_SIZE {
            let n = row_buffer[0].len() as u64;
            for (col_idx, col_data) in row_buffer.iter_mut().enumerate() {
                seg_writer.write_column_block(col_idx, col_data, column_types[col_idx])?;
                col_data.clear();
            }
            *total_rows += n;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::aggregators::*;
    use crate::batch::SFrameRows;
    use crate::execute::{compile, materialize_sync};
    use crate::planner::PlannerNode;

    /// Helper: run groupby and read result back as SFrameRows for assertions.
    fn groupby_to_rows(
        plan: &Arc<PlannerNode>,
        key_columns: &[usize],
        agg_specs: &[AggSpec],
        column_names: &[String],
        column_types: &[FlexTypeEnum],
    ) -> SFrameRows {
        let path = groupby(plan, key_columns, agg_specs, column_names, column_types).unwrap();
        let result = read_cache_sframe(&path);
        let _ = global_cache_fs().remove_dir(&path);
        result
    }

    /// Helper: run groupby_with_config and read result back.
    fn groupby_with_config_to_rows(
        plan: &Arc<PlannerNode>,
        key_columns: &[usize],
        agg_specs: &[AggSpec],
        column_names: &[String],
        column_types: &[FlexTypeEnum],
        num_segments: usize,
        per_segment_buffer: usize,
    ) -> SFrameRows {
        let path = groupby_with_config(
            plan,
            key_columns,
            agg_specs,
            column_names,
            column_types,
            num_segments,
            per_segment_buffer,
        )
        .unwrap();
        let result = read_cache_sframe(&path);
        let _ = global_cache_fs().remove_dir(&path);
        result
    }

    /// Read an SFrame from a cache:// path into SFrameRows.
    fn read_cache_sframe(path: &str) -> SFrameRows {
        use sframe_storage::sframe_reader::SFrameReader;
        let vfs = ArcCacheFsVfs(global_cache_fs().clone());
        let reader = SFrameReader::open_with_fs(&vfs, path).unwrap();
        let col_names = reader.column_names().to_vec();
        let col_types: Vec<FlexTypeEnum> = reader
            .group_index
            .columns
            .iter()
            .map(|c| c.dtype)
            .collect();
        let num_rows = reader.num_rows();

        if num_rows == 0 {
            return SFrameRows::empty(&col_types);
        }

        // cache:// paths are auto-detected by the source compiler
        let source = PlannerNode::sframe_source(path, col_names, col_types, num_rows);
        let stream = compile(&source).unwrap();
        materialize_sync(stream).unwrap()
    }

    #[test]
    fn test_groupby_simple() {
        let rows = vec![
            vec![FlexType::String("Phoenix".into()), FlexType::Integer(10)],
            vec![
                FlexType::String("Scottsdale".into()),
                FlexType::Integer(20),
            ],
            vec![FlexType::String("Phoenix".into()), FlexType::Integer(30)],
            vec![
                FlexType::String("Scottsdale".into()),
                FlexType::Integer(40),
            ],
        ];
        let dtypes = [FlexTypeEnum::String, FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let plan = PlannerNode::materialized(batch);

        let col_names: Vec<String> = vec![
            "city".into(),
            "total_score".into(),
            "count".into(),
            "avg_score".into(),
        ];
        let col_types = vec![
            FlexTypeEnum::String,
            FlexTypeEnum::Integer,
            FlexTypeEnum::Integer,
            FlexTypeEnum::Float,
        ];

        let result = groupby_to_rows(
            &plan,
            &[0],
            &[
                AggSpec::sum(1, "total_score"),
                AggSpec::count(1, "count"),
                AggSpec::mean(1, "avg_score"),
            ],
            &col_names,
            &col_types,
        );

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 4);

        let mut results: HashMap<String, Vec<FlexType>> = HashMap::new();
        for i in 0..result.num_rows() {
            let row = result.row(i);
            let city = match &row[0] {
                FlexType::String(s) => s.to_string(),
                other => panic!("Expected String, got {other:?}"),
            };
            results.insert(city, row[1..].to_vec());
        }

        let phoenix = &results["Phoenix"];
        assert_eq!(phoenix[0], FlexType::Integer(40));
        assert_eq!(phoenix[1], FlexType::Integer(2));
        match &phoenix[2] {
            FlexType::Float(v) => assert!((v - 20.0).abs() < 1e-10),
            other => panic!("Expected Float, got {other:?}"),
        }

        let scottsdale = &results["Scottsdale"];
        assert_eq!(scottsdale[0], FlexType::Integer(60));
        assert_eq!(scottsdale[1], FlexType::Integer(2));
        match &scottsdale[2] {
            FlexType::Float(v) => assert!((v - 30.0).abs() < 1e-10),
            other => panic!("Expected Float, got {other:?}"),
        }
    }

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{manifest}/../../samples")
    }

    #[test]
    fn test_groupby_business_sf() {
        use sframe_storage::sframe_reader::SFrameReader;

        let path = format!("{}/business.sf", samples_dir());
        let reader = SFrameReader::open(&path).unwrap();
        let col_names = reader.column_names().to_vec();
        let col_types: Vec<FlexTypeEnum> = reader
            .group_index
            .columns
            .iter()
            .map(|c| c.dtype)
            .collect();
        let num_rows = reader.num_rows();

        let source = PlannerNode::sframe_source(&path, col_names.clone(), col_types.clone(), num_rows);

        // Key column is index 10 (state), agg on column 0
        let key_type = col_types[10];
        let out_col_names: Vec<String> = vec![col_names[10].clone(), "count".into()];
        let out_col_types = vec![key_type, FlexTypeEnum::Integer];

        let result = groupby_to_rows(
            &source,
            &[10],
            &[AggSpec::count(0, "count")],
            &out_col_names,
            &out_col_types,
        );

        assert!(result.num_rows() > 0);

        let mut total = 0i64;
        for i in 0..result.num_rows() {
            let row = result.row(i);
            match &row[1] {
                FlexType::Integer(c) => total += c,
                other => panic!("Expected Integer count, got {other:?}"),
            }
        }
        assert_eq!(total, 11536);
    }

    /// Test that forces spilling by using a very small per-segment buffer.
    /// Verifies the spill+merge path produces identical results to the in-memory path.
    #[test]
    fn test_groupby_forced_spill() {
        let mut rows = Vec::new();
        for i in 0..100 {
            let city = format!("city_{}", i % 10);
            rows.push(vec![
                FlexType::String(city.into()),
                FlexType::Integer(i as i64),
            ]);
        }
        let dtypes = [FlexTypeEnum::String, FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let plan = PlannerNode::materialized(batch);

        let col_names: Vec<String> = vec![
            "city".into(),
            "total".into(),
            "count".into(),
            "avg".into(),
            "min".into(),
            "max".into(),
        ];
        let col_types = vec![
            FlexTypeEnum::String,
            FlexTypeEnum::Integer,
            FlexTypeEnum::Integer,
            FlexTypeEnum::Float,
            FlexTypeEnum::Integer,
            FlexTypeEnum::Integer,
        ];
        let agg_specs = vec![
            AggSpec::sum(1, "total"),
            AggSpec::count(1, "count"),
            AggSpec::mean(1, "avg"),
            AggSpec::min(1, "min"),
            AggSpec::max(1, "max"),
        ];

        // Run with forced spilling (per_segment_buffer = 2)
        let spill_result = groupby_with_config_to_rows(
            &plan,
            &[0],
            &agg_specs,
            &col_names,
            &col_types,
            4,
            2,
        );

        // Run with in-memory path (large buffer, no spills)
        let mem_result = groupby_to_rows(
            &plan,
            &[0],
            &agg_specs,
            &col_names,
            &col_types,
        );

        assert_eq!(spill_result.num_rows(), mem_result.num_rows());
        assert_eq!(spill_result.num_rows(), 10);

        let collect_map = |result: &SFrameRows| -> HashMap<String, Vec<FlexType>> {
            let mut map = HashMap::new();
            for i in 0..result.num_rows() {
                let row = result.row(i);
                let city = match &row[0] {
                    FlexType::String(s) => s.to_string(),
                    other => panic!("Expected String, got {other:?}"),
                };
                map.insert(city, row[1..].to_vec());
            }
            map
        };

        let spill_map = collect_map(&spill_result);
        let mem_map = collect_map(&mem_result);

        for (city, mem_vals) in &mem_map {
            let spill_vals = spill_map
                .get(city)
                .unwrap_or_else(|| panic!("City {city} missing from spill result"));
            for (i, (m, s)) in mem_vals.iter().zip(spill_vals.iter()).enumerate() {
                match (m, s) {
                    (FlexType::Integer(a), FlexType::Integer(b)) => {
                        assert_eq!(a, b, "Mismatch for city {city} agg {i}");
                    }
                    (FlexType::Float(a), FlexType::Float(b)) => {
                        assert!(
                            (a - b).abs() < 1e-10,
                            "Float mismatch for city {city} agg {i}: {a} vs {b}"
                        );
                    }
                    _ => {
                        assert_eq!(m, s, "Type mismatch for city {city} agg {i}");
                    }
                }
            }
        }
    }

    /// Test groupby with empty input.
    #[test]
    fn test_groupby_empty() {
        let dtypes = [FlexTypeEnum::String, FlexTypeEnum::Integer];
        let batch = SFrameRows::empty(&dtypes);
        let plan = PlannerNode::materialized(batch);

        let col_names: Vec<String> = vec!["key".into(), "total".into()];
        let col_types = vec![FlexTypeEnum::String, FlexTypeEnum::Integer];

        let path = groupby(
            &plan,
            &[0],
            &[AggSpec::sum(1, "total")],
            &col_names,
            &col_types,
        )
        .unwrap();

        let result = read_cache_sframe(&path);
        let _ = global_cache_fs().remove_dir(&path);

        assert_eq!(result.num_rows(), 0);
    }

    /// Test groupby where all rows have the same key.
    #[test]
    fn test_groupby_single_group_spill() {
        let mut rows = Vec::new();
        for i in 0..20 {
            rows.push(vec![
                FlexType::String("only".into()),
                FlexType::Integer(i as i64),
            ]);
        }
        let dtypes = [FlexTypeEnum::String, FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let plan = PlannerNode::materialized(batch);

        let col_names: Vec<String> = vec!["key".into(), "total".into(), "count".into()];
        let col_types = vec![
            FlexTypeEnum::String,
            FlexTypeEnum::Integer,
            FlexTypeEnum::Integer,
        ];

        let result = groupby_with_config_to_rows(
            &plan,
            &[0],
            &[AggSpec::sum(1, "total"), AggSpec::count(1, "count")],
            &col_names,
            &col_types,
            2,
            3,
        );

        assert_eq!(result.num_rows(), 1);
        let row = result.row(0);
        assert_eq!(row[0], FlexType::String("only".into()));
        assert_eq!(row[1], FlexType::Integer(190));
        assert_eq!(row[2], FlexType::Integer(20));
    }
}
