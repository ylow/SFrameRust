//! Hash-partitioned external memory groupby algorithm.
//!
//! Two-phase algorithm with bounded memory:
//! - Phase 1: Ingest rows, hash-partition into segments, spill to disk when full
//! - Phase 2: Per-segment k-way merge of spilled chunks, combine matching keys
//!
//! Spill files are stored in `cache://` via the CacheFs VFS backend, which
//! provides automatic cleanup when the SpillState is dropped.

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use futures::StreamExt;

use sframe_io::cache_fs::CacheFs;
use sframe_io::vfs::VirtualFileSystem;
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};
use sframe_types::serialization::{read_flex_type, read_u64, write_flex_type, write_u64};

use crate::algorithms::aggregators::AggSpec;
use crate::batch::{ColumnData, SFrameRows};
use crate::execute::BatchStream;
use crate::planner::Aggregator;

/// Default number of hash-partition segments.
const DEFAULT_NUM_SEGMENTS: usize = 16;

/// Default total buffer capacity in distinct keys across all segments.
const DEFAULT_BUFFER_NUM_ROWS: usize = 1_048_576; // ~1M keys

/// Perform a groupby operation on a batch stream.
///
/// Groups by `key_columns` and applies `agg_specs` to produce the aggregated result.
/// Returns a single SFrameRows batch with one row per group.
pub async fn groupby(
    input: BatchStream,
    key_columns: &[usize],
    agg_specs: &[AggSpec],
) -> Result<SFrameRows> {
    groupby_with_config(
        input,
        key_columns,
        agg_specs,
        DEFAULT_NUM_SEGMENTS,
        DEFAULT_BUFFER_NUM_ROWS / DEFAULT_NUM_SEGMENTS,
    )
    .await
}

/// Groupby with explicit configuration for testing.
pub(crate) async fn groupby_with_config(
    mut input: BatchStream,
    key_columns: &[usize],
    agg_specs: &[AggSpec],
    num_segments: usize,
    per_segment_buffer: usize,
) -> Result<SFrameRows> {
    let num_key_cols = key_columns.len();
    let num_agg_specs = agg_specs.len();

    // Initialize segments
    let mut segments: Vec<GroupBySegment> = (0..num_segments)
        .map(|_| GroupBySegment {
            groups: HashMap::new(),
        })
        .collect();
    let mut spill_state: Option<SpillState> = None;

    // Phase 1: Ingest rows
    while let Some(batch_result) = input.next().await {
        let batch = batch_result?;
        for row_idx in 0..batch.num_rows() {
            // Extract group key
            let key: Vec<FlexTypeKey> = key_columns
                .iter()
                .map(|&col| FlexTypeKey(batch.column(col).get(row_idx)))
                .collect();

            // Hash the key to determine segment
            let key_hash = compute_key_hash(&key);
            let seg_id = (key_hash as usize) % num_segments;

            let segment = &mut segments[seg_id];

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
                let state = spill_state.get_or_insert_with(|| {
                    SpillState::new(num_segments).expect("Failed to create CacheFs for spill")
                });
                flush_segment(&mut segment.groups, seg_id, state)?;
            }
        }
    }

    // Phase 2: Produce output
    let mut all_keys: Vec<Vec<FlexType>> = Vec::new();
    let mut all_aggs: Vec<Vec<FlexType>> = Vec::new();

    for seg_id in 0..num_segments {
        let segment = &mut segments[seg_id];

        if let Some(state) = &mut spill_state {
            if !state.chunks[seg_id].is_empty() {
                // This segment had spills — flush remaining in-memory entries as final chunk
                if !segment.groups.is_empty() {
                    flush_segment(&mut segment.groups, seg_id, state)?;
                }

                // K-way merge of all chunks for this segment
                merge_segment_chunks(
                    state,
                    seg_id,
                    num_key_cols,
                    agg_specs,
                    &mut all_keys,
                    &mut all_aggs,
                )?;
                continue;
            }
        }

        // Fast path: no spills for this segment, finalize in-memory
        for (key, mut aggs) in segment.groups.drain() {
            let keys: Vec<FlexType> = key.into_iter().map(|k| k.0).collect();
            let mut row_aggs = Vec::with_capacity(num_agg_specs);
            for agg in aggs.iter_mut() {
                row_aggs.push(agg.finalize());
            }
            all_keys.push(keys);
            all_aggs.push(row_aggs);
        }
    }

    // Build output
    if all_keys.is_empty() {
        let mut dtypes = Vec::new();
        for _ in 0..num_key_cols {
            dtypes.push(FlexTypeEnum::Integer);
        }
        for _ in agg_specs {
            dtypes.push(FlexTypeEnum::Integer);
        }
        return Ok(SFrameRows::empty(&dtypes));
    }

    // Determine output types from actual values
    let key_types: Vec<FlexTypeEnum> = (0..num_key_cols)
        .map(|i| {
            all_keys
                .iter()
                .map(|row| row[i].type_enum())
                .find(|&t| t != FlexTypeEnum::Undefined)
                .unwrap_or(FlexTypeEnum::Integer)
        })
        .collect();

    let agg_types: Vec<FlexTypeEnum> = (0..num_agg_specs)
        .map(|i| {
            all_aggs
                .iter()
                .map(|row| row[i].type_enum())
                .find(|&t| t != FlexTypeEnum::Undefined)
                .unwrap_or(FlexTypeEnum::Integer)
        })
        .collect();

    let mut columns: Vec<ColumnData> = key_types
        .iter()
        .chain(agg_types.iter())
        .map(|&dt| ColumnData::empty(dt))
        .collect();

    for (group_idx, keys) in all_keys.iter().enumerate() {
        for (i, val) in keys.iter().enumerate() {
            columns[i].push(val)?;
        }
        for (i, val) in all_aggs[group_idx].iter().enumerate() {
            columns[num_key_cols + i].push(val)?;
        }
    }

    SFrameRows::new(columns)
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Per-segment in-memory hash table.
struct GroupBySegment {
    groups: HashMap<Vec<FlexTypeKey>, Vec<Box<dyn Aggregator>>>,
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

/// Wrapper around FlexType to implement Hash + Eq for use as HashMap keys.
#[derive(Clone, Debug)]
pub(crate) struct FlexTypeKey(pub FlexType);

impl PartialEq for FlexTypeKey {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (FlexType::Integer(a), FlexType::Integer(b)) => a == b,
            (FlexType::Float(a), FlexType::Float(b)) => a.to_bits() == b.to_bits(),
            (FlexType::String(a), FlexType::String(b)) => a == b,
            (FlexType::Undefined, FlexType::Undefined) => true,
            _ => false,
        }
    }
}

impl Eq for FlexTypeKey {}

impl Hash for FlexTypeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(&self.0).hash(state);
        match &self.0 {
            FlexType::Integer(i) => i.hash(state),
            FlexType::Float(f) => f.to_bits().hash(state),
            FlexType::String(s) => s.hash(state),
            FlexType::Undefined => {}
            FlexType::Vector(v) => {
                v.len().hash(state);
                for f in v.iter() {
                    f.to_bits().hash(state);
                }
            }
            FlexType::List(l) => l.len().hash(state),
            FlexType::Dict(d) => d.len().hash(state),
            FlexType::DateTime(dt) => {
                dt.posix_timestamp.hash(state);
                dt.microsecond.hash(state);
            }
        }
    }
}

/// Ordering for merge: compare by (hash, keys).
impl PartialOrd for FlexTypeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FlexTypeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let tag = |ft: &FlexType| -> u8 {
            match ft {
                FlexType::Undefined => 0,
                FlexType::Integer(_) => 1,
                FlexType::Float(_) => 2,
                FlexType::String(_) => 3,
                FlexType::Vector(_) => 4,
                FlexType::List(_) => 5,
                FlexType::Dict(_) => 6,
                FlexType::DateTime(_) => 7,
            }
        };
        let t1 = tag(&self.0);
        let t2 = tag(&other.0);
        if t1 != t2 {
            return t1.cmp(&t2);
        }
        match (&self.0, &other.0) {
            (FlexType::Integer(a), FlexType::Integer(b)) => a.cmp(b),
            (FlexType::Float(a), FlexType::Float(b)) => a.to_bits().cmp(&b.to_bits()),
            (FlexType::String(a), FlexType::String(b)) => a.cmp(b),
            (FlexType::Undefined, FlexType::Undefined) => std::cmp::Ordering::Equal,
            _ => format!("{:?}", self.0).cmp(&format!("{:?}", other.0)),
        }
    }
}

// ---------------------------------------------------------------------------
// Spill / flush
// ---------------------------------------------------------------------------

/// Compute a u64 hash of a key vector.
fn compute_key_hash(key: &[FlexTypeKey]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for k in key {
        k.hash(&mut hasher);
    }
    hasher.finish()
}

/// Flush a segment's in-memory groups to a new cache:// spill file.
fn flush_segment(
    groups: &mut HashMap<Vec<FlexTypeKey>, Vec<Box<dyn Aggregator>>>,
    seg_id: usize,
    state: &mut SpillState,
) -> Result<()> {
    // Collect and sort entries by (hash, keys) for merge-friendly ordering
    let mut entries: Vec<(u64, Vec<FlexTypeKey>, Vec<Box<dyn Aggregator>>)> = groups
        .drain()
        .map(|(key, mut aggs)| {
            let h = compute_key_hash(&key);
            for agg in aggs.iter_mut() {
                agg.partial_finalize();
            }
            (h, key, aggs)
        })
        .collect();

    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let entry_count = entries.len() as u64;

    // Allocate a new cache:// path and write the chunk
    let path = state.cache_fs.alloc_path();
    {
        let mut writer = state.cache_fs.open_cache_write(&path)?;
        for (hash, key, aggs) in &entries {
            write_u64(&mut writer, *hash)?;
            for k in key {
                write_flex_type(&mut writer, &k.0)?;
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
            keys.push(FlexTypeKey(read_flex_type(&mut self.reader)?));
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
    keys: Vec<FlexTypeKey>,
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
            .then_with(|| other.entry.keys.cmp(&self.entry.keys))
    }
}

/// K-way merge of all chunks for a segment, combining matching keys.
fn merge_segment_chunks(
    state: &mut SpillState,
    seg_id: usize,
    num_key_cols: usize,
    agg_specs: &[AggSpec],
    out_keys: &mut Vec<Vec<FlexType>>,
    out_aggs: &mut Vec<Vec<FlexType>>,
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

        // Finalize and emit
        let keys: Vec<FlexType> = current_keys.into_iter().map(|k| k.0).collect();
        let mut row_aggs = Vec::with_capacity(agg_specs.len());
        for agg in combined_aggs.iter_mut() {
            row_aggs.push(agg.finalize());
        }
        out_keys.push(keys);
        out_aggs.push(row_aggs);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::aggregators::*;
    use crate::execute::compile;
    use crate::planner::PlannerNode;
    use futures::stream;

    #[tokio::test]
    async fn test_groupby_simple() {
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

        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));

        let result = groupby(
            input,
            &[0],
            &[
                AggSpec::sum(1, "total_score"),
                AggSpec::count(1, "count"),
                AggSpec::mean(1, "avg_score"),
            ],
        )
        .await
        .unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 4);

        let mut results: HashMap<String, Vec<FlexType>> = HashMap::new();
        for i in 0..result.num_rows() {
            let row = result.row(i);
            let city = match &row[0] {
                FlexType::String(s) => s.to_string(),
                other => panic!("Expected String, got {:?}", other),
            };
            results.insert(city, row[1..].to_vec());
        }

        let phoenix = &results["Phoenix"];
        assert_eq!(phoenix[0], FlexType::Integer(40));
        assert_eq!(phoenix[1], FlexType::Integer(2));
        match &phoenix[2] {
            FlexType::Float(v) => assert!((v - 20.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }

        let scottsdale = &results["Scottsdale"];
        assert_eq!(scottsdale[0], FlexType::Integer(60));
        assert_eq!(scottsdale[1], FlexType::Integer(2));
        match &scottsdale[2] {
            FlexType::Float(v) => assert!((v - 30.0).abs() < 1e-10),
            other => panic!("Expected Float, got {:?}", other),
        }
    }

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../samples", manifest)
    }

    #[tokio::test]
    async fn test_groupby_business_sf() {
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

        let source = PlannerNode::sframe_source(&path, col_names, col_types, num_rows);
        let stream = compile(&source).unwrap();

        let result = groupby(stream, &[10], &[AggSpec::count(0, "count")])
            .await
            .unwrap();

        assert!(result.num_rows() > 0);

        let mut total = 0i64;
        for i in 0..result.num_rows() {
            let row = result.row(i);
            match &row[1] {
                FlexType::Integer(c) => total += c,
                other => panic!("Expected Integer count, got {:?}", other),
            }
        }
        assert_eq!(total, 11536);
    }

    /// Test that forces spilling by using a very small per-segment buffer.
    /// Verifies the spill+merge path produces identical results to the in-memory path.
    #[tokio::test]
    async fn test_groupby_forced_spill() {
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
        let batch2 = batch.clone();

        // Run with forced spilling (per_segment_buffer = 2)
        let input: BatchStream = Box::pin(stream::once(async move { Ok(batch) }));
        let spill_result = groupby_with_config(
            input,
            &[0],
            &[
                AggSpec::sum(1, "total"),
                AggSpec::count(1, "count"),
                AggSpec::mean(1, "avg"),
                AggSpec::min(1, "min"),
                AggSpec::max(1, "max"),
            ],
            4,
            2,
        )
        .await
        .unwrap();

        // Run with in-memory path (large buffer, no spills)
        let input2: BatchStream = Box::pin(stream::once(async move { Ok(batch2) }));
        let mem_result = groupby(
            input2,
            &[0],
            &[
                AggSpec::sum(1, "total"),
                AggSpec::count(1, "count"),
                AggSpec::mean(1, "avg"),
                AggSpec::min(1, "min"),
                AggSpec::max(1, "max"),
            ],
        )
        .await
        .unwrap();

        assert_eq!(spill_result.num_rows(), mem_result.num_rows());
        assert_eq!(spill_result.num_rows(), 10);

        let collect_map = |result: &SFrameRows| -> HashMap<String, Vec<FlexType>> {
            let mut map = HashMap::new();
            for i in 0..result.num_rows() {
                let row = result.row(i);
                let city = match &row[0] {
                    FlexType::String(s) => s.to_string(),
                    other => panic!("Expected String, got {:?}", other),
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
                .unwrap_or_else(|| panic!("City {} missing from spill result", city));
            for (i, (m, s)) in mem_vals.iter().zip(spill_vals.iter()).enumerate() {
                match (m, s) {
                    (FlexType::Integer(a), FlexType::Integer(b)) => {
                        assert_eq!(a, b, "Mismatch for city {} agg {}", city, i);
                    }
                    (FlexType::Float(a), FlexType::Float(b)) => {
                        assert!(
                            (a - b).abs() < 1e-10,
                            "Float mismatch for city {} agg {}: {} vs {}",
                            city, i, a, b
                        );
                    }
                    _ => {
                        assert_eq!(m, s, "Type mismatch for city {} agg {}", city, i);
                    }
                }
            }
        }
    }

    /// Test groupby with empty input.
    #[tokio::test]
    async fn test_groupby_empty() {
        let dtypes = [FlexTypeEnum::String, FlexTypeEnum::Integer];
        let batch = SFrameRows::empty(&dtypes);

        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));
        let result = groupby(input, &[0], &[AggSpec::sum(1, "total")])
            .await
            .unwrap();

        assert_eq!(result.num_rows(), 0);
    }

    /// Test groupby where all rows have the same key.
    #[tokio::test]
    async fn test_groupby_single_group_spill() {
        let mut rows = Vec::new();
        for i in 0..20 {
            rows.push(vec![
                FlexType::String("only".into()),
                FlexType::Integer(i as i64),
            ]);
        }
        let dtypes = [FlexTypeEnum::String, FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));
        let result = groupby_with_config(
            input,
            &[0],
            &[AggSpec::sum(1, "total"), AggSpec::count(1, "count")],
            2,
            3,
        )
        .await
        .unwrap();

        assert_eq!(result.num_rows(), 1);
        let row = result.row(0);
        assert_eq!(row[0], FlexType::String("only".into()));
        assert_eq!(row[1], FlexType::Integer(190));
        assert_eq!(row[2], FlexType::Integer(20));
    }
}
