//! Columnar sort with EC-Sort optimization and external merge sort.
//!
//! Uses index-based sorting: only key column values are accessed during
//! comparisons, and the permutation is applied to all columns in a single
//! pass via `take()`. This is the in-memory equivalent of the C++ EC-Sort
//! algorithm which avoids shuffling large value columns.
//!
//! When data exceeds the sort memory budget (`SFRAME_SORT_BUFFER_SIZE`),
//! an external merge sort is used: input is streamed into sorted runs on
//! CacheFs, then k-way merged via a min-heap into a `BatchIterator`.

use std::sync::Arc;

use rayon::prelude::*;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_storage::segment_reader::{CachedSegmentReader, SegmentReader};
use sframe_storage::segment_writer::SegmentWriter;
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::{ColumnData, SFrameRows};
use crate::execute::{BatchCo, BatchCommand, BatchIterator, BatchResponse};

/// Sort order for a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Sort specification: column index + direction.
#[derive(Debug, Clone)]
pub struct SortKey {
    pub column: usize,
    pub order: SortOrder,
}

impl SortKey {
    pub fn asc(column: usize) -> Self {
        SortKey {
            column,
            order: SortOrder::Ascending,
        }
    }

    pub fn desc(column: usize) -> Self {
        SortKey {
            column,
            order: SortOrder::Descending,
        }
    }
}

/// Metadata for a sorted run stored as a segment on CacheFs.
struct SortedRunInfo {
    path: String,
    num_rows: u64,
}

/// RAII guard that removes the sort scratch directory from CacheFs on drop.
struct SortCleanup {
    base_path: String,
}

impl Drop for SortCleanup {
    fn drop(&mut self) {
        let _ = global_cache_fs().remove_dir(&self.base_path);
    }
}

/// Sort the buffer in memory and write the sorted data as a segment to CacheFs.
///
/// Uses index-based sorting to avoid creating a full sorted copy.
/// Data is written in `source_batch_size` chunks so the segment has
/// multiple blocks for efficient sub-range reads during merge.
fn spill_sorted_run(
    buffer: &SFrameRows,
    keys: &[SortKey],
    dtypes: &[FlexTypeEnum],
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    run_id: usize,
) -> Result<SortedRunInfo> {
    let indices = build_sort_indices(buffer, keys);
    let seg_path = format!("{base_path}/run_{run_id:04}");
    let file = vfs.open_write(&seg_path)?;
    let mut seg_writer = SegmentWriter::new(file, dtypes.len());

    let chunk_size = sframe_config::global().source_batch_size;
    for chunk in indices.chunks(chunk_size) {
        for (col_idx, col) in buffer.columns().iter().enumerate() {
            let values: Vec<FlexType> = chunk.iter().map(|&i| col.get(i)).collect();
            seg_writer.write_column_block(col_idx, &values, dtypes[col_idx])?;
        }
    }

    seg_writer.finish()?;
    Ok(SortedRunInfo {
        path: seg_path,
        num_rows: buffer.num_rows() as u64,
    })
}

/// Reads sorted rows from a spilled segment in batches.
///
/// Buffers `read_ahead` rows at a time for sequential access. Usage pattern:
/// check `is_exhausted()`, read `current_value(col)`, call `advance()`.
struct RunCursor {
    reader: CachedSegmentReader,
    all_columns: Vec<usize>,
    total_rows: u64,
    next_read_row: u64,
    buffer: Vec<Vec<FlexType>>,
    cursor: usize,
    buffer_len: usize,
    read_ahead: u64,
}

impl RunCursor {
    fn open(
        vfs: &dyn VirtualFileSystem,
        path: &str,
        dtypes: &[FlexTypeEnum],
        total_rows: u64,
        read_ahead: u64,
    ) -> Result<Self> {
        let file = vfs.open_read(path)?;
        let file_size = file.size()?;
        let seg_reader = SegmentReader::open(Box::new(file), file_size, dtypes.to_vec())?;
        let max_blocks = sframe_config::global().max_blocks_in_cache;
        let reader = CachedSegmentReader::new(seg_reader, max_blocks);
        let ncols = dtypes.len();

        let mut cursor = RunCursor {
            reader,
            all_columns: (0..ncols).collect(),
            total_rows,
            next_read_row: 0,
            buffer: Vec::new(),
            cursor: 0,
            buffer_len: 0,
            read_ahead,
        };
        if total_rows > 0 {
            cursor.refill()?;
        }
        Ok(cursor)
    }

    fn refill(&mut self) -> Result<()> {
        if self.next_read_row >= self.total_rows {
            self.buffer.clear();
            self.buffer_len = 0;
            return Ok(());
        }
        let end = (self.next_read_row + self.read_ahead).min(self.total_rows);
        self.buffer = self.reader.read_columns_rows(
            &self.all_columns,
            self.next_read_row,
            end,
        )?;
        self.cursor = 0;
        self.buffer_len = if self.buffer.is_empty() { 0 } else { self.buffer[0].len() };
        self.next_read_row = end;
        Ok(())
    }

    fn is_exhausted(&self) -> bool {
        self.cursor >= self.buffer_len && self.next_read_row >= self.total_rows
    }

    fn current_value(&self, col: usize) -> &FlexType {
        &self.buffer[col][self.cursor]
    }

    fn advance(&mut self) -> Result<()> {
        self.cursor += 1;
        if self.cursor >= self.buffer_len && self.next_read_row < self.total_rows {
            self.refill()?;
        }
        Ok(())
    }
}

/// Entry for the k-way merge min-heap.
///
/// `BinaryHeap` is a max-heap, so `Ord` is reversed to get min-heap behavior.
struct MergeHeapEntry {
    run_idx: usize,
    key_values: Vec<FlexType>,
    descending: Vec<bool>,
}

impl Ord for MergeHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for (i, (a, b)) in self
            .key_values
            .iter()
            .zip(other.key_values.iter())
            .enumerate()
        {
            let mut cmp = compare_flex_type(a, b);
            if self.descending[i] {
                cmp = cmp.reverse();
            }
            if cmp != std::cmp::Ordering::Equal {
                return cmp.reverse(); // reverse for min-heap
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl PartialOrd for MergeHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for MergeHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for MergeHeapEntry {}

/// k-way merge of sorted runs into a BatchIterator.
///
/// Opens a RunCursor for each run, seeds a min-heap with the first row
/// from each, then repeatedly pops the minimum and emits output batches.
/// The SortCleanup guard is captured by the producer closure so that
/// CacheFs scratch files are cleaned up when the iterator is dropped.
fn merge_sorted_runs(
    runs: Vec<SortedRunInfo>,
    keys: &[SortKey],
    dtypes: &[FlexTypeEnum],
    vfs: Arc<dyn VirtualFileSystem>,
    base_path: String,
) -> Result<BatchIterator> {
    use std::collections::BinaryHeap;

    let keys = keys.to_vec();
    let dtypes = dtypes.to_vec();
    let ncols = dtypes.len();
    let batch_size = sframe_config::global().source_batch_size;

    let descending: Vec<bool> = keys
        .iter()
        .map(|k| k.order == SortOrder::Descending)
        .collect();

    let key_columns: Vec<usize> = keys.iter().map(|k| k.column).collect();

    Ok(BatchIterator::new(move |co: BatchCo| async move {
        let _cleanup = SortCleanup {
            base_path: base_path.clone(),
        };

        let mut cursors: Vec<RunCursor> = Vec::with_capacity(runs.len());
        for run in &runs {
            match RunCursor::open(&*vfs, &run.path, &dtypes, run.num_rows, batch_size as u64) {
                Ok(c) => cursors.push(c),
                Err(e) => {
                    co.yield_(BatchResponse::Ready).await;
                    co.yield_(BatchResponse::Batch(Err(e))).await;
                    return;
                }
            }
        }

        let mut heap = BinaryHeap::new();
        for (i, cursor) in cursors.iter().enumerate() {
            if !cursor.is_exhausted() {
                let kv: Vec<FlexType> = key_columns
                    .iter()
                    .map(|&c| cursor.current_value(c).clone())
                    .collect();
                heap.push(MergeHeapEntry {
                    run_idx: i,
                    key_values: kv,
                    descending: descending.clone(),
                });
            }
        }

        let mut cmd = co.yield_(BatchResponse::Ready).await;

        let mut out_cols: Vec<Vec<FlexType>> =
            (0..ncols).map(|_| Vec::with_capacity(batch_size)).collect();

        while let Some(entry) = heap.pop() {
            if !matches!(cmd, BatchCommand::NextBatch) {
                break;
            }

            // Collect the full row from this run
            let run = &cursors[entry.run_idx];
            for (col, out_col) in out_cols.iter_mut().enumerate().take(ncols) {
                out_col.push(run.current_value(col).clone());
            }

            // Advance cursor
            let run = &mut cursors[entry.run_idx];
            if let Err(e) = run.advance() {
                co.yield_(BatchResponse::Batch(Err(e))).await;
                return;
            }

            // Re-insert into heap if run has more data
            if !run.is_exhausted() {
                let kv: Vec<FlexType> = key_columns
                    .iter()
                    .map(|&c| run.current_value(c).clone())
                    .collect();
                heap.push(MergeHeapEntry {
                    run_idx: entry.run_idx,
                    key_values: kv,
                    descending: descending.clone(),
                });
            }

            // Emit batch when full
            if out_cols[0].len() >= batch_size {
                let columns: Vec<ColumnData> = out_cols
                    .iter()
                    .zip(dtypes.iter())
                    .map(|(data, &dt)| ColumnData::from_flex_slice(data, dt))
                    .collect();
                match SFrameRows::new(columns) {
                    Ok(batch) => cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await,
                    Err(e) => {
                        co.yield_(BatchResponse::Batch(Err(e))).await;
                        return;
                    }
                }
                out_cols = (0..ncols).map(|_| Vec::with_capacity(batch_size)).collect();
            }
        }

        // Emit remaining rows
        if !out_cols.is_empty() && !out_cols[0].is_empty() {
            let columns: Vec<ColumnData> = out_cols
                .iter()
                .zip(dtypes.iter())
                .map(|(data, &dt)| ColumnData::from_flex_slice(data, dt))
                .collect();
            if let Ok(batch) = SFrameRows::new(columns) {
                co.yield_(BatchResponse::Batch(Ok(batch))).await;
            }
        }
    }))
}

/// Sort a batch stream by the given keys.
///
/// Streams input batches into a memory-bounded buffer. When the buffer
/// exceeds `SFRAME_SORT_BUFFER_SIZE`, sorts and spills to CacheFs.
/// After all input, either returns sorted data directly (fast path)
/// or k-way merges the sorted runs (external sort path).
pub fn sort(input: BatchIterator, keys: &[SortKey]) -> Result<BatchIterator> {
    let budget = sframe_config::global().sort_memory_budget;
    sort_with_budget(input, keys, budget)
}

/// Sort with an explicit memory budget (for testing).
fn sort_with_budget(
    mut input: BatchIterator,
    keys: &[SortKey],
    budget: usize,
) -> Result<BatchIterator> {
    if keys.is_empty() {
        return Ok(input);
    }

    let cache_fs = global_cache_fs();
    let vfs: Arc<dyn VirtualFileSystem> =
        Arc::new(ArcCacheFsVfs(cache_fs.clone()));
    let base_path = cache_fs.alloc_dir();
    VirtualFileSystem::mkdir_p(&*vfs, &base_path)?;

    let mut buffer: Option<SFrameRows> = None;
    let mut buffer_size: usize = 0;
    let mut runs: Vec<SortedRunInfo> = Vec::new();
    let mut dtypes: Option<Vec<FlexTypeEnum>> = None;
    let mut run_id: usize = 0;

    // Phase 1: Run generation
    while let Some(batch_result) = input.next_batch() {
        let batch = batch_result?;
        if batch.num_rows() == 0 {
            continue;
        }

        if dtypes.is_none() {
            dtypes = Some(
                (0..batch.num_columns())
                    .map(|i| batch.column(i).dtype())
                    .collect(),
            );
        }

        buffer_size += estimate_batch_size(&batch);
        match &mut buffer {
            None => buffer = Some(batch),
            Some(existing) => existing.append(&batch)?,
        }

        if buffer_size >= budget {
            let buf = buffer.take().unwrap();
            let info = spill_sorted_run(
                &buf,
                keys,
                dtypes.as_ref().unwrap(),
                &*vfs,
                &base_path,
                run_id,
            )?;
            runs.push(info);
            run_id += 1;
            buffer_size = 0;
        }
    }

    let dtypes = match dtypes {
        Some(d) => d,
        None => {
            // Empty input
            let _ = cache_fs.remove_dir(&base_path);
            return Ok(BatchIterator::new(|co: BatchCo| async move {
                co.yield_(BatchResponse::Ready).await;
            }));
        }
    };

    // Phase 2: Fast path — everything fit in memory
    if runs.is_empty() {
        let _ = cache_fs.remove_dir(&base_path);
        let buf = buffer.unwrap();
        if buf.num_rows() <= 1 {
            return Ok(single_batch_iter(buf));
        }
        let indices = build_sort_indices(&buf, keys);
        let batch_size = sframe_config::global().source_batch_size;
        return Ok(BatchIterator::new(move |co: BatchCo| async move {
            let mut cmd = co.yield_(BatchResponse::Ready).await;
            for chunk in indices.chunks(batch_size) {
                if !matches!(cmd, BatchCommand::NextBatch) {
                    break;
                }
                match buf.take(chunk) {
                    Ok(batch) => {
                        cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await;
                    }
                    Err(e) => {
                        co.yield_(BatchResponse::Batch(Err(e))).await;
                        return;
                    }
                }
            }
        }));
    }

    // Phase 3: Spill remaining buffer as final run
    if let Some(buf) = buffer.take() {
        if buf.num_rows() > 0 {
            let info = spill_sorted_run(&buf, keys, &dtypes, &*vfs, &base_path, run_id)?;
            runs.push(info);
        }
    }

    // Phase 4: k-way merge
    merge_sorted_runs(runs, keys, &dtypes, vfs, base_path)
}

fn single_batch_iter(batch: SFrameRows) -> BatchIterator {
    BatchIterator::new(move |co: BatchCo| async move {
        let cmd = co.yield_(BatchResponse::Ready).await;
        if matches!(cmd, BatchCommand::NextBatch) {
            co.yield_(BatchResponse::Batch(Ok(batch))).await;
        }
    })
}

/// Materializes the input stream and returns the original batch together
/// with the sorted index permutation, *without* building a sorted copy.
///
/// Callers can use the indices to write data in sorted order in chunks
/// (e.g. via `CacheSFrameBuilder::write_indexed_chunked`) to avoid
/// holding a full sorted copy in memory.
pub fn sort_indices(
    mut input: BatchIterator,
    keys: &[SortKey],
) -> Result<(SFrameRows, Vec<usize>)> {
    // Materialize all batches
    let mut result: Option<SFrameRows> = None;
    while let Some(batch_result) = input.next_batch() {
        let batch = batch_result?;
        match &mut result {
            None => result = Some(batch),
            Some(existing) => existing.append(&batch)?,
        }
    }

    let batch = match result {
        Some(b) => b,
        None => {
            let empty = SFrameRows::empty(&[]);
            return Ok((empty, Vec::new()));
        }
    };

    if batch.num_rows() <= 1 || keys.is_empty() {
        let n = batch.num_rows();
        let indices: Vec<usize> = (0..n).collect();
        return Ok((batch, indices));
    }

    let indices = build_sort_indices(&batch, keys);
    Ok((batch, indices))
}

/// Build a sorted index permutation for the batch by the given sort keys.
///
/// Uses rayon's parallel sort for datasets larger than 10K rows.
pub fn build_sort_indices(batch: &SFrameRows, keys: &[SortKey]) -> Vec<usize> {
    let n = batch.num_rows();
    let mut indices: Vec<usize> = (0..n).collect();

    let cmp = |&a: &usize, &b: &usize| -> std::cmp::Ordering {
        for key in keys {
            let va = batch.column(key.column).get(a);
            let vb = batch.column(key.column).get(b);
            let cmp = compare_flex_type(&va, &vb);
            let cmp = match key.order {
                SortOrder::Ascending => cmp,
                SortOrder::Descending => cmp.reverse(),
            };
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    };

    if n > 10_000 {
        indices.par_sort_by(cmp);
    } else {
        indices.sort_by(cmp);
    }

    indices
}

/// Estimate the memory size of a batch in bytes (rough).
pub fn estimate_batch_size(batch: &SFrameRows) -> usize {
    let n = batch.num_rows();
    if n == 0 {
        return 0;
    }

    let mut size = 0usize;
    for col_idx in 0..batch.num_columns() {
        let col = batch.column(col_idx);
        // Rough estimate per element based on column type
        let per_elem = match col.dtype() {
            sframe_types::flex_type::FlexTypeEnum::Integer => 9,  // Option<i64>
            sframe_types::flex_type::FlexTypeEnum::Float => 9,    // Option<f64>
            sframe_types::flex_type::FlexTypeEnum::String => 32,  // Option<Arc<str>>
            sframe_types::flex_type::FlexTypeEnum::Vector => 64,  // Option<Arc<[f64]>>
            sframe_types::flex_type::FlexTypeEnum::List => 64,
            sframe_types::flex_type::FlexTypeEnum::Dict => 64,
            sframe_types::flex_type::FlexTypeEnum::DateTime => 16,
            sframe_types::flex_type::FlexTypeEnum::Undefined => 1,
        };
        size += n * per_elem;
    }
    size
}

/// Compare two FlexType values for ordering.
/// Undefined sorts last. Cross-type comparison: Integer < Float < String < Vector < rest.
pub fn compare_flex_type(a: &FlexType, b: &FlexType) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a, b) {
        (FlexType::Undefined, FlexType::Undefined) => Ordering::Equal,
        (FlexType::Undefined, _) => Ordering::Greater,
        (_, FlexType::Undefined) => Ordering::Less,

        (FlexType::Integer(x), FlexType::Integer(y)) => x.cmp(y),
        (FlexType::Float(x), FlexType::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (FlexType::String(x), FlexType::String(y)) => x.as_ref().cmp(y.as_ref()),

        // Cross-type numeric comparison
        (FlexType::Integer(x), FlexType::Float(y)) => {
            (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal)
        }
        (FlexType::Float(x), FlexType::Integer(y)) => {
            x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal)
        }

        // Type ordering fallback
        (a, b) => type_rank(a).cmp(&type_rank(b)),
    }
}

fn type_rank(v: &FlexType) -> u8 {
    match v {
        FlexType::Integer(_) => 0,
        FlexType::Float(_) => 1,
        FlexType::String(_) => 2,
        FlexType::Vector(_) => 3,
        FlexType::List(_) => 4,
        FlexType::Dict(_) => 5,
        FlexType::DateTime(_) => 6,
        FlexType::Undefined => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::SFrameRows;
    use crate::execute::{BatchCo, BatchCommand, BatchIterator, BatchResponse};
    use sframe_types::flex_type::FlexTypeEnum;

    fn make_sort_input(batch: SFrameRows) -> BatchIterator {
        BatchIterator::new(move |co: BatchCo| async move {
            let cmd = co.yield_(BatchResponse::Ready).await;
            if matches!(cmd, BatchCommand::NextBatch) {
                co.yield_(BatchResponse::Batch(Ok(batch))).await;
            }
        })
    }

    /// Collect all batches from a BatchIterator into a single SFrameRows.
    fn collect_batches(iter: &mut BatchIterator) -> SFrameRows {
        let mut result: Option<SFrameRows> = None;
        while let Some(batch_result) = iter.next_batch() {
            let batch = batch_result.unwrap();
            match &mut result {
                None => result = Some(batch),
                Some(existing) => existing.append(&batch).unwrap(),
            }
        }
        result.unwrap()
    }

    #[test]
    fn test_sort_integers() {
        let rows = vec![
            vec![FlexType::Integer(3)],
            vec![FlexType::Integer(1)],
            vec![FlexType::Integer(4)],
            vec![FlexType::Integer(1)],
            vec![FlexType::Integer(5)],
        ];
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input = make_sort_input(batch);

        let mut iter = sort(input, &[SortKey::asc(0)]).unwrap();
        let result = collect_batches(&mut iter);

        let expected = [1, 1, 3, 4, 5];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(result.row(i), vec![FlexType::Integer(exp)]);
        }
    }

    #[test]
    fn test_sort_descending() {
        let rows = vec![
            vec![FlexType::Float(1.5)],
            vec![FlexType::Float(3.5)],
            vec![FlexType::Float(2.5)],
        ];
        let dtypes = [FlexTypeEnum::Float];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input = make_sort_input(batch);

        let mut iter = sort(input, &[SortKey::desc(0)]).unwrap();
        let result = collect_batches(&mut iter);

        let expected = [3.5, 2.5, 1.5];
        for (i, &exp) in expected.iter().enumerate() {
            match &result.row(i)[0] {
                FlexType::Float(v) => assert!((v - exp).abs() < 1e-10),
                other => panic!("Expected Float, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_sort_strings() {
        let rows = vec![
            vec![FlexType::String("cherry".into())],
            vec![FlexType::String("apple".into())],
            vec![FlexType::String("banana".into())],
        ];
        let dtypes = [FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input = make_sort_input(batch);

        let mut iter = sort(input, &[SortKey::asc(0)]).unwrap();
        let result = collect_batches(&mut iter);

        assert_eq!(result.row(0), vec![FlexType::String("apple".into())]);
        assert_eq!(result.row(1), vec![FlexType::String("banana".into())]);
        assert_eq!(result.row(2), vec![FlexType::String("cherry".into())]);
    }

    #[test]
    fn test_sort_multi_key() {
        let rows = vec![
            vec![FlexType::Integer(2), FlexType::String("b".into())],
            vec![FlexType::Integer(1), FlexType::String("b".into())],
            vec![FlexType::Integer(2), FlexType::String("a".into())],
            vec![FlexType::Integer(1), FlexType::String("a".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input = make_sort_input(batch);

        let mut iter = sort(
            input,
            &[SortKey::asc(0), SortKey::asc(1)],
        )
        .unwrap();
        let result = collect_batches(&mut iter);

        assert_eq!(
            result.row(0),
            vec![FlexType::Integer(1), FlexType::String("a".into())]
        );
        assert_eq!(
            result.row(1),
            vec![FlexType::Integer(1), FlexType::String("b".into())]
        );
        assert_eq!(
            result.row(2),
            vec![FlexType::Integer(2), FlexType::String("a".into())]
        );
        assert_eq!(
            result.row(3),
            vec![FlexType::Integer(2), FlexType::String("b".into())]
        );
    }

    #[test]
    fn test_sort_with_undefined() {
        let rows = vec![
            vec![FlexType::Integer(3)],
            vec![FlexType::Undefined],
            vec![FlexType::Integer(1)],
            vec![FlexType::Undefined],
        ];
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input = make_sort_input(batch);

        let mut iter = sort(input, &[SortKey::asc(0)]).unwrap();
        let result = collect_batches(&mut iter);

        // Undefined sorts last
        assert_eq!(result.row(0), vec![FlexType::Integer(1)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(3)]);
        assert_eq!(result.row(2), vec![FlexType::Undefined]);
        assert_eq!(result.row(3), vec![FlexType::Undefined]);
    }

    #[test]
    fn test_estimate_batch_size() {
        let rows = vec![
            vec![FlexType::Integer(1), FlexType::Float(1.0)],
            vec![FlexType::Integer(2), FlexType::Float(2.0)],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::Float];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let size = estimate_batch_size(&batch);
        // 2 rows × (9 bytes for int + 9 bytes for float) = 36
        assert_eq!(size, 36);
    }

    #[test]
    fn test_spill_sorted_run() {
        use sframe_io::cache_fs::global_cache_fs;
        use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
        use sframe_storage::segment_reader::SegmentReader;

        let rows = vec![
            vec![FlexType::Integer(3), FlexType::String("c".into())],
            vec![FlexType::Integer(1), FlexType::String("a".into())],
            vec![FlexType::Integer(2), FlexType::String("b".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let cache_fs = global_cache_fs();
        let vfs: std::sync::Arc<dyn VirtualFileSystem> =
            std::sync::Arc::new(ArcCacheFsVfs(cache_fs.clone()));
        let base_path = cache_fs.alloc_dir();
        VirtualFileSystem::mkdir_p(&*vfs, &base_path).unwrap();

        let keys = [SortKey::asc(0)];
        let info = spill_sorted_run(&batch, &keys, &dtypes, &*vfs, &base_path, 0).unwrap();
        assert_eq!(info.num_rows, 3);

        // Read back via SegmentReader and verify sorted order
        let file = vfs.open_read(&info.path).unwrap();
        let file_size = file.size().unwrap();
        let mut reader = SegmentReader::open(Box::new(file), file_size, dtypes.to_vec()).unwrap();
        let col0 = reader.read_column(0).unwrap();
        // Data should be sorted by column 0: 1, 2, 3
        assert_eq!(col0, vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)]);
        let col1 = reader.read_column(1).unwrap();
        // Column 1 should follow the sort order: a, b, c
        assert_eq!(col1, vec![FlexType::String("a".into()), FlexType::String("b".into()), FlexType::String("c".into())]);

        let _ = cache_fs.remove_dir(&base_path);
    }

    #[test]
    fn test_run_cursor() {
        use sframe_io::cache_fs::global_cache_fs;
        use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};

        let rows: Vec<Vec<FlexType>> = (0..10)
            .map(|i| vec![FlexType::Integer(i)])
            .collect();
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let cache_fs = global_cache_fs();
        let vfs: std::sync::Arc<dyn VirtualFileSystem> =
            std::sync::Arc::new(ArcCacheFsVfs(cache_fs.clone()));
        let base_path = cache_fs.alloc_dir();
        VirtualFileSystem::mkdir_p(&*vfs, &base_path).unwrap();

        let info = spill_sorted_run(&batch, &[SortKey::asc(0)], &dtypes, &*vfs, &base_path, 0).unwrap();

        // Create RunCursor with small read-ahead (3 rows)
        let mut cursor = RunCursor::open(&*vfs, &info.path, &dtypes, info.num_rows, 3).unwrap();

        let mut values = Vec::new();
        while !cursor.is_exhausted() {
            values.push(cursor.current_value(0).clone());
            cursor.advance().unwrap();
        }
        assert_eq!(values.len(), 10);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, FlexType::Integer(i as i64));
        }

        let _ = cache_fs.remove_dir(&base_path);
    }

    #[test]
    fn test_merge_heap_entry_ordering() {
        use std::collections::BinaryHeap;

        let descending = vec![false]; // ascending
        let entries = vec![
            MergeHeapEntry {
                run_idx: 0,
                key_values: vec![FlexType::Integer(3)],
                descending: descending.clone(),
            },
            MergeHeapEntry {
                run_idx: 1,
                key_values: vec![FlexType::Integer(1)],
                descending: descending.clone(),
            },
            MergeHeapEntry {
                run_idx: 2,
                key_values: vec![FlexType::Integer(2)],
                descending: descending.clone(),
            },
        ];

        let mut heap = BinaryHeap::new();
        for e in entries {
            heap.push(e);
        }

        // Min-heap: should pop in ascending order 1, 2, 3
        assert_eq!(heap.pop().unwrap().key_values[0], FlexType::Integer(1));
        assert_eq!(heap.pop().unwrap().key_values[0], FlexType::Integer(2));
        assert_eq!(heap.pop().unwrap().key_values[0], FlexType::Integer(3));
    }

    #[test]
    fn test_merge_sorted_runs() {
        use sframe_io::cache_fs::global_cache_fs;
        use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};

        let dtypes = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];
        let cache_fs = global_cache_fs();
        let vfs: std::sync::Arc<dyn VirtualFileSystem> =
            std::sync::Arc::new(ArcCacheFsVfs(cache_fs.clone()));
        let base_path = cache_fs.alloc_dir();
        VirtualFileSystem::mkdir_p(&*vfs, &base_path).unwrap();

        // Run 0: [1, 4, 7]
        let batch0 = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(1), FlexType::String("a".into())],
                vec![FlexType::Integer(4), FlexType::String("d".into())],
                vec![FlexType::Integer(7), FlexType::String("g".into())],
            ],
            &dtypes,
        )
        .unwrap();
        let run0 =
            spill_sorted_run(&batch0, &[SortKey::asc(0)], &dtypes, &*vfs, &base_path, 0).unwrap();

        // Run 1: [2, 5, 8]
        let batch1 = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(2), FlexType::String("b".into())],
                vec![FlexType::Integer(5), FlexType::String("e".into())],
                vec![FlexType::Integer(8), FlexType::String("h".into())],
            ],
            &dtypes,
        )
        .unwrap();
        let run1 =
            spill_sorted_run(&batch1, &[SortKey::asc(0)], &dtypes, &*vfs, &base_path, 1).unwrap();

        // Run 2: [3, 6, 9]
        let batch2 = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(3), FlexType::String("c".into())],
                vec![FlexType::Integer(6), FlexType::String("f".into())],
                vec![FlexType::Integer(9), FlexType::String("i".into())],
            ],
            &dtypes,
        )
        .unwrap();
        let run2 =
            spill_sorted_run(&batch2, &[SortKey::asc(0)], &dtypes, &*vfs, &base_path, 2).unwrap();

        let runs = vec![run0, run1, run2];
        let keys = vec![SortKey::asc(0)];
        let mut iter = merge_sorted_runs(runs, &keys, &dtypes, vfs.clone(), base_path).unwrap();

        let mut all_ints: Vec<i64> = Vec::new();
        let mut all_strs: Vec<String> = Vec::new();
        while let Some(batch_result) = iter.next_batch() {
            let batch = batch_result.unwrap();
            for i in 0..batch.num_rows() {
                if let FlexType::Integer(v) = batch.column(0).get(i) {
                    all_ints.push(v);
                }
                if let FlexType::String(v) = batch.column(1).get(i) {
                    all_strs.push(v.to_string());
                }
            }
        }
        assert_eq!(all_ints, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            all_strs,
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        );
    }

    #[test]
    fn test_sort_with_tiny_budget() {
        let n = 1000i64;
        let rows: Vec<Vec<FlexType>> = (0..n)
            .rev()
            .map(|i| vec![FlexType::Integer(i)])
            .collect();
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input = make_sort_input(batch);

        // Budget of 1 byte forces every batch to spill
        let mut result = sort_with_budget(input, &[SortKey::asc(0)], 1).unwrap();

        let mut values = Vec::new();
        while let Some(batch_result) = result.next_batch() {
            let batch = batch_result.unwrap();
            for i in 0..batch.num_rows() {
                if let FlexType::Integer(v) = batch.column(0).get(i) {
                    values.push(v);
                }
            }
        }

        let expected: Vec<i64> = (0..n).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn test_sort_descending_external() {
        let n = 500i64;
        let rows: Vec<Vec<FlexType>> = (0..n)
            .map(|i| vec![FlexType::Integer(i)])
            .collect();
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input = make_sort_input(batch);

        let mut result = sort_with_budget(input, &[SortKey::desc(0)], 1).unwrap();
        let collected = collect_batches(&mut result);

        for i in 0..n as usize {
            assert_eq!(collected.row(i), vec![FlexType::Integer(n - 1 - i as i64)]);
        }
    }

    #[test]
    fn test_sort_multi_key_external() {
        let rows = vec![
            vec![FlexType::Integer(2), FlexType::String("b".into())],
            vec![FlexType::Integer(1), FlexType::String("b".into())],
            vec![FlexType::Integer(2), FlexType::String("a".into())],
            vec![FlexType::Integer(1), FlexType::String("a".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input = make_sort_input(batch);

        let mut result = sort_with_budget(
            input,
            &[SortKey::asc(0), SortKey::asc(1)],
            1,
        ).unwrap();
        let collected = collect_batches(&mut result);

        assert_eq!(collected.row(0), vec![FlexType::Integer(1), FlexType::String("a".into())]);
        assert_eq!(collected.row(1), vec![FlexType::Integer(1), FlexType::String("b".into())]);
        assert_eq!(collected.row(2), vec![FlexType::Integer(2), FlexType::String("a".into())]);
        assert_eq!(collected.row(3), vec![FlexType::Integer(2), FlexType::String("b".into())]);
    }

    #[test]
    fn test_sort_empty_external() {
        let input = BatchIterator::new(|co: BatchCo| async move {
            co.yield_(BatchResponse::Ready).await;
        });
        let mut result = sort_with_budget(input, &[SortKey::asc(0)], 1).unwrap();
        assert!(result.next_batch().is_none());
    }
}
