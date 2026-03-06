//! Columnar sort with EC-Sort optimization.
//!
//! Uses index-based sorting: only key column values are accessed during
//! comparisons, and the permutation is applied to all columns in a single
//! pass via `take()`. This is the in-memory equivalent of the C++ EC-Sort
//! algorithm which avoids shuffling large value columns.
//!
//! For data that fits in the sort memory budget, the entire dataset is
//! materialized and sorted in-memory. The EC-Sort path (disk-based) can
//! be added later for truly out-of-core datasets.

use rayon::prelude::*;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::vfs::VirtualFileSystem;
use sframe_storage::segment_writer::SegmentWriter;
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::SFrameRows;
use crate::execute::BatchIterator;

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
    let seg_path = format!("{}/run_{:04}", base_path, run_id);
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

/// Sort a batch stream by the given keys.
///
/// Materializes the input stream, then sorts using an index-based
/// permutation (EC-Sort pattern). Only key columns are accessed during
/// comparison; the permutation is applied to all columns in one pass.
pub fn sort(input: BatchIterator, keys: &[SortKey]) -> Result<SFrameRows> {
    let (batch, indices) = sort_indices(input, keys)?;
    batch.take(&indices)
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

        let result = sort(input, &[SortKey::asc(0)]).unwrap();

        let expected = vec![1, 1, 3, 4, 5];
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

        let result = sort(input, &[SortKey::desc(0)]).unwrap();

        let expected = vec![3.5, 2.5, 1.5];
        for (i, &exp) in expected.iter().enumerate() {
            match &result.row(i)[0] {
                FlexType::Float(v) => assert!((v - exp).abs() < 1e-10),
                other => panic!("Expected Float, got {:?}", other),
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

        let result = sort(input, &[SortKey::asc(0)]).unwrap();

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

        let result = sort(
            input,
            &[SortKey::asc(0), SortKey::asc(1)],
        )
        .unwrap();

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

        let result = sort(input, &[SortKey::asc(0)]).unwrap();

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
}
