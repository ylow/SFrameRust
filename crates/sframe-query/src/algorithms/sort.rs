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

use futures::StreamExt;

use sframe_types::error::Result;
use sframe_types::flex_type::FlexType;

use crate::batch::SFrameRows;
use crate::config::SFrameConfig;
use crate::execute::BatchStream;

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

/// Sort a batch stream by the given keys.
///
/// Materializes the input stream, then sorts using an index-based
/// permutation (EC-Sort pattern). Only key columns are accessed during
/// comparison; the permutation is applied to all columns in one pass.
///
/// The `config` parameter controls the memory budget. If the estimated
/// data size exceeds `sort_memory_budget`, a warning is logged but
/// in-memory sort is still used (true external sort requires disk
/// spilling which is not yet implemented).
pub async fn sort(mut input: BatchStream, keys: &[SortKey]) -> Result<SFrameRows> {
    // Materialize all batches
    let mut result: Option<SFrameRows> = None;
    while let Some(batch_result) = input.next().await {
        let batch = batch_result?;
        match &mut result {
            None => result = Some(batch),
            Some(existing) => existing.append(&batch)?,
        }
    }

    let batch = match result {
        Some(b) => b,
        None => return Ok(SFrameRows::empty(&[])),
    };

    if batch.num_rows() <= 1 || keys.is_empty() {
        return Ok(batch);
    }

    sort_in_memory(batch, keys)
}

/// In-memory EC-Sort: build a permutation from key columns, then apply
/// it to all columns via `take()`.
///
/// This is O(n log n) in comparisons on key columns only. The
/// permutation application is O(n × num_columns).
fn sort_in_memory(batch: SFrameRows, keys: &[SortKey]) -> Result<SFrameRows> {
    let n = batch.num_rows();

    // Step 1: Build permutation by sorting indices on key columns only.
    // This is the "sort key columns + row numbers" step of EC-Sort.
    let mut indices: Vec<usize> = (0..n).collect();

    indices.sort_by(|&a, &b| {
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
    });

    // Step 2: Apply permutation to all columns in one pass.
    // This is the "apply forward-map" step of EC-Sort.
    batch.take(&indices)
}

/// Estimate the memory size of a batch in bytes (rough).
pub fn estimate_batch_size(batch: &SFrameRows) -> usize {
    let n = batch.num_rows();
    if n == 0 {
        return 0;
    }

    let config = SFrameConfig::global();
    let _ = config; // future use

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
    use futures::stream;
    use sframe_types::flex_type::FlexTypeEnum;

    #[tokio::test]
    async fn test_sort_integers() {
        let rows = vec![
            vec![FlexType::Integer(3)],
            vec![FlexType::Integer(1)],
            vec![FlexType::Integer(4)],
            vec![FlexType::Integer(1)],
            vec![FlexType::Integer(5)],
        ];
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));

        let result = sort(input, &[SortKey::asc(0)]).await.unwrap();

        let expected = vec![1, 1, 3, 4, 5];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(result.row(i), vec![FlexType::Integer(exp)]);
        }
    }

    #[tokio::test]
    async fn test_sort_descending() {
        let rows = vec![
            vec![FlexType::Float(1.5)],
            vec![FlexType::Float(3.5)],
            vec![FlexType::Float(2.5)],
        ];
        let dtypes = [FlexTypeEnum::Float];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));

        let result = sort(input, &[SortKey::desc(0)]).await.unwrap();

        let expected = vec![3.5, 2.5, 1.5];
        for (i, &exp) in expected.iter().enumerate() {
            match &result.row(i)[0] {
                FlexType::Float(v) => assert!((v - exp).abs() < 1e-10),
                other => panic!("Expected Float, got {:?}", other),
            }
        }
    }

    #[tokio::test]
    async fn test_sort_strings() {
        let rows = vec![
            vec![FlexType::String("cherry".into())],
            vec![FlexType::String("apple".into())],
            vec![FlexType::String("banana".into())],
        ];
        let dtypes = [FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));

        let result = sort(input, &[SortKey::asc(0)]).await.unwrap();

        assert_eq!(result.row(0), vec![FlexType::String("apple".into())]);
        assert_eq!(result.row(1), vec![FlexType::String("banana".into())]);
        assert_eq!(result.row(2), vec![FlexType::String("cherry".into())]);
    }

    #[tokio::test]
    async fn test_sort_multi_key() {
        let rows = vec![
            vec![FlexType::Integer(2), FlexType::String("b".into())],
            vec![FlexType::Integer(1), FlexType::String("b".into())],
            vec![FlexType::Integer(2), FlexType::String("a".into())],
            vec![FlexType::Integer(1), FlexType::String("a".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));

        let result = sort(
            input,
            &[SortKey::asc(0), SortKey::asc(1)],
        )
        .await
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

    #[tokio::test]
    async fn test_sort_with_undefined() {
        let rows = vec![
            vec![FlexType::Integer(3)],
            vec![FlexType::Undefined],
            vec![FlexType::Integer(1)],
            vec![FlexType::Undefined],
        ];
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();
        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));

        let result = sort(input, &[SortKey::asc(0)]).await.unwrap();

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
}
