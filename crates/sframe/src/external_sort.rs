//! External sort for datasets that exceed the in-memory sort budget.
//!
//! Uses a 5-phase algorithm:
//! 1. Build a quantile sketch of the primary sort key.
//! 2. Determine partition cut points from the sketch.
//! 3. Partition data into P temporary SFrames.
//! 4. Sort each partition in-memory.
//! 5. Write sorted partitions sequentially into a single output SFrame.

use futures::StreamExt;

use sframe_query::algorithms::quantile_sketch::QuantileSketch;
use sframe_query::algorithms::sort::{compare_flex_type, SortKey, SortOrder};
use sframe_query::config::SFrameConfig;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexType;

use crate::sframe::{SFrame, SFrameBuilder};

const CHUNK_SIZE: usize = 8192;

/// Perform an external sort on `sf` using the given sort keys.
///
/// The data is partitioned by the primary sort key into chunks that fit
/// within the sort memory budget. Each partition is sorted in-memory,
/// then the sorted partitions are written sequentially into a new SFrame.
pub(crate) fn external_sort(sf: &SFrame, sort_keys: &[SortKey]) -> Result<SFrame> {
    if sort_keys.is_empty() {
        return Err(SFrameError::Format(
            "external_sort requires at least one sort key".to_string(),
        ));
    }

    let primary_key = &sort_keys[0];
    let budget = SFrameConfig::global().sort_memory_budget;
    let estimated_size = sf.estimate_size();

    // Phase 1: Build quantile sketch of the primary sort key
    let sketch = build_sketch(sf, primary_key.column)?;

    if sketch.count() == 0 {
        // Empty input — return empty SFrame with same schema
        return sf.head(0);
    }

    // Phase 2: Determine number of partitions and cut points
    let num_partitions = (estimated_size / budget).max(2);
    let cut_points = sketch.quantiles(num_partitions);

    // Phase 3: Partition data into P temporary SFrames
    let partitions = partition_data(sf, primary_key, &cut_points)?;

    // Phase 4+5: Sort each partition in-memory and write to output
    // Iterate partitions in the correct global order:
    // ascending primary key → partitions 0..P (low to high)
    // descending primary key → partitions P-1..0 (high to low)
    let partition_order: Vec<usize> = if primary_key.order == SortOrder::Descending {
        (0..partitions.len()).rev().collect()
    } else {
        (0..partitions.len()).collect()
    };

    let mut builder =
        SFrameBuilder::anonymous(sf.column_names().to_vec(), sf.column_types())?;

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| SFrameError::Format(format!("Runtime error: {}", e)))?;

    for &part_idx in &partition_order {
        let part = &partitions[part_idx];
        let num_rows = part.num_rows().unwrap_or(0);
        if num_rows == 0 {
            continue;
        }

        // Sort this partition in-memory
        let sorted = part.sort_in_memory(sort_keys)?;

        // Stream sorted data into the output builder
        let stream = sorted.compile_stream()?;
        rt.block_on(async {
            let mut stream = stream;
            while let Some(batch_result) = stream.next().await {
                let batch = batch_result?;
                builder.write_batch_chunked(&batch, CHUNK_SIZE)?;
            }
            Ok::<(), sframe_types::error::SFrameError>(())
        })?;
        // `sorted` and `part` dropped here — memory freed before next partition
    }

    // Drop partitions explicitly before finishing
    drop(partitions);

    builder.finish()
}

/// Phase 1: Stream the data and build a quantile sketch of the primary sort key.
fn build_sketch(sf: &SFrame, key_column: usize) -> Result<QuantileSketch> {
    let mut sketch = QuantileSketch::new(0.01);

    let stream = sf.compile_stream()?;
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| SFrameError::Format(format!("Runtime error: {}", e)))?;

    rt.block_on(async {
        let mut stream = stream;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let col = batch.column(key_column);
            for i in 0..batch.num_rows() {
                sketch.insert(col.get(i));
            }
        }
        Ok::<(), sframe_types::error::SFrameError>(())
    })?;

    sketch.finish();
    Ok(sketch)
}

/// Phase 3: Partition data by the primary sort key using the cut points.
///
/// Produces `cut_points.len() + 1` partitions. For ascending sort:
/// - Partition 0: values <= cut_points[0]
/// - Partition i: cut_points[i-1] < values <= cut_points[i]
/// - Partition P-1: values > cut_points[P-2]
///
/// Each partition is written to an anonymous SFrame in the cache.
fn partition_data(
    sf: &SFrame,
    primary_key: &SortKey,
    cut_points: &[FlexType],
) -> Result<Vec<SFrame>> {
    let num_partitions = cut_points.len() + 1;
    let column_names = sf.column_names().to_vec();
    let dtypes = sf.column_types();

    // Create one builder per partition
    let mut builders: Vec<SFrameBuilder> = Vec::with_capacity(num_partitions);
    for _ in 0..num_partitions {
        builders.push(SFrameBuilder::anonymous(
            column_names.clone(),
            dtypes.clone(),
        )?);
    }

    // Stream input and scatter rows into partitions
    let stream = sf.compile_stream()?;
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| SFrameError::Format(format!("Runtime error: {}", e)))?;

    rt.block_on(async {
        let mut stream = stream;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            scatter_batch(&batch, primary_key.column, cut_points, &mut builders)?;
        }
        Ok::<(), sframe_types::error::SFrameError>(())
    })?;

    // Finish all builders
    let mut partitions = Vec::with_capacity(num_partitions);
    for builder in builders {
        partitions.push(builder.finish()?);
    }

    Ok(partitions)
}

/// Scatter rows from a batch into partition builders based on the primary key value.
fn scatter_batch(
    batch: &sframe_query::batch::SFrameRows,
    key_column: usize,
    cut_points: &[FlexType],
    builders: &mut [SFrameBuilder],
) -> Result<()> {
    let nrows = batch.num_rows();
    let ncols = batch.num_columns();

    // Build per-partition row buffers (column-major)
    let num_partitions = builders.len();
    let mut partition_bufs: Vec<Vec<Vec<FlexType>>> = (0..num_partitions)
        .map(|_| (0..ncols).map(|_| Vec::new()).collect())
        .collect();

    // Classify each row
    for row_idx in 0..nrows {
        let key_val = batch.column(key_column).get(row_idx);
        let part_idx = find_partition(&key_val, cut_points);

        for col_idx in 0..ncols {
            partition_bufs[part_idx][col_idx].push(batch.column(col_idx).get(row_idx));
        }
    }

    // Write each non-empty buffer to its builder
    for (part_idx, bufs) in partition_bufs.iter().enumerate() {
        if bufs[0].is_empty() {
            continue;
        }
        builders[part_idx].write_columns(bufs)?;
    }

    Ok(())
}

/// Binary search to find which partition a value belongs to.
///
/// Returns partition index in 0..cut_points.len()+1.
/// - Values <= cut_points[0] go to partition 0
/// - Values <= cut_points[i] go to partition i
/// - Values > all cut points go to the last partition
fn find_partition(value: &FlexType, cut_points: &[FlexType]) -> usize {
    // Binary search: find the first cut point >= value
    let mut lo = 0;
    let mut hi = cut_points.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if compare_flex_type(value, &cut_points[mid]) == std::cmp::Ordering::Greater {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sarray::SArray;
    use sframe_types::flex_type::FlexTypeEnum;

    #[test]
    fn test_find_partition() {
        let cuts = vec![
            FlexType::Integer(10),
            FlexType::Integer(20),
            FlexType::Integer(30),
        ];
        // 4 partitions: [..10], (10..20], (20..30], (30..]
        assert_eq!(find_partition(&FlexType::Integer(5), &cuts), 0);
        assert_eq!(find_partition(&FlexType::Integer(10), &cuts), 0);
        assert_eq!(find_partition(&FlexType::Integer(15), &cuts), 1);
        assert_eq!(find_partition(&FlexType::Integer(20), &cuts), 1);
        assert_eq!(find_partition(&FlexType::Integer(25), &cuts), 2);
        assert_eq!(find_partition(&FlexType::Integer(35), &cuts), 3);
    }

    #[test]
    fn test_external_sort_forced() {
        // Create an SFrame with enough data, then force external sort by
        // using a tiny budget.
        let n = 500;
        let values: Vec<FlexType> = (0..n)
            .map(|i| FlexType::Integer(n - 1 - i)) // descending: 499, 498, ..., 0
            .collect();

        let sf = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(values, FlexTypeEnum::Integer).unwrap(),
        )])
        .unwrap();

        // Test external sort directly (bypassing the budget check)
        let sort_keys = vec![SortKey::asc(0)];
        let sorted = external_sort(&sf, &sort_keys).unwrap();

        let rows = sorted.iter_rows().unwrap();
        assert_eq!(rows.len(), n as usize);

        // Verify ascending order
        for i in 0..rows.len() {
            assert_eq!(rows[i], vec![FlexType::Integer(i as i64)]);
        }
    }

    #[test]
    fn test_external_sort_descending() {
        let n = 500;
        let sf = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(
                (0..n).map(|i| FlexType::Integer(i)).collect(),
                FlexTypeEnum::Integer,
            )
            .unwrap(),
        )])
        .unwrap();

        let sort_keys = vec![SortKey::desc(0)];
        let sorted = external_sort(&sf, &sort_keys).unwrap();

        let rows = sorted.iter_rows().unwrap();
        assert_eq!(rows.len(), n as usize);

        // Verify descending order
        for i in 0..rows.len() {
            assert_eq!(rows[i], vec![FlexType::Integer(n - 1 - i as i64)]);
        }
    }

    #[test]
    fn test_external_sort_multi_key() {
        let sf = SFrame::from_columns(vec![
            (
                "a",
                SArray::from_vec(
                    vec![
                        FlexType::Integer(2),
                        FlexType::Integer(1),
                        FlexType::Integer(2),
                        FlexType::Integer(1),
                    ],
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
            (
                "b",
                SArray::from_vec(
                    vec![
                        FlexType::Integer(20),
                        FlexType::Integer(10),
                        FlexType::Integer(10),
                        FlexType::Integer(20),
                    ],
                    FlexTypeEnum::Integer,
                )
                .unwrap(),
            ),
        ])
        .unwrap();

        let sort_keys = vec![SortKey::asc(0), SortKey::desc(1)];
        let sorted = external_sort(&sf, &sort_keys).unwrap();

        let rows = sorted.iter_rows().unwrap();
        assert_eq!(rows.len(), 4);
        // (1,20), (1,10), (2,20), (2,10) — asc on a, desc on b
        assert_eq!(
            rows[0],
            vec![FlexType::Integer(1), FlexType::Integer(20)]
        );
        assert_eq!(
            rows[1],
            vec![FlexType::Integer(1), FlexType::Integer(10)]
        );
        assert_eq!(
            rows[2],
            vec![FlexType::Integer(2), FlexType::Integer(20)]
        );
        assert_eq!(
            rows[3],
            vec![FlexType::Integer(2), FlexType::Integer(10)]
        );
    }

    #[test]
    fn test_external_sort_empty() {
        let sf = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(Vec::new(), FlexTypeEnum::Integer).unwrap(),
        )])
        .unwrap();

        let sort_keys = vec![SortKey::asc(0)];
        let sorted = external_sort(&sf, &sort_keys).unwrap();
        assert_eq!(sorted.num_rows().unwrap(), 0);
    }

    #[test]
    fn test_external_sort_single_row() {
        let sf = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(vec![FlexType::Integer(42)], FlexTypeEnum::Integer).unwrap(),
        )])
        .unwrap();

        let sort_keys = vec![SortKey::asc(0)];
        let sorted = external_sort(&sf, &sort_keys).unwrap();
        let rows = sorted.iter_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec![FlexType::Integer(42)]);
    }
}
