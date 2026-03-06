//! External sort for datasets that exceed the in-memory sort budget.
//!
//! Uses a 5-phase algorithm:
//! 1. Build a quantile sketch of the primary sort key.
//! 2. Determine partition cut points from the sketch.
//! 3. Partition data into P temporary SFrames.
//! 4. Sort each partition in-memory.
//! 5. Write sorted partitions sequentially into a single output SFrame.

use std::sync::Arc;

use rayon::prelude::*;

use sframe_query::algorithms::quantile_sketch::QuantileSketch;
use sframe_query::algorithms::sort::{compare_flex_type, SortKey, SortOrder};
use sframe_query::execute::{compile, parallel_slice_row_count};
use sframe_query::planner::{clone_plan_with_row_range, PlannerNode};
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexType;

use crate::sframe::{SFrame, SFrameBuilder};

const CHUNK_SIZE: usize = 8192;
const CHANNEL_BUFFER_DEPTH: usize = 8;
const PARTITION_FLUSH_ROWS: usize = 1024;

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
    let budget = sframe_config::global().sort_memory_budget;
    let estimated_size = sf.estimate_size();

    // Check if parallel execution is possible
    let plan = sf.fuse_plan()?;
    let total_rows = parallel_slice_row_count(&plan);

    // Phase 1: Build quantile sketch (parallel if possible)
    let sketch = if let Some(total_rows) = total_rows {
        build_sketch_parallel(&plan, primary_key.column, total_rows)?
    } else {
        build_sketch(sf, primary_key.column)?
    };

    if sketch.count() == 0 {
        return sf.head(0);
    }

    // Phase 2: Determine partition cut points
    let num_partitions = (estimated_size / budget).max(2);
    let cut_points = sketch.quantiles(num_partitions);

    // Phase 3: Partition data (parallel if possible)
    let partitions = if let Some(total_rows) = total_rows {
        partition_data_parallel(&plan, sf, primary_key, &cut_points, total_rows)?
    } else {
        partition_data(sf, primary_key, &cut_points)?
    };

    // Phase 4+5: Sort partitions in parallel + assemble output
    sort_partitions_and_assemble(partitions, sort_keys, primary_key, sf)
}

/// Phase 1: Stream the data and build a quantile sketch of the primary sort key.
fn build_sketch(sf: &SFrame, key_column: usize) -> Result<QuantileSketch> {
    let mut sketch = QuantileSketch::new(0.01);

    let mut stream = sf.compile_stream()?;
    while let Some(batch_result) = stream.next_batch() {
        let batch = batch_result?;
        let col = batch.column(key_column);
        for i in 0..batch.num_rows() {
            sketch.insert(col.get(i));
        }
    }

    sketch.finish();
    Ok(sketch)
}

/// Split a parallel-sliceable plan into N worker plans with row ranges.
fn make_worker_plans(
    plan: &Arc<PlannerNode>,
    total_rows: u64,
    n_workers: usize,
) -> Vec<Arc<PlannerNode>> {
    (0..n_workers)
        .filter_map(|i| {
            let begin = (i as u64 * total_rows) / n_workers as u64;
            let end = ((i as u64 + 1) * total_rows) / n_workers as u64;
            if begin >= end {
                return None;
            }
            Some(clone_plan_with_row_range(plan, begin, end))
        })
        .collect()
}

/// Phase 1 (parallel): Build quantile sketch using N rayon workers, each
/// reading a row-range slice, then merge all sketches.
fn build_sketch_parallel(
    plan: &Arc<PlannerNode>,
    key_column: usize,
    total_rows: u64,
) -> Result<QuantileSketch> {
    let n_workers = rayon::current_num_threads().max(1);
    let worker_plans = make_worker_plans(plan, total_rows, n_workers);

    let sketches: Vec<Result<QuantileSketch>> = worker_plans
        .into_par_iter()
        .map(|wp| {
            let mut sketch = QuantileSketch::new(0.01);
            let mut iter = compile(&wp)?;
            while let Some(batch_result) = iter.next_batch() {
                let batch = batch_result?;
                let col = batch.column(key_column);
                for i in 0..batch.num_rows() {
                    sketch.insert(col.get(i));
                }
            }
            sketch.finish();
            Ok(sketch)
        })
        .collect();

    let mut combined = QuantileSketch::new(0.01);
    for s in sketches {
        combined.merge(&s?);
    }
    Ok(combined)
}

/// Phase 3 (parallel): Partition data using N rayon readers and P MPSC writer threads.
///
/// Each reader reads a row-range slice, classifies rows by partition, and sends
/// batched column data through sync_channels to P writer threads. Each writer
/// drains its channel into an SFrameBuilder.
fn partition_data_parallel(
    plan: &Arc<PlannerNode>,
    sf: &SFrame,
    primary_key: &SortKey,
    cut_points: &[FlexType],
    total_rows: u64,
) -> Result<Vec<SFrame>> {
    let num_partitions = cut_points.len() + 1;
    let column_names = sf.column_names().to_vec();
    let dtypes = sf.column_types();
    let key_column = primary_key.column;
    let ncols = dtypes.len();
    let n_workers = rayon::current_num_threads().max(1);
    let worker_plans = make_worker_plans(plan, total_rows, n_workers);

    // Create P MPSC channels (one per partition)
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..num_partitions)
        .map(|_| std::sync::mpsc::sync_channel::<Vec<Vec<FlexType>>>(CHANNEL_BUFFER_DEPTH))
        .unzip();

    // Spawn P writer threads — each drains its channel into an SFrameBuilder
    let writer_handles: Vec<std::thread::JoinHandle<Result<SFrame>>> = receivers
        .into_iter()
        .map(|rx| {
            let names = column_names.clone();
            let dt = dtypes.clone();
            std::thread::spawn(move || {
                let mut builder = SFrameBuilder::anonymous(names, dt)?;
                for columns in rx {
                    builder.write_columns(&columns)?;
                }
                builder.finish()
            })
        })
        .collect();

    // N rayon readers: read slice -> scatter rows -> send to channels
    let reader_results: Vec<Result<()>> = worker_plans
        .into_par_iter()
        .map(|wp| {
            let my_senders = senders.to_vec();
            let mut iter = compile(&wp)?;

            // Per-partition row buffers (column-major)
            let mut bufs: Vec<Vec<Vec<FlexType>>> = (0..num_partitions)
                .map(|_| (0..ncols).map(|_| Vec::new()).collect())
                .collect();

            while let Some(batch_result) = iter.next_batch() {
                let batch = batch_result?;
                for row_idx in 0..batch.num_rows() {
                    let key_val = batch.column(key_column).get(row_idx);
                    let part_idx = find_partition(&key_val, cut_points);
                    for col_idx in 0..ncols {
                        bufs[part_idx][col_idx].push(batch.column(col_idx).get(row_idx));
                    }
                    if bufs[part_idx][0].len() >= PARTITION_FLUSH_ROWS {
                        let buf = std::mem::replace(
                            &mut bufs[part_idx],
                            (0..ncols).map(|_| Vec::new()).collect(),
                        );
                        my_senders[part_idx]
                            .send(buf)
                            .map_err(|_| SFrameError::Format(
                                "Partition writer closed unexpectedly".into(),
                            ))?;
                    }
                }
            }

            // Flush remaining buffers
            for (part_idx, buf) in bufs.into_iter().enumerate() {
                if !buf.is_empty() && !buf[0].is_empty() {
                    my_senders[part_idx]
                        .send(buf)
                        .map_err(|_| SFrameError::Format(
                            "Partition writer closed unexpectedly".into(),
                        ))?;
                }
            }
            drop(my_senders);
            Ok(())
        })
        .collect();

    // Close original senders so writer threads see EOF
    drop(senders);

    // Check reader errors
    for r in reader_results {
        r?;
    }

    // Collect writer results
    let mut partitions = Vec::with_capacity(num_partitions);
    for handle in writer_handles {
        let result = handle
            .join()
            .map_err(|_| SFrameError::Format("Partition writer thread panicked".into()))?;
        partitions.push(result?);
    }

    Ok(partitions)
}

/// Phase 4+5 (parallel sort, sequential assembly): Sort each partition
/// in parallel using rayon, then write sorted partitions to output
/// in the correct global order.
fn sort_partitions_and_assemble(
    partitions: Vec<SFrame>,
    sort_keys: &[SortKey],
    primary_key: &SortKey,
    sf: &SFrame,
) -> Result<SFrame> {
    // Determine output order: ascending → 0..P, descending → P-1..0
    let partition_order: Vec<usize> = if primary_key.order == SortOrder::Descending {
        (0..partitions.len()).rev().collect()
    } else {
        (0..partitions.len()).collect()
    };

    // Sort all partitions in parallel
    let sort_keys_owned = sort_keys.to_vec();
    let sorted: Vec<Option<Result<SFrame>>> = partition_order
        .par_iter()
        .map(|&part_idx| {
            let num_rows = partitions[part_idx].num_rows().unwrap_or(0);
            if num_rows == 0 {
                return None;
            }
            Some(partitions[part_idx].sort_in_memory(&sort_keys_owned))
        })
        .collect();

    // Write sorted partitions to output in order
    let mut builder =
        SFrameBuilder::anonymous(sf.column_names().to_vec(), sf.column_types())?;

    for sort_result in sorted.into_iter().flatten() {
        let sorted_part = sort_result?;
        let mut stream = sorted_part.compile_stream()?;
        while let Some(batch_result) = stream.next_batch() {
            builder.write_batch_chunked(&batch_result?, CHUNK_SIZE)?;
        }
    }

    drop(partitions);
    builder.finish()
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
    let mut stream = sf.compile_stream()?;
    while let Some(batch_result) = stream.next_batch() {
        let batch = batch_result?;
        scatter_batch(&batch, primary_key.column, cut_points, &mut builders)?;
    }

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
                (0..n).map(FlexType::Integer).collect(),
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

    #[test]
    fn test_partition_data_parallel() {
        // Values 0..999 should scatter across partitions
        let n = 1000i64;
        let values: Vec<FlexType> = (0..n).map(FlexType::Integer).collect();
        let sf = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(values, FlexTypeEnum::Integer).unwrap(),
        )])
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("part_test.sf");
        sf.save(path.to_str().unwrap()).unwrap();
        let sf2 = SFrame::read(path.to_str().unwrap()).unwrap();

        let plan = sf2.fuse_plan().unwrap();
        let total_rows = parallel_slice_row_count(&plan).unwrap();

        // Use cut points [250, 500, 750] -> 4 partitions
        let cut_points = vec![
            FlexType::Integer(250),
            FlexType::Integer(500),
            FlexType::Integer(750),
        ];
        let primary_key = SortKey::asc(0);

        let partitions =
            partition_data_parallel(&plan, &sf2, &primary_key, &cut_points, total_rows).unwrap();
        assert_eq!(partitions.len(), 4);

        // Total rows across all partitions should equal input
        let total: u64 = partitions.iter().map(|p| p.num_rows().unwrap_or(0)).sum();
        assert_eq!(total, n as u64);

        // Each partition should have roughly 250 rows
        for (i, p) in partitions.iter().enumerate() {
            let rows = p.num_rows().unwrap_or(0);
            assert!(
                rows > 200 && rows < 300,
                "partition {i} has {rows} rows, expected ~250"
            );
        }
    }

    #[test]
    fn test_build_sketch_parallel() {
        let n = 10_000i64;
        let values: Vec<FlexType> = (0..n).map(FlexType::Integer).collect();
        let sf = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(values, FlexTypeEnum::Integer).unwrap(),
        )])
        .unwrap();

        // Save + read to get SFrameSource plan (parallel-sliceable)
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sketch_test.sf");
        sf.save(path.to_str().unwrap()).unwrap();
        let sf2 = SFrame::read(path.to_str().unwrap()).unwrap();

        let plan = sf2.fuse_plan().unwrap();
        let total_rows = parallel_slice_row_count(&plan).unwrap();

        let sketch = build_sketch_parallel(&plan, 0, total_rows).unwrap();
        assert_eq!(sketch.count(), n as usize);

        // Median should be near 5000
        let median = sketch.query(0.5);
        match median {
            FlexType::Integer(v) => assert!(
                (v - 5000).unsigned_abs() < 500,
                "parallel median off: {v} vs 5000"
            ),
            other => panic!("Expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn test_sort_partitions_and_assemble() {
        // Create 3 partitions with unsorted data within each
        let p0 = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(
                vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(2)],
                FlexTypeEnum::Integer,
            )
            .unwrap(),
        )])
        .unwrap();
        let p1 = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(
                vec![FlexType::Integer(6), FlexType::Integer(4), FlexType::Integer(5)],
                FlexTypeEnum::Integer,
            )
            .unwrap(),
        )])
        .unwrap();
        let p2 = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(
                vec![FlexType::Integer(9), FlexType::Integer(7), FlexType::Integer(8)],
                FlexTypeEnum::Integer,
            )
            .unwrap(),
        )])
        .unwrap();

        let partitions = vec![p0, p1, p2];
        let sort_keys = vec![SortKey::asc(0)];
        // Build a schema SFrame to pass for column names/types
        let sf_schema = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(Vec::new(), FlexTypeEnum::Integer).unwrap(),
        )])
        .unwrap();

        let result =
            sort_partitions_and_assemble(partitions, &sort_keys, &sort_keys[0], &sf_schema)
                .unwrap();

        let rows = result.iter_rows().unwrap();
        assert_eq!(rows.len(), 9);
        for i in 0..9 {
            assert_eq!(rows[i], vec![FlexType::Integer(i as i64 + 1)]);
        }
    }

    #[test]
    fn test_external_sort_parallel_path() {
        let n = 5000i64;
        let values: Vec<FlexType> = (0..n).rev().map(FlexType::Integer).collect(); // descending

        let sf = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(values, FlexTypeEnum::Integer).unwrap(),
        )])
        .unwrap();

        // Save + read to get parallel-sliceable plan
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("par_sort_test.sf");
        sf.save(path.to_str().unwrap()).unwrap();
        let sf2 = SFrame::read(path.to_str().unwrap()).unwrap();

        // Verify plan is parallel-sliceable
        let plan = sf2.fuse_plan().unwrap();
        assert!(parallel_slice_row_count(&plan).is_some());

        // Sort ascending
        let sort_keys = vec![SortKey::asc(0)];
        let sorted = external_sort(&sf2, &sort_keys).unwrap();

        let rows = sorted.iter_rows().unwrap();
        assert_eq!(rows.len(), n as usize);
        for i in 0..rows.len() {
            assert_eq!(rows[i], vec![FlexType::Integer(i as i64)]);
        }
    }

    #[test]
    fn test_external_sort_parallel_descending() {
        let n = 5000i64;
        let values: Vec<FlexType> = (0..n).map(FlexType::Integer).collect(); // ascending

        let sf = SFrame::from_columns(vec![(
            "x",
            SArray::from_vec(values, FlexTypeEnum::Integer).unwrap(),
        )])
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("par_sort_desc.sf");
        sf.save(path.to_str().unwrap()).unwrap();
        let sf2 = SFrame::read(path.to_str().unwrap()).unwrap();

        let sort_keys = vec![SortKey::desc(0)];
        let sorted = external_sort(&sf2, &sort_keys).unwrap();

        let rows = sorted.iter_rows().unwrap();
        assert_eq!(rows.len(), n as usize);
        for i in 0..rows.len() {
            assert_eq!(rows[i], vec![FlexType::Integer(n - 1 - i as i64)]);
        }
    }

    #[test]
    fn test_external_sort_parallel_multi_key() {
        let n = 200i64;
        let mut col_a = Vec::new();
        let mut col_b = Vec::new();
        for a in (0..10).rev() {
            for b in (0..20).rev() {
                col_a.push(FlexType::Integer(a));
                col_b.push(FlexType::Integer(b));
            }
        }

        let sf = SFrame::from_columns(vec![
            (
                "a",
                SArray::from_vec(col_a, FlexTypeEnum::Integer).unwrap(),
            ),
            (
                "b",
                SArray::from_vec(col_b, FlexTypeEnum::Integer).unwrap(),
            ),
        ])
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("par_sort_multi.sf");
        sf.save(path.to_str().unwrap()).unwrap();
        let sf2 = SFrame::read(path.to_str().unwrap()).unwrap();

        let sort_keys = vec![SortKey::asc(0), SortKey::desc(1)];
        let sorted = external_sort(&sf2, &sort_keys).unwrap();

        let rows = sorted.iter_rows().unwrap();
        assert_eq!(rows.len(), n as usize);

        // Verify: asc on a, desc on b
        for i in 1..rows.len() {
            let a_prev = &rows[i - 1][0];
            let a_curr = &rows[i][0];
            let b_prev = &rows[i - 1][1];
            let b_curr = &rows[i][1];
            let cmp_a = compare_flex_type(a_prev, a_curr);
            assert!(
                cmp_a != std::cmp::Ordering::Greater,
                "row {i}: a not ascending"
            );
            if cmp_a == std::cmp::Ordering::Equal {
                assert!(
                    compare_flex_type(b_prev, b_curr) != std::cmp::Ordering::Less,
                    "row {i}: b not descending within same a"
                );
            }
        }
    }
}
