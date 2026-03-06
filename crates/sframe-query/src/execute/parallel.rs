//! Data-parallel query execution via input slicing.
//!
//! Divides a plan's input rows across N worker threads, where each
//! worker runs the full operator pipeline on its slice independently.
//! Each worker writes its output to a segment file on CacheFs.
//! After all workers finish, the segments are assembled into an SFrame.

use std::sync::Arc;

use rayon::prelude::*;

use sframe_types::error::Result;
use sframe_types::flex_type::FlexTypeEnum;

use crate::planner::{clone_plan_with_row_range, LogicalOp, PlannerNode};

/// Minimum row count to justify parallel execution.
#[allow(dead_code)]
const MIN_ROWS_FOR_PARALLEL: u64 = 10_000;

/// Check if a plan is parallel-sliceable and return the total row count.
///
/// A plan is parallel-sliceable when:
/// - Every leaf node is `SFrameSource`
/// - All `SFrameSource` leaves have the same path (or are the same Arc)
/// - All `SFrameSource` leaves read the full range (begin_row=0, end_row=num_rows)
/// - No `Reduce`, `Append`, `Union`, `MaterializedSource`, or `Range` operators
pub fn parallel_slice_row_count(plan: &Arc<PlannerNode>) -> Option<u64> {
    let mut total_rows: Option<u64> = None;
    let mut source_path: Option<String> = None;

    if !check_sliceable(plan, &mut total_rows, &mut source_path) {
        return None;
    }

    total_rows
}

/// Recursively check sliceability constraints.
fn check_sliceable(
    node: &Arc<PlannerNode>,
    total_rows: &mut Option<u64>,
    source_path: &mut Option<String>,
) -> bool {
    match &node.op {
        LogicalOp::SFrameSource {
            path,
            num_rows,
            begin_row,
            end_row,
            ..
        } => {
            // Must be a full-range source
            if *begin_row != 0 || *end_row != *num_rows {
                return false;
            }

            // All sources must have the same path
            match source_path {
                None => {
                    *source_path = Some(path.clone());
                    *total_rows = Some(*num_rows);
                }
                Some(existing) => {
                    if existing != path {
                        return false;
                    }
                    // num_rows should match (same SFrame)
                    if *total_rows != Some(*num_rows) {
                        return false;
                    }
                }
            }
            true
        }

        LogicalOp::ParquetSource {
            source_id,
            num_rows,
            begin_row,
            end_row,
            ..
        } => {
            if *begin_row != 0 || *end_row != *num_rows {
                return false;
            }

            match source_path {
                None => {
                    *source_path = Some(source_id.clone());
                    *total_rows = Some(*num_rows);
                }
                Some(existing) => {
                    if existing != source_id {
                        return false;
                    }
                    if *total_rows != Some(*num_rows) {
                        return false;
                    }
                }
            }
            true
        }

        // These operators cannot be sliced
        LogicalOp::Reduce { .. }
        | LogicalOp::Append
        | LogicalOp::Union
        | LogicalOp::MaterializedSource { .. }
        | LogicalOp::Range { .. } => false,

        // These operators are fine — recurse into inputs
        LogicalOp::Project { .. }
        | LogicalOp::Filter { .. }
        | LogicalOp::Transform { .. }
        | LogicalOp::BinaryTransform { .. }
        | LogicalOp::GeneralizedTransform { .. }
        | LogicalOp::LogicalFilter
        | LogicalOp::ColumnUnion => {
            node.inputs
                .iter()
                .all(|input| check_sliceable(input, total_rows, source_path))
        }
    }
}

/// Execute a plan in parallel by slicing input rows across workers.
///
/// Each worker compiles and runs its slice sequentially, writing output
/// to a segment file on CacheFs. After all workers finish, the segments
/// are assembled into an SFrame on CacheFs.
///
/// Returns the CacheFs SFrame path. The caller is responsible for
/// cleanup via `cache_fs.remove_dir()` when the result is no longer needed.
pub fn execute_parallel(
    plan: &Arc<PlannerNode>,
    total_rows: u64,
    column_names: &[String],
    dtypes: &[FlexTypeEnum],
) -> Result<String> {
    let n_workers = rayon::current_num_threads().max(1);

    // Build N plans with row-range-scoped sources
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

    let cache_fs = sframe_io::cache_fs::global_cache_fs();
    let base_path = cache_fs.alloc_dir();
    let vfs = Arc::new(sframe_io::vfs::ArcCacheFsVfs(cache_fs.clone()));

    // Ensure directory exists
    sframe_io::vfs::VirtualFileSystem::mkdir_p(&*vfs, &base_path)?;

    // Each worker writes a segment file and returns metadata.
    let worker_results: Vec<Result<(String, Vec<u64>, u64)>> = worker_plans
        .into_par_iter()
        .enumerate()
        .map(|(i, plan)| {
            let seg_name = format!("seg.{i:04}");
            let seg_path = format!("{base_path}/{seg_name}");
            let file = sframe_io::vfs::VirtualFileSystem::open_write(&*vfs, &seg_path)?;
            let seg_writer = sframe_storage::segment_writer::BufferedSegmentWriter::new(file, dtypes);

            let mut iter = super::compile_single_threaded(&plan)?;
            let (segment_sizes, row_count) =
                super::consumer::consume_to_segment(&mut iter, seg_writer, dtypes)?;

            Ok((seg_name, segment_sizes, row_count))
        })
        .collect();

    // Collect results, propagating any worker error.
    let mut segment_files = Vec::new();
    let mut all_segment_sizes = Vec::new();
    let mut total_written: u64 = 0;
    for result in worker_results {
        let (seg_name, sizes, rows) = result?;
        segment_files.push(seg_name);
        all_segment_sizes.push(sizes);
        total_written += rows;
    }

    // Assemble SFrame metadata
    let col_name_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
    sframe_storage::sframe_writer::assemble_sframe_from_segments(
        &*vfs,
        &base_path,
        &col_name_refs,
        dtypes,
        &segment_files,
        &all_segment_sizes,
        total_written,
        &std::collections::HashMap::new(),
    )?;

    Ok(base_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::SFrameRows;
    use sframe_types::flex_type::FlexTypeEnum;

    #[test]
    fn test_sliceable_filter_source() {
        use sframe_types::flex_type::FlexType;

        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            100_000,
        );
        let filtered = PlannerNode::filter(
            source,
            0,
            Arc::new(|v: &FlexType| matches!(v, FlexType::Integer(i) if *i > 0)),
        );

        assert_eq!(parallel_slice_row_count(&filtered), Some(100_000));
    }

    #[test]
    fn test_sliceable_shared_source() {
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            100_000,
        );
        // Shared source: same Arc feeds both data and mask of LogicalFilter
        let mask = PlannerNode::transform(
            source.clone(),
            0,
            Arc::new(|v: &sframe_types::flex_type::FlexType| v.clone()),
            FlexTypeEnum::Integer,
        );
        let lf = PlannerNode::logical_filter(source, mask);

        assert_eq!(parallel_slice_row_count(&lf), Some(100_000));
    }

    #[test]
    fn test_not_sliceable_reduce() {
        let source = PlannerNode::sframe_source(
            "test.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            100_000,
        );
        // Reduce has its own parallel path
        let reduced = PlannerNode::reduce(
            source,
            Arc::new(crate::algorithms::aggregators::SumAggregator::new()),
        );
        assert_eq!(parallel_slice_row_count(&reduced), None);
    }

    #[test]
    fn test_not_sliceable_append() {
        let s1 = PlannerNode::sframe_source(
            "t1.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        let s2 = PlannerNode::sframe_source(
            "t2.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            100,
        );
        let appended = PlannerNode::append(s1, s2);
        assert_eq!(parallel_slice_row_count(&appended), None);
    }

    #[test]
    fn test_not_sliceable_materialized() {
        let mat = PlannerNode::materialized(SFrameRows::empty(&[FlexTypeEnum::Integer]));
        assert_eq!(parallel_slice_row_count(&mat), None);
    }

    #[test]
    fn test_not_sliceable_range() {
        let range = PlannerNode::range(0, 1, 100);
        assert_eq!(parallel_slice_row_count(&range), None);
    }

    #[test]
    fn test_not_sliceable_different_paths() {
        let s1 = PlannerNode::sframe_source(
            "a.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            100_000,
        );
        let s2 = PlannerNode::sframe_source(
            "b.sf",
            vec!["a".into()],
            vec![FlexTypeEnum::Integer],
            100_000,
        );
        // LogicalFilter with different-path sources
        let lf = PlannerNode::logical_filter(s1, s2);
        assert_eq!(parallel_slice_row_count(&lf), None);
    }

    #[test]
    fn test_execute_parallel_roundtrip() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let path = format!("{manifest}/../../samples/business.sf");
        let reader = sframe_storage::sframe_reader::SFrameReader::open(&path).unwrap();
        let col_names: Vec<String> = reader.column_names().to_vec();
        let col_types: Vec<FlexTypeEnum> = reader
            .group_index
            .columns
            .iter()
            .map(|c| c.dtype)
            .collect();
        let num_rows = reader.num_rows();

        let source =
            PlannerNode::sframe_source(&path, col_names.clone(), col_types.clone(), num_rows);

        // Execute in parallel
        let result_path =
            super::execute_parallel(&source, num_rows, &col_names, &col_types).unwrap();

        // Read back from CacheFs and verify
        let cache_fs = sframe_io::cache_fs::global_cache_fs();
        let vfs = sframe_io::vfs::ArcCacheFsVfs(cache_fs.clone());
        let result_meta =
            sframe_storage::sframe_reader::SFrameMetadata::open_with_fs(&vfs, &result_path)
                .unwrap();

        // Total rows across all segments should match
        let total: u64 = result_meta.group_index.columns[0]
            .segment_sizes
            .iter()
            .sum();
        assert_eq!(total, num_rows);

        // Column count should match
        assert_eq!(result_meta.group_index.columns.len(), col_types.len());

        // Clean up
        cache_fs.remove_dir(&result_path).unwrap();
    }
}
