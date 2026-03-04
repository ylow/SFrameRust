//! Data-parallel query execution via input slicing.
//!
//! Divides a plan's input rows across N worker threads, where each
//! worker runs the full operator pipeline on its slice independently.
//! Results are collected in order and yielded as a stream.

use std::sync::Arc;

use futures::stream;
use rayon::prelude::*;

use sframe_types::error::Result;

use crate::batch::SFrameRows;
use crate::planner::{clone_plan_with_row_range, LogicalOp, PlannerNode};

use super::BatchStream;

/// Minimum row count to justify parallel execution.
const MIN_ROWS_FOR_PARALLEL: u64 = 10_000;

/// Check if a plan is parallel-sliceable and return the total row count.
///
/// A plan is parallel-sliceable when:
/// - Every leaf node is `SFrameSource`
/// - All `SFrameSource` leaves have the same path (or are the same Arc)
/// - All `SFrameSource` leaves read the full range (begin_row=0, end_row=num_rows)
/// - No `Reduce`, `Append`, `Union`, `MaterializedSource`, or `Range` operators
pub(super) fn parallel_slice_row_count(plan: &Arc<PlannerNode>) -> Option<u64> {
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
        | LogicalOp::LogicalFilter => {
            node.inputs
                .iter()
                .all(|input| check_sliceable(input, total_rows, source_path))
        }
    }
}

/// Execute a plan in parallel by slicing the input rows across workers.
///
/// Each worker gets a cloned plan with adjusted source row ranges,
/// compiles and materializes it independently, then results are
/// yielded in order as a stream.
pub(super) fn compile_parallel(
    plan: &Arc<PlannerNode>,
    total_rows: u64,
) -> Option<BatchStream> {
    let n_workers = rayon::current_num_threads().max(1);

    // Not worth parallelizing for small data or single thread
    if total_rows < MIN_ROWS_FOR_PARALLEL || n_workers <= 1 {
        return None;
    }

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

    // Execute all workers in parallel, each with its own tokio runtime
    let results: Vec<Result<SFrameRows>> = worker_plans
        .into_par_iter()
        .map(|plan| {
            let stream = super::compile_single_threaded(&plan)?;
            super::materialize_sync(stream)
        })
        .collect();

    // Yield results in order as a stream
    Some(Box::pin(stream::iter(results)))
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
