//! Reduce operator: full aggregation (sequential and parallel).
//!
//! Parallel reduce works by checking if the Reduce's input subplan is
//! parallel-sliceable (via `parallel_slice_row_count`). If so, the input
//! rows are divided across rayon workers, each worker compiles and runs
//! the full pipeline on its slice, reduces locally, and partial aggregators
//! are merged. This works for any source type (SFrame, Parquet, etc.)
//! and any pipeline shape (filters, transforms, etc.).

use std::sync::Arc;

use sframe_types::error::Result;

use crate::batch::{ColumnData, SFrameRows};
use crate::planner::{clone_plan_with_row_range, Aggregator, LogicalOp, PlannerNode};

use super::batch_iter::BatchIterator;
use super::parallel::parallel_slice_row_count;

/// Extracted parallel reduce plan: the input subplan is sliceable.
pub(super) struct ParallelReducePlan {
    /// The input subplan (below the Reduce node).
    input_plan: Arc<PlannerNode>,
    total_rows: u64,
    aggregator: Arc<dyn Aggregator>,
}

/// Try to extract a plan suitable for parallel reduce.
///
/// Checks if the node is a `Reduce` whose input subplan is
/// parallel-sliceable. Works for any source type.
pub(super) fn try_extract_parallel_reduce_plan(node: &Arc<PlannerNode>) -> Option<ParallelReducePlan> {
    let aggregator = match &node.op {
        LogicalOp::Reduce { aggregator } => aggregator.clone(),
        _ => return None,
    };

    let input = node.inputs.first()?;
    let total_rows = parallel_slice_row_count(input)?;

    Some(ParallelReducePlan {
        input_plan: input.clone(),
        total_rows,
        aggregator,
    })
}

/// Execute a parallel reduce by splitting rows across rayon workers.
///
/// Each worker gets a row-range slice of the input plan, compiles and
/// runs it through the normal execution engine, and reduces locally.
/// Partial aggregators are merged at the end.
pub(super) fn execute_parallel_reduce(plan: &ParallelReducePlan) -> Result<SFrameRows> {
    use rayon::prelude::*;

    let n_workers = rayon::current_num_threads().max(1);

    // Build per-worker plans with row-range-scoped sources
    let worker_plans: Vec<Arc<PlannerNode>> = (0..n_workers)
        .filter_map(|i| {
            let begin = (i as u64 * plan.total_rows) / n_workers as u64;
            let end = ((i as u64 + 1) * plan.total_rows) / n_workers as u64;
            if begin >= end {
                return None;
            }
            Some(clone_plan_with_row_range(&plan.input_plan, begin, end))
        })
        .collect();

    // Each worker compiles its pipeline slice and reduces locally
    let partial_aggs: Vec<Result<Box<dyn Aggregator>>> = worker_plans
        .into_par_iter()
        .map(|worker_plan| {
            let mut local_agg = plan.aggregator.box_clone();
            let mut iter = super::compile_single_threaded(&worker_plan)?;

            while let Some(batch_result) = iter.next_batch() {
                let batch = batch_result?;
                for i in 0..batch.num_rows() {
                    let row = batch.row(i);
                    local_agg.add(&row);
                }
            }

            Ok(local_agg)
        })
        .collect();

    // Merge all partial aggregators sequentially
    let mut final_agg = plan.aggregator.box_clone();
    for partial in partial_aggs {
        let partial = partial?;
        final_agg.merge(&*partial);
    }

    let result = final_agg.finalize();
    let dtype = result.type_enum();
    let mut col = ColumnData::empty(dtype);
    col.push(&result)?;
    SFrameRows::new(vec![col])
}

/// Execute a reduce operation by consuming the entire input BatchIterator.
pub(super) fn execute_reduce_iter(
    input: &mut BatchIterator,
    aggregator: Arc<dyn Aggregator>,
) -> Result<SFrameRows> {
    let mut agg = aggregator.box_clone();

    while let Some(batch_result) = input.next_batch() {
        let batch = batch_result?;
        for i in 0..batch.num_rows() {
            let row = batch.row(i);
            agg.add(&row);
        }
    }

    let result = agg.finalize();
    let dtype = result.type_enum();
    let mut col = ColumnData::empty(dtype);
    col.push(&result)?;
    SFrameRows::new(vec![col])
}
