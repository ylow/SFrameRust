//! Reduce operator: full aggregation (sequential and parallel).

use std::sync::Arc;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::local_fs::LocalFileSystem;
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_storage::sframe_reader::SFrameMetadata;
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::{ColumnData, SFrameRows};
use crate::planner::{Aggregator, LogicalOp, PlannerNode};

use super::batch_iter::BatchIterator;
use super::source;

/// Extracted parallel reduce plan.
pub(super) struct ParallelReducePlan {
    path: String,
    column_types: Vec<FlexTypeEnum>,
    column_indices: Vec<usize>,
    aggregator: Arc<dyn Aggregator>,
    _keep_alive: Option<Arc<dyn Send + Sync>>,
}

/// Try to extract a plan suitable for parallel reduce.
///
/// Matches two patterns:
/// - `Reduce → SFrameSource` (identity project was optimized away)
/// - `Reduce → Project → SFrameSource`
///
/// Returns `None` if the plan doesn't match these patterns.
pub(super) fn try_extract_parallel_reduce_plan(node: &Arc<PlannerNode>) -> Option<ParallelReducePlan> {
    let aggregator = match &node.op {
        LogicalOp::Reduce { aggregator } => aggregator.clone(),
        _ => return None,
    };

    let input = node.inputs.first()?;

    // Pattern 1: Reduce → SFrameSource (only full-range sources)
    if let LogicalOp::SFrameSource {
        path,
        column_types,
        num_rows,
        begin_row,
        end_row,
        _keep_alive,
        ..
    } = &input.op
    {
        // Only use parallel reduce for full-range sources
        if *begin_row == 0 && *end_row == *num_rows {
            let column_indices: Vec<usize> = (0..column_types.len()).collect();
            return Some(ParallelReducePlan {
                path: path.clone(),
                column_types: column_types.clone(),
                column_indices,
                aggregator,
                _keep_alive: _keep_alive.clone(),
            });
        }
    }

    // Pattern 2: Reduce → Project → SFrameSource
    if let LogicalOp::Project { column_indices } = &input.op {
        let source = input.inputs.first()?;
        if let LogicalOp::SFrameSource {
            path,
            column_types,
            num_rows,
            begin_row,
            end_row,
            _keep_alive,
            ..
        } = &source.op
        {
            if *begin_row != 0 || *end_row != *num_rows {
                return None;
            }
            return Some(ParallelReducePlan {
                path: path.clone(),
                column_types: column_types.clone(),
                column_indices: column_indices.clone(),
                aggregator,
                _keep_alive: _keep_alive.clone(),
            });
        }
    }

    None
}

/// Execute a parallel reduce by splitting rows across rayon workers.
///
/// Divides the total row range into N equal slices (one per rayon thread),
/// each worker reads its slice (with block-level skipping), creates a local
/// aggregator, feeds rows, then all partial aggregators are merged sequentially.
pub(super) fn execute_parallel_reduce(plan: &ParallelReducePlan) -> Result<SFrameRows> {
    use rayon::prelude::*;

    let is_cache = plan.path.starts_with("cache://");
    let vfs: Arc<dyn VirtualFileSystem> = if is_cache {
        Arc::new(ArcCacheFsVfs(global_cache_fs().clone()))
    } else {
        Arc::new(LocalFileSystem)
    };

    let meta = SFrameMetadata::open_with_fs(&*vfs, &plan.path)?;
    let segment_paths: Vec<String> = meta
        .group_index
        .segment_files
        .iter()
        .map(|f| format!("{}/{}", plan.path, f))
        .collect();

    let segment_sizes: Vec<u64> = meta
        .group_index
        .columns[0]
        .segment_sizes
        .clone();

    let total_rows: u64 = segment_sizes.iter().sum();
    let n_workers = rayon::current_num_threads().max(1);
    let single_column = plan.column_indices.len() == 1;

    // Build equal row-range slices for each worker
    let worker_ranges: Vec<(u64, u64)> = (0..n_workers)
        .filter_map(|i| {
            let begin = (i as u64 * total_rows) / n_workers as u64;
            let end = ((i as u64 + 1) * total_rows) / n_workers as u64;
            if begin >= end { None } else { Some((begin, end)) }
        })
        .collect();

    // Process row ranges in parallel, reading in sub-chunks to bound memory
    const CHUNK_SIZE: u64 = 256 * 1024; // 256K rows per chunk

    let partial_aggs: Vec<Result<Box<dyn Aggregator>>> = worker_ranges
        .par_iter()
        .map(|&(begin, end)| {
            let mut local_agg = plan.aggregator.box_clone();

            // Process the range in fixed-size chunks to avoid materializing
            // the entire worker slice at once
            let mut chunk_begin = begin;
            while chunk_begin < end {
                let chunk_end = (chunk_begin + CHUNK_SIZE).min(end);

                let columns = source::read_projected_row_range(
                    &*vfs,
                    &segment_paths,
                    &segment_sizes,
                    &plan.column_types,
                    &plan.column_indices,
                    chunk_begin,
                    chunk_end,
                )?;

                let num_rows = if columns.is_empty() {
                    0
                } else {
                    columns[0].len()
                };

                if single_column {
                    for i in 0..num_rows {
                        local_agg.add(std::slice::from_ref(&columns[0][i]));
                    }
                } else {
                    for i in 0..num_rows {
                        let row: Vec<FlexType> = columns.iter().map(|c| c[i].clone()).collect();
                        local_agg.add(&row);
                    }
                }

                chunk_begin = chunk_end;
                // `columns` is dropped here, freeing memory before next chunk
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
