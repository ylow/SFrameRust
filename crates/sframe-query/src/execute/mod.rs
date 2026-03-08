//! Physical execution engine.
//!
//! Compiles a logical plan DAG into synchronous generator-based `BatchIterator`
//! pipelines. Each operator is a genawaiter coroutine that yields `BatchResponse`
//! values and receives `BatchCommand` via `resume_with`.
//!
//! Shared subexpressions (same `Arc<PlannerNode>`) are detected and compiled
//! once, with a broadcast adapter fanning the output to all consumers.
//! Rate analysis inserts rebatch adapters at lockstep multi-input operators
//! when input batch boundaries may be misaligned.

mod batch_iter;
mod broadcast;
mod consumer;
mod filter;
mod parallel;
mod range;
mod rebatch;
mod reduce;
mod source;
mod transform;

pub use batch_iter::{BatchCo, BatchCommand, BatchIterator, BatchResponse};

use std::collections::HashMap;
use std::sync::Arc;

use sframe_types::error::Result;

use crate::batch::SFrameRows;
use crate::optimizer;
use crate::planner::{LogicalOp, OperatorRate, PlannerNode};

use broadcast::BroadcastState;

/// Deprecated alias — use `BatchIterator` directly.
pub type BatchStream = BatchIterator;

pub use consumer::consume_to_segment;
pub use parallel::{execute_parallel, parallel_slice_row_count};
pub use reduce::reduce_to_aggregator;

/// Default broadcast buffer capacity (batches per consumer channel).
const BROADCAST_BUFFER: usize = 4;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compile a logical plan node into a BatchIterator.
///
/// Always compiles single-threaded. Data parallelism is handled by
/// the caller via `parallel::execute_parallel()`.
pub fn compile(node: &Arc<PlannerNode>) -> Result<BatchIterator> {
    let node = optimizer::optimize(node);
    compile_single_threaded(&node)
}

/// Single-threaded compilation: fan-out detection (broadcast for shared
/// nodes) and rate-mismatch detection (rebatch at lockstep operators).
pub(super) fn compile_single_threaded(node: &Arc<PlannerNode>) -> Result<BatchIterator> {
    // Phase 1: count how many times each node appears as an input.
    let mut refcounts: HashMap<usize, usize> = HashMap::new();
    count_refs(node, &mut refcounts);

    // Phase 2: assign rate IDs bottom-up.
    let mut rate_ids: HashMap<usize, usize> = HashMap::new();
    let mut rate_counter: usize = 0;
    assign_rate_ids(node, &mut rate_ids, &mut rate_counter);

    // Phase 3: compile with memoization for shared nodes.
    let mut memo: HashMap<usize, BroadcastState> = HashMap::new();
    compile_memoized(node, &refcounts, &rate_ids, &mut memo)
}

// ---------------------------------------------------------------------------
// Phase 1: reference counting
// ---------------------------------------------------------------------------

/// Walk the DAG and count how many times each node appears as an input
/// to some other node.  The root node itself is counted once.
fn count_refs(node: &Arc<PlannerNode>, counts: &mut HashMap<usize, usize>) {
    let id = Arc::as_ptr(node) as usize;
    *counts.entry(id).or_insert(0) += 1;
    if counts[&id] == 1 {
        // First visit — recurse into inputs.
        for input in &node.inputs {
            count_refs(input, counts);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: rate ID assignment
// ---------------------------------------------------------------------------

/// Assign a rate ID to every node in the DAG.
///
/// - `Source` nodes get a fresh rate ID.
/// - `Linear` single-input ops inherit from their input.
/// - `SubLinear` ops get a fresh rate ID.
/// - Multi-input sequential ops (Append, Union): first input's rate.
/// - Multi-input lockstep ops: first input's rate (mismatch detected later).
fn assign_rate_ids(
    node: &Arc<PlannerNode>,
    ids: &mut HashMap<usize, usize>,
    counter: &mut usize,
) {
    let id = Arc::as_ptr(node) as usize;
    if ids.contains_key(&id) {
        return; // already assigned
    }

    // Recurse into inputs first (bottom-up).
    for input in &node.inputs {
        assign_rate_ids(input, ids, counter);
    }

    let rate = match node.op.rate() {
        OperatorRate::Source => {
            let r = *counter;
            *counter += 1;
            r
        }
        OperatorRate::Linear => {
            if let Some(first_input) = node.inputs.first() {
                let input_id = Arc::as_ptr(first_input) as usize;
                ids[&input_id]
            } else {
                // No inputs but classified as Linear — shouldn't happen,
                // treat as source.
                let r = *counter;
                *counter += 1;
                r
            }
        }
        OperatorRate::SubLinear => {
            let r = *counter;
            *counter += 1;
            r
        }
    };

    ids.insert(id, rate);
}

// ---------------------------------------------------------------------------
// Phase 3: memoized compilation
// ---------------------------------------------------------------------------

/// Compile a node, using broadcast for shared nodes.
fn compile_memoized(
    node: &Arc<PlannerNode>,
    refcounts: &HashMap<usize, usize>,
    rate_ids: &HashMap<usize, usize>,
    memo: &mut HashMap<usize, BroadcastState>,
) -> Result<BatchIterator> {
    let id = Arc::as_ptr(node) as usize;

    // If this node was already compiled (shared), hand out a subscriber.
    if let Some(state) = memo.get_mut(&id) {
        return Ok(state.subscribe());
    }

    // Compile the node itself.
    let iter = compile_node(node, refcounts, rate_ids, memo)?;

    // If the node has multiple downstream consumers, wrap in broadcast.
    let consumers = refcounts.get(&id).copied().unwrap_or(1);
    if consumers > 1 {
        let mut bc = BroadcastState::new(iter, consumers, BROADCAST_BUFFER);
        let subscriber = bc.subscribe();
        memo.insert(id, bc);
        return Ok(subscriber);
    }

    Ok(iter)
}

// ---------------------------------------------------------------------------
// Linear operator helper
// ---------------------------------------------------------------------------

/// Create a BatchIterator that applies a transformation to each batch
/// from an input, passing through SkipBatch to the input.
fn linear_operator(
    mut input: BatchIterator,
    process: impl Fn(SFrameRows) -> Result<SFrameRows> + 'static,
) -> BatchIterator {
    BatchIterator::new(move |co: BatchCo| async move {
        let mut cmd = co.yield_(BatchResponse::Ready).await;
        loop {
            match cmd {
                BatchCommand::NextBatch => {
                    match input.next_batch() {
                        None => return,
                        Some(Err(e)) => {
                            cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                        }
                        Some(Ok(batch)) => {
                            cmd = co.yield_(BatchResponse::Batch(process(batch))).await;
                        }
                    }
                }
                BatchCommand::SkipBatch => {
                    match input.skip_batch() {
                        None => return,
                        Some(()) => {
                            cmd = co.yield_(BatchResponse::Skipped).await;
                        }
                    }
                }
                BatchCommand::Start => unreachable!(),
            }
        }
    })
}

/// Create a BatchIterator that filters batches from an input,
/// skipping empty results.
fn filter_operator(
    mut input: BatchIterator,
    filter_fn: impl Fn(SFrameRows) -> Result<SFrameRows> + 'static,
) -> BatchIterator {
    BatchIterator::new(move |co: BatchCo| async move {
        let mut cmd = co.yield_(BatchResponse::Ready).await;
        loop {
            match cmd {
                BatchCommand::NextBatch => {
                    loop {
                        match input.next_batch() {
                            None => return,
                            Some(Err(e)) => {
                                cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                                break;
                            }
                            Some(Ok(batch)) => {
                                match filter_fn(batch) {
                                    Err(e) => {
                                        cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                                        break;
                                    }
                                    Ok(filtered) if filtered.num_rows() == 0 => {
                                        continue; // skip empty batches, pull next
                                    }
                                    Ok(filtered) => {
                                        cmd = co.yield_(BatchResponse::Batch(Ok(filtered))).await;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                BatchCommand::SkipBatch => {
                    // Filter cannot propagate SkipBatch — we don't know
                    // output size without running the predicate. Pull one
                    // input batch and discard.
                    match input.next_batch() {
                        None => return,
                        Some(Err(e)) => {
                            cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                        }
                        Some(Ok(_discarded)) => {
                            cmd = co.yield_(BatchResponse::Skipped).await;
                        }
                    }
                }
                BatchCommand::Start => unreachable!(),
            }
        }
    })
}

// ---------------------------------------------------------------------------
// compile_node: dispatch on LogicalOp
// ---------------------------------------------------------------------------

/// Compile a single node (dispatches on LogicalOp).
fn compile_node(
    node: &Arc<PlannerNode>,
    refcounts: &HashMap<usize, usize>,
    rate_ids: &HashMap<usize, usize>,
    memo: &mut HashMap<usize, BroadcastState>,
) -> Result<BatchIterator> {
    match &node.op {
        LogicalOp::SFrameSource {
            path,
            column_names: _,
            column_types,
            num_rows: _,
            begin_row,
            end_row,
            ..
        } => source::compile_sframe_source(path, column_types, *begin_row, *end_row),

        LogicalOp::ParquetSource { begin_row, end_row, source_fn, .. } => {
            source_fn(*begin_row, *end_row)
        }

        LogicalOp::MaterializedSource { data } => {
            let data = data.clone();
            Ok(BatchIterator::new(move |co: BatchCo| async move {
                let cmd = co.yield_(BatchResponse::Ready).await;
                if matches!(cmd, BatchCommand::NextBatch) {
                    co.yield_(BatchResponse::Batch(Ok((*data).clone()))).await;
                }
            }))
        }

        LogicalOp::Project { column_indices } => {
            // Optimization: fuse Project → SFrameSource into projected source read.
            if node.inputs.len() == 1 {
                if let LogicalOp::SFrameSource { path, column_types, begin_row, end_row, .. } =
                    &node.inputs[0].op
                {
                    // Only fuse if the source is not shared (broadcast already
                    // owns the source stream in that case).
                    let src_id = Arc::as_ptr(&node.inputs[0]) as usize;
                    let src_consumers = refcounts.get(&src_id).copied().unwrap_or(1);
                    if src_consumers == 1 {
                        return source::compile_sframe_source_projected(
                            path,
                            column_types,
                            *begin_row,
                            *end_row,
                            column_indices,
                        );
                    }
                }
            }
            let input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let indices = column_indices.clone();
            Ok(linear_operator(input, move |batch| {
                batch.select_columns(&indices)
            }))
        }

        LogicalOp::LogicalFilter => {
            let mut data_input =
                compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let mut mask_input =
                compile_memoized(&node.inputs[1], refcounts, rate_ids, memo)?;

            // Rate mismatch → rebatch both sides to a common batch size.
            let data_rate = rate_ids
                .get(&(Arc::as_ptr(&node.inputs[0]) as usize))
                .copied();
            let mask_rate = rate_ids
                .get(&(Arc::as_ptr(&node.inputs[1]) as usize))
                .copied();

            if data_rate != mask_rate {
                let target = sframe_config::global().source_batch_size();
                data_input = rebatch::rebatch(data_input, target);
                mask_input = rebatch::rebatch(mask_input, target);
            }

            Ok(BatchIterator::new(move |co: BatchCo| async move {
                let mut cmd = co.yield_(BatchResponse::Ready).await;
                loop {
                    match cmd {
                        BatchCommand::NextBatch => {
                            // Pull mask first — if all zeros, skip the data batch.
                            loop {
                                let mask_batch = mask_input.next_batch();
                                match mask_batch {
                                    None => return,
                                    Some(Err(e)) => {
                                        cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                                        break;
                                    }
                                    Some(Ok(mask)) => {
                                        if !mask.column(0).any_truthy() {
                                            // Mask is all zeros — skip the data batch.
                                            match data_input.skip_batch() {
                                                None => return,
                                                Some(()) => continue,
                                            }
                                        }
                                        // Mask has truthy values — pull data.
                                        match data_input.next_batch() {
                                            None => return,
                                            Some(Err(e)) => {
                                                cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                                                break;
                                            }
                                            Some(Ok(data)) => {
                                                match filter::logical_filter_batch(&data, &mask) {
                                                    Err(e) => {
                                                        cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                                                        break;
                                                    }
                                                    Ok(batch) if batch.num_rows() == 0 => continue,
                                                    Ok(batch) => {
                                                        cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        BatchCommand::SkipBatch => {
                            // Skip both inputs.
                            let mask_skip = mask_input.skip_batch();
                            let data_skip = data_input.skip_batch();
                            match (data_skip, mask_skip) {
                                (None, _) | (_, None) => return,
                                (Some(()), Some(())) => {
                                    cmd = co.yield_(BatchResponse::Skipped).await;
                                }
                            }
                        }
                        BatchCommand::Start => unreachable!(),
                    }
                }
            }))
        }

        LogicalOp::Filter { column, predicate } => {
            let input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let col = *column;
            let pred = predicate.clone();
            Ok(filter_operator(input, move |batch| {
                batch.filter_by_column(col, &*pred)
            }))
        }

        LogicalOp::Transform {
            input_column,
            func,
            output_type,
        } => {
            let input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let col = *input_column;
            let f = func.clone();
            let out_type = *output_type;
            Ok(linear_operator(input, move |batch| {
                transform::apply_transform(&batch, col, &*f, out_type)
            }))
        }

        LogicalOp::BinaryTransform {
            left_column,
            right_column,
            func,
            output_type,
        } => {
            let input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let l_col = *left_column;
            let r_col = *right_column;
            let f = func.clone();
            let out_type = *output_type;
            Ok(linear_operator(input, move |batch| {
                transform::apply_binary_transform(&batch, l_col, r_col, &*f, out_type)
            }))
        }

        LogicalOp::GeneralizedTransform {
            func,
            output_types,
        } => {
            let input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let f = func.clone();
            let out_types = output_types.clone();
            Ok(linear_operator(input, move |batch| {
                transform::apply_generalized_transform(&batch, &*f, &out_types)
            }))
        }

        LogicalOp::Append => {
            let mut left = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let mut right = compile_memoized(&node.inputs[1], refcounts, rate_ids, memo)?;
            Ok(BatchIterator::new(move |co: BatchCo| async move {
                let mut cmd = co.yield_(BatchResponse::Ready).await;
                // Drain left first
                loop {
                    match cmd {
                        BatchCommand::NextBatch => {
                            match left.next_batch() {
                                Some(result) => {
                                    cmd = co.yield_(BatchResponse::Batch(result)).await;
                                }
                                None => break,
                            }
                        }
                        BatchCommand::SkipBatch => {
                            match left.skip_batch() {
                                Some(()) => {
                                    cmd = co.yield_(BatchResponse::Skipped).await;
                                }
                                None => break,
                            }
                        }
                        BatchCommand::Start => unreachable!(),
                    }
                }
                // Then drain right
                loop {
                    match cmd {
                        BatchCommand::NextBatch => {
                            match right.next_batch() {
                                None => return,
                                Some(result) => {
                                    cmd = co.yield_(BatchResponse::Batch(result)).await;
                                }
                            }
                        }
                        BatchCommand::SkipBatch => {
                            match right.skip_batch() {
                                None => return,
                                Some(()) => {
                                    cmd = co.yield_(BatchResponse::Skipped).await;
                                }
                            }
                        }
                        BatchCommand::Start => unreachable!(),
                    }
                }
            }))
        }

        LogicalOp::Union => {
            let mut inputs: Vec<BatchIterator> = Vec::new();
            for input_node in &node.inputs {
                let input = compile_memoized(input_node, refcounts, rate_ids, memo)?;
                inputs.push(input);
            }
            Ok(BatchIterator::new(move |co: BatchCo| async move {
                let mut cmd = co.yield_(BatchResponse::Ready).await;
                for input in inputs.iter_mut() {
                    loop {
                        match cmd {
                            BatchCommand::NextBatch => {
                                match input.next_batch() {
                                    Some(result) => {
                                        cmd = co.yield_(BatchResponse::Batch(result)).await;
                                    }
                                    None => break,
                                }
                            }
                            BatchCommand::SkipBatch => {
                                match input.skip_batch() {
                                    Some(()) => {
                                        cmd = co.yield_(BatchResponse::Skipped).await;
                                    }
                                    None => break,
                                }
                            }
                            BatchCommand::Start => unreachable!(),
                        }
                    }
                }
            }))
        }

        LogicalOp::ColumnUnion => {
            let mut inputs: Vec<BatchIterator> = Vec::new();
            for input_node in &node.inputs {
                let mut s = compile_memoized(input_node, refcounts, rate_ids, memo)?;

                // Rate mismatch → rebatch to a common batch size.
                let input_rate = rate_ids
                    .get(&(Arc::as_ptr(input_node) as usize))
                    .copied();
                let first_rate = rate_ids
                    .get(&(Arc::as_ptr(&node.inputs[0]) as usize))
                    .copied();
                if input_rate != first_rate {
                    let target = sframe_config::global().source_batch_size();
                    s = rebatch::rebatch(s, target);
                }

                inputs.push(s);
            }
            // Also rebatch the first input if any mismatch was detected
            if inputs.len() > 1 {
                let first_rate = rate_ids
                    .get(&(Arc::as_ptr(&node.inputs[0]) as usize))
                    .copied();
                let any_mismatch = node.inputs[1..].iter().any(|inp| {
                    rate_ids.get(&(Arc::as_ptr(inp) as usize)).copied() != first_rate
                });
                if any_mismatch {
                    let target = sframe_config::global().source_batch_size();
                    let first = inputs.remove(0);
                    inputs.insert(0, rebatch::rebatch(first, target));
                }
            }

            if inputs.len() == 1 {
                return Ok(inputs.remove(0));
            }

            Ok(BatchIterator::new(move |co: BatchCo| async move {
                let mut cmd = co.yield_(BatchResponse::Ready).await;
                loop {
                    match cmd {
                        BatchCommand::NextBatch => {
                            // Pull from all inputs in lockstep and hconcat.
                            let mut batches: Vec<SFrameRows> = Vec::with_capacity(inputs.len());
                            let mut error = None;
                            let mut any_none = false;

                            for input in inputs.iter_mut() {
                                match input.next_batch() {
                                    None => {
                                        any_none = true;
                                        break;
                                    }
                                    Some(Err(e)) => {
                                        error = Some(e);
                                        break;
                                    }
                                    Some(Ok(batch)) => {
                                        batches.push(batch);
                                    }
                                }
                            }

                            if any_none {
                                return;
                            }
                            if let Some(e) = error {
                                cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                                continue;
                            }

                            // hconcat all batches
                            let mut combined = batches.remove(0);
                            let mut concat_error = None;
                            for batch in batches {
                                match combined.hconcat(&batch) {
                                    Ok(merged) => combined = merged,
                                    Err(e) => {
                                        concat_error = Some(e);
                                        break;
                                    }
                                }
                            }

                            if let Some(e) = concat_error {
                                cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                            } else {
                                cmd = co.yield_(BatchResponse::Batch(Ok(combined))).await;
                            }
                        }
                        BatchCommand::SkipBatch => {
                            // Skip all inputs.
                            let mut any_none = false;
                            for input in inputs.iter_mut() {
                                if input.skip_batch().is_none() {
                                    any_none = true;
                                    break;
                                }
                            }
                            if any_none {
                                return;
                            }
                            cmd = co.yield_(BatchResponse::Skipped).await;
                        }
                        BatchCommand::Start => unreachable!(),
                    }
                }
            }))
        }

        LogicalOp::Range { start, step, count } => range::compile_range(*start, *step, *count),

        LogicalOp::Reduce { aggregator } => {
            if let Some(par_plan) = reduce::try_extract_parallel_reduce_plan(node) {
                let result = reduce::execute_parallel_reduce(&par_plan)?;
                return Ok(BatchIterator::new(move |co: BatchCo| async move {
                    let cmd = co.yield_(BatchResponse::Ready).await;
                    if matches!(cmd, BatchCommand::NextBatch) {
                        co.yield_(BatchResponse::Batch(Ok(result))).await;
                    }
                }));
            }
            let mut input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let agg = aggregator.clone();
            let result = reduce::execute_reduce_iter(&mut input, agg)?;
            Ok(BatchIterator::new(move |co: BatchCo| async move {
                let cmd = co.yield_(BatchResponse::Ready).await;
                if matches!(cmd, BatchCommand::NextBatch) {
                    co.yield_(BatchResponse::Batch(Ok(result))).await;
                }
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Materialization helpers (synchronous)
// ---------------------------------------------------------------------------

/// Materialize a BatchIterator into a single SFrameRows batch.
pub fn materialize(iter: &mut BatchIterator) -> Result<SFrameRows> {
    let mut result: Option<SFrameRows> = None;

    while let Some(batch_result) = iter.next_batch() {
        let batch = batch_result?;
        match &mut result {
            None => result = Some(batch),
            Some(existing) => existing.append(&batch)?,
        }
    }

    Ok(result.unwrap_or_else(|| SFrameRows::empty(&[])))
}

/// Convenience wrapper for [`materialize`] that takes ownership of the iterator.
pub fn materialize_sync(mut iter: BatchIterator) -> Result<SFrameRows> {
    materialize(&mut iter)
}

/// Materialize at most `limit` rows from a BatchIterator, then stop pulling.
///
/// This enables efficient `head(n)` operations: only enough batches are
/// consumed to fill the requested row count. Remaining batches are never
/// read from the source.
pub fn materialize_head(iter: &mut BatchIterator, limit: usize) -> Result<SFrameRows> {
    let mut result: Option<SFrameRows> = None;
    let mut remaining = limit;

    while remaining > 0 {
        match iter.next_batch() {
            None => break,
            Some(Err(e)) => return Err(e),
            Some(Ok(batch)) => {
                let batch = if batch.num_rows() > remaining {
                    let indices: Vec<usize> = (0..remaining).collect();
                    batch.take(&indices)?
                } else {
                    batch
                };
                remaining -= batch.num_rows();
                match &mut result {
                    None => result = Some(batch),
                    Some(existing) => existing.append(&batch)?,
                }
            }
        }
    }

    Ok(result.unwrap_or_else(|| SFrameRows::empty(&[])))
}

/// Convenience wrapper for [`materialize_head`] that takes ownership of the iterator.
pub fn materialize_head_sync(mut iter: BatchIterator, limit: usize) -> Result<SFrameRows> {
    materialize_head(&mut iter, limit)
}

/// Materialize only the last `limit` rows from a BatchIterator.
///
/// Streams all batches but keeps a bounded ring buffer of the most recent
/// rows, so memory stays O(limit) regardless of total stream size.
pub fn materialize_tail(iter: &mut BatchIterator, limit: usize) -> Result<SFrameRows> {
    use std::collections::VecDeque;

    if limit == 0 {
        return Ok(SFrameRows::empty(&[]));
    }

    let mut ring: VecDeque<SFrameRows> = VecDeque::new();
    let mut buffered_rows: usize = 0;

    while let Some(batch_result) = iter.next_batch() {
        let batch = batch_result?;
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        buffered_rows += n;
        ring.push_back(batch);

        // Trim from the front while we have more than `limit` rows buffered.
        while buffered_rows > limit {
            let front_rows = ring.front().unwrap().num_rows();
            let excess = buffered_rows - limit;
            if excess >= front_rows {
                // Drop the entire front batch.
                ring.pop_front();
                buffered_rows -= front_rows;
            } else {
                // Trim the front batch: keep only the tail portion.
                let front = ring.pop_front().unwrap();
                let indices: Vec<usize> = (excess..front_rows).collect();
                let trimmed = front.take(&indices)?;
                buffered_rows -= excess;
                ring.push_front(trimmed);
                break;
            }
        }
    }

    // Concatenate the ring into a single batch.
    let mut result: Option<SFrameRows> = None;
    for batch in ring {
        match &mut result {
            None => result = Some(batch),
            Some(existing) => existing.append(&batch)?,
        }
    }
    Ok(result.unwrap_or_else(|| SFrameRows::empty(&[])))
}

/// Convenience wrapper for [`materialize_tail`] that takes ownership of the iterator.
pub fn materialize_tail_sync(mut iter: BatchIterator, limit: usize) -> Result<SFrameRows> {
    materialize_tail(&mut iter, limit)
}

/// Consume a BatchIterator batch-by-batch, calling `callback` for
/// each batch. This avoids collecting the entire result into memory.
pub fn for_each_batch_sync<F>(mut iter: BatchIterator, mut callback: F) -> Result<()>
where
    F: FnMut(SFrameRows) -> Result<()>,
{
    while let Some(batch_result) = iter.next_batch() {
        let batch = batch_result?;
        callback(batch)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sframe_storage::sframe_reader::SFrameReader;
    use sframe_types::flex_type::{FlexType, FlexTypeEnum};

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{manifest}/../../samples")
    }

    #[test]
    fn test_materialized_source() {
        let rows = vec![
            vec![FlexType::Integer(1), FlexType::String("a".into())],
            vec![FlexType::Integer(2), FlexType::String("b".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let node = PlannerNode::materialized(batch);
        let mut iter = compile(&node).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.row(0), rows[0]);
        assert_eq!(result.row(1), rows[1]);
    }

    #[test]
    fn test_project() {
        let rows = vec![
            vec![
                FlexType::Integer(1),
                FlexType::Float(1.5),
                FlexType::String("a".into()),
            ],
            vec![
                FlexType::Integer(2),
                FlexType::Float(2.5),
                FlexType::String("b".into()),
            ],
        ];
        let dtypes = [
            FlexTypeEnum::Integer,
            FlexTypeEnum::Float,
            FlexTypeEnum::String,
        ];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let source = PlannerNode::materialized(batch);
        let projected = PlannerNode::project(source, vec![0, 2]);
        let mut iter = compile(&projected).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(
            result.row(0),
            vec![FlexType::Integer(1), FlexType::String("a".into())]
        );
    }

    #[test]
    fn test_filter() {
        let rows = vec![
            vec![FlexType::Integer(1)],
            vec![FlexType::Integer(2)],
            vec![FlexType::Integer(3)],
            vec![FlexType::Integer(4)],
        ];
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let source = PlannerNode::materialized(batch);
        let filtered = PlannerNode::filter(
            source,
            0,
            Arc::new(|v| matches!(v, FlexType::Integer(i) if *i > 2)),
        );
        let mut iter = compile(&filtered).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.row(0), vec![FlexType::Integer(3)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(4)]);
    }

    #[test]
    fn test_transform() {
        let rows = vec![
            vec![FlexType::Integer(10)],
            vec![FlexType::Integer(20)],
        ];
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let source = PlannerNode::materialized(batch);
        let transformed = PlannerNode::transform(
            source,
            0,
            Arc::new(|v| match v {
                FlexType::Integer(i) => FlexType::Float(*i as f64 * 1.5),
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Float,
        );
        let mut iter = compile(&transformed).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.row(0), vec![FlexType::Float(15.0)]);
        assert_eq!(result.row(1), vec![FlexType::Float(30.0)]);
    }

    #[test]
    fn test_append() {
        let rows1 = vec![vec![FlexType::Integer(1)]];
        let rows2 = vec![vec![FlexType::Integer(2)]];
        let dtypes = [FlexTypeEnum::Integer];

        let src1 = PlannerNode::materialized(SFrameRows::from_rows(&rows1, &dtypes).unwrap());
        let src2 = PlannerNode::materialized(SFrameRows::from_rows(&rows2, &dtypes).unwrap());
        let appended = PlannerNode::append(src1, src2);

        let mut iter = compile(&appended).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.row(0), vec![FlexType::Integer(1)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(2)]);
    }

    #[test]
    fn test_range() {
        let node = PlannerNode::range(0, 2, 5);
        let mut iter = compile(&node).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_rows(), 5);
        assert_eq!(result.row(0), vec![FlexType::Integer(0)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(2)]);
        assert_eq!(result.row(2), vec![FlexType::Integer(4)]);
        assert_eq!(result.row(3), vec![FlexType::Integer(6)]);
        assert_eq!(result.row(4), vec![FlexType::Integer(8)]);
    }

    #[test]
    fn test_sframe_source() {
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

        let node = PlannerNode::sframe_source(&path, col_names, col_types, num_rows);
        let mut iter = compile(&node).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_rows(), 11536);
        assert_eq!(result.num_columns(), 12);
    }

    #[test]
    fn test_sframe_source_filter_project() {
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
        let filtered = PlannerNode::filter(
            source,
            9, // stars column
            Arc::new(|v| matches!(v, FlexType::Float(f) if *f >= 4.5)),
        );
        let projected = PlannerNode::project(filtered, vec![0, 2]);

        let mut iter = compile(&projected).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_rows(), 3020);
        assert_eq!(result.num_columns(), 2);

        for i in 0..result.num_rows() {
            let row = result.row(i);
            assert!(matches!(&row[0], FlexType::String(_)));
            assert!(matches!(&row[1], FlexType::String(_)));
        }
    }

    #[test]
    fn test_materialize_head_partial() {
        let node = PlannerNode::range(0, 1, 10);
        let mut iter = compile(&node).unwrap();
        let result = materialize_head(&mut iter, 3).unwrap();

        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.row(0), vec![FlexType::Integer(0)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(1)]);
        assert_eq!(result.row(2), vec![FlexType::Integer(2)]);
    }

    #[test]
    fn test_materialize_head_exceeds_total() {
        let node = PlannerNode::range(0, 1, 5);
        let mut iter = compile(&node).unwrap();
        let result = materialize_head(&mut iter, 100).unwrap();

        assert_eq!(result.num_rows(), 5);
    }

    #[test]
    fn test_materialize_head_zero() {
        let node = PlannerNode::range(0, 1, 10);
        let mut iter = compile(&node).unwrap();
        let result = materialize_head(&mut iter, 0).unwrap();

        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_for_each_batch_sync() {
        let node = PlannerNode::range(0, 1, 10);
        let iter = compile(&node).unwrap();

        let mut total_rows = 0usize;
        for_each_batch_sync(iter, |batch| {
            total_rows += batch.num_rows();
            Ok(())
        })
        .unwrap();

        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_chained_operations() {
        let source = PlannerNode::range(0, 1, 10);
        let filtered = PlannerNode::filter(
            source,
            0,
            Arc::new(|v| matches!(v, FlexType::Integer(i) if *i > 5)),
        );
        let transformed = PlannerNode::transform(
            filtered,
            0,
            Arc::new(|v| match v {
                FlexType::Integer(i) => FlexType::Integer(*i * 2),
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Integer,
        );
        let mut iter = compile(&transformed).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.row(0), vec![FlexType::Integer(12)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(14)]);
        assert_eq!(result.row(2), vec![FlexType::Integer(16)]);
        assert_eq!(result.row(3), vec![FlexType::Integer(18)]);
    }

    // --- Fan-out / broadcast tests ---

    #[test]
    fn test_shared_source_broadcast() {
        // Same materialized source feeds two branches → Append.
        // Both branches should see the same data, source compiled once.
        let rows = vec![
            vec![FlexType::Integer(1)],
            vec![FlexType::Integer(2)],
            vec![FlexType::Integer(3)],
        ];
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let source = PlannerNode::materialized(batch);
        // source is shared (two consumers)
        let left = PlannerNode::transform(
            source.clone(),
            0,
            Arc::new(|v| match v {
                FlexType::Integer(i) => FlexType::Integer(*i * 10),
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Integer,
        );
        let right = PlannerNode::transform(
            source, // same Arc
            0,
            Arc::new(|v| match v {
                FlexType::Integer(i) => FlexType::Integer(*i * 100),
                _ => FlexType::Undefined,
            }),
            FlexTypeEnum::Integer,
        );
        let appended = PlannerNode::append(left, right);

        let mut iter = compile(&appended).unwrap();
        let result = materialize(&mut iter).unwrap();

        assert_eq!(result.num_rows(), 6);
        // Left branch: 10, 20, 30
        assert_eq!(result.row(0), vec![FlexType::Integer(10)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(20)]);
        assert_eq!(result.row(2), vec![FlexType::Integer(30)]);
        // Right branch: 100, 200, 300
        assert_eq!(result.row(3), vec![FlexType::Integer(100)]);
        assert_eq!(result.row(4), vec![FlexType::Integer(200)]);
        assert_eq!(result.row(5), vec![FlexType::Integer(300)]);
    }

    #[test]
    fn test_logical_filter_shared_source() {
        // Shared source → data path (identity) + mask path (transform to bool).
        // LogicalFilter should zip correctly with broadcast alignment.
        let rows = vec![
            vec![FlexType::Integer(1)],
            vec![FlexType::Integer(2)],
            vec![FlexType::Integer(3)],
            vec![FlexType::Integer(4)],
        ];
        let dtypes = [FlexTypeEnum::Integer];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let source = PlannerNode::materialized(batch);
        let mask = PlannerNode::transform(
            source.clone(),
            0,
            Arc::new(|v| match v {
                FlexType::Integer(i) if *i % 2 == 0 => FlexType::Integer(1),
                _ => FlexType::Integer(0),
            }),
            FlexTypeEnum::Integer,
        );
        let filtered = PlannerNode::logical_filter(source, mask);

        let mut iter = compile(&filtered).unwrap();
        let result = materialize(&mut iter).unwrap();

        // Only even values pass: 2, 4
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.row(0), vec![FlexType::Integer(2)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(4)]);
    }

    // --- Rebatch tests ---

    #[test]
    fn test_rebatch_normalizes_batch_sizes() {
        // Create a BatchIterator with variable batch sizes, rebatch to target=3.
        let b1 = SFrameRows::from_rows(
            &[vec![FlexType::Integer(1)]],
            &[FlexTypeEnum::Integer],
        )
        .unwrap();
        let b2 = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(2)],
                vec![FlexType::Integer(3)],
                vec![FlexType::Integer(4)],
                vec![FlexType::Integer(5)],
            ],
            &[FlexTypeEnum::Integer],
        )
        .unwrap();
        let b3 = SFrameRows::from_rows(
            &[
                vec![FlexType::Integer(6)],
                vec![FlexType::Integer(7)],
            ],
            &[FlexTypeEnum::Integer],
        )
        .unwrap();

        let batches = vec![b1, b2, b3];
        let input = BatchIterator::new(move |co: BatchCo| async move {
            let mut cmd = co.yield_(BatchResponse::Ready).await;
            for batch in batches {
                match cmd {
                    BatchCommand::NextBatch => {
                        cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await;
                    }
                    BatchCommand::SkipBatch => {
                        cmd = co.yield_(BatchResponse::Skipped).await;
                    }
                    _ => break,
                }
            }
        });
        let mut rebatched = rebatch::rebatch(input, 3);
        let result = materialize(&mut rebatched).unwrap();

        // 7 total rows, rebatched to target=3 → batches of 3, 3, 1
        assert_eq!(result.num_rows(), 7);
        for i in 0..7 {
            assert_eq!(
                result.row(i),
                vec![FlexType::Integer((i + 1) as i64)]
            );
        }
    }

    #[test]
    fn test_rebatch_batch_boundaries() {
        // Verify exact batch sizes: 5 rows, target=2 → [2, 2, 1]
        let rows: Vec<Vec<FlexType>> = (0..5).map(|i| vec![FlexType::Integer(i)]).collect();
        let batch =
            SFrameRows::from_rows(&rows, &[FlexTypeEnum::Integer]).unwrap();
        let input = BatchIterator::new(move |co: BatchCo| async move {
            let cmd = co.yield_(BatchResponse::Ready).await;
            if matches!(cmd, BatchCommand::NextBatch) {
                co.yield_(BatchResponse::Batch(Ok(batch))).await;
            }
        });

        let mut rebatched = rebatch::rebatch(input, 2);

        let mut batch_sizes = vec![];
        while let Some(Ok(b)) = rebatched.next_batch() {
            batch_sizes.push(b.num_rows());
        }
        assert_eq!(batch_sizes, vec![2, 2, 1]);
    }
}
