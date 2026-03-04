//! Physical execution engine.
//!
//! Compiles a logical plan DAG into async streams of `SFrameRows` batches.
//! Each operator wraps its input stream(s) and produces output batches.
//!
//! Shared subexpressions (same `Arc<PlannerNode>`) are detected and compiled
//! once, with a broadcast adapter fanning the output to all consumers.
//! Rate analysis inserts rebatch adapters at lockstep multi-input operators
//! when input batch boundaries may be misaligned.

mod broadcast;
mod filter;
mod parallel;
mod range;
mod rebatch;
mod reduce;
mod source;
mod transform;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::stream::{self, Stream, StreamExt};

use sframe_types::error::{Result, SFrameError};

use crate::batch::SFrameRows;
use crate::optimizer;
use crate::planner::{LogicalOp, OperatorRate, PlannerNode};

use broadcast::BroadcastState;

/// A stream of SFrameRows batches.
pub type BatchStream = Pin<Box<dyn Stream<Item = Result<SFrameRows>> + Send>>;

/// Default broadcast buffer capacity (batches per consumer channel).
const BROADCAST_BUFFER: usize = 4;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compile a logical plan node into a BatchStream.
///
/// Tries data-parallel execution first (splitting input rows across
/// worker threads). Falls back to single-threaded compilation when
/// the plan isn't parallel-sliceable or the data is too small.
pub fn compile(node: &Arc<PlannerNode>) -> Result<BatchStream> {
    let node = optimizer::optimize(node);

    // Try data-parallel execution
    if let Some(total_rows) = parallel::parallel_slice_row_count(&node) {
        if let Some(stream) = parallel::compile_parallel(&node, total_rows) {
            return Ok(stream);
        }
    }

    compile_single_threaded(&node)
}

/// Single-threaded compilation: fan-out detection (broadcast for shared
/// nodes) and rate-mismatch detection (rebatch at lockstep operators).
pub(super) fn compile_single_threaded(node: &Arc<PlannerNode>) -> Result<BatchStream> {
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
) -> Result<BatchStream> {
    let id = Arc::as_ptr(node) as usize;

    // If this node was already compiled (shared), hand out a subscriber.
    if let Some(state) = memo.get_mut(&id) {
        return Ok(state.subscribe());
    }

    // Compile the node itself.
    let stream = compile_node(node, refcounts, rate_ids, memo)?;

    // If the node has multiple downstream consumers, wrap in broadcast.
    let consumers = refcounts.get(&id).copied().unwrap_or(1);
    if consumers > 1 {
        let mut bc = BroadcastState::new(stream, consumers, BROADCAST_BUFFER);
        let subscriber = bc.subscribe();
        memo.insert(id, bc);
        return Ok(subscriber);
    }

    Ok(stream)
}

/// Compile a single node (dispatches on LogicalOp).
fn compile_node(
    node: &Arc<PlannerNode>,
    refcounts: &HashMap<usize, usize>,
    rate_ids: &HashMap<usize, usize>,
    memo: &mut HashMap<usize, BroadcastState>,
) -> Result<BatchStream> {
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

        LogicalOp::MaterializedSource { data } => {
            let data = data.clone();
            Ok(Box::pin(stream::once(async move { Ok((*data).clone()) })))
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
            Ok(Box::pin(input.map(move |batch_result| {
                batch_result.and_then(|batch| batch.select_columns(&indices))
            })))
        }

        LogicalOp::LogicalFilter => {
            let mut data_stream =
                compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let mut mask_stream =
                compile_memoized(&node.inputs[1], refcounts, rate_ids, memo)?;

            // Rate mismatch → rebatch both sides to a common batch size.
            let data_rate = rate_ids
                .get(&(Arc::as_ptr(&node.inputs[0]) as usize))
                .copied();
            let mask_rate = rate_ids
                .get(&(Arc::as_ptr(&node.inputs[1]) as usize))
                .copied();

            if data_rate != mask_rate {
                let target = sframe_config::global().source_batch_size;
                data_stream = rebatch::rebatch(data_stream, target);
                mask_stream = rebatch::rebatch(mask_stream, target);
            }

            Ok(Box::pin(
                data_stream
                    .zip(mask_stream)
                    .filter_map(|(data_result, mask_result)| async {
                        match (data_result, mask_result) {
                            (Err(e), _) | (_, Err(e)) => Some(Err(e)),
                            (Ok(data), Ok(mask)) => {
                                match filter::logical_filter_batch(&data, &mask) {
                                    Err(e) => Some(Err(e)),
                                    Ok(batch) if batch.num_rows() == 0 => None,
                                    Ok(batch) => Some(Ok(batch)),
                                }
                            }
                        }
                    }),
            ))
        }

        LogicalOp::Filter { column, predicate } => {
            let input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let col = *column;
            let pred = predicate.clone();
            Ok(Box::pin(input.filter_map(move |batch_result| {
                let pred = pred.clone();
                async move {
                    match batch_result {
                        Err(e) => Some(Err(e)),
                        Ok(batch) => match batch.filter_by_column(col, &*pred) {
                            Err(e) => Some(Err(e)),
                            Ok(filtered) => {
                                if filtered.num_rows() == 0 {
                                    None // skip empty batches
                                } else {
                                    Some(Ok(filtered))
                                }
                            }
                        },
                    }
                }
            })))
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
            Ok(Box::pin(input.map(move |batch_result| {
                batch_result
                    .and_then(|batch| transform::apply_transform(&batch, col, &*f, out_type))
            })))
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
            Ok(Box::pin(input.map(move |batch_result| {
                batch_result.and_then(|batch| {
                    transform::apply_binary_transform(&batch, l_col, r_col, &*f, out_type)
                })
            })))
        }

        LogicalOp::GeneralizedTransform {
            func,
            output_types,
        } => {
            let input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let f = func.clone();
            let out_types = output_types.clone();
            Ok(Box::pin(input.map(move |batch_result| {
                batch_result.and_then(|batch| {
                    transform::apply_generalized_transform(&batch, &*f, &out_types)
                })
            })))
        }

        LogicalOp::Append => {
            let left = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let right = compile_memoized(&node.inputs[1], refcounts, rate_ids, memo)?;
            Ok(Box::pin(left.chain(right)))
        }

        LogicalOp::Union => {
            let mut combined: BatchStream = Box::pin(stream::empty());
            for input_node in &node.inputs {
                let input = compile_memoized(input_node, refcounts, rate_ids, memo)?;
                combined = Box::pin(combined.chain(input));
            }
            Ok(combined)
        }

        LogicalOp::Range { start, step, count } => range::compile_range(*start, *step, *count),

        LogicalOp::Reduce { aggregator } => {
            if let Some(par_plan) = reduce::try_extract_parallel_reduce_plan(node) {
                return Ok(Box::pin(stream::once(async move {
                    reduce::execute_parallel_reduce(&par_plan)
                })));
            }
            let input = compile_memoized(&node.inputs[0], refcounts, rate_ids, memo)?;
            let agg = aggregator.clone();
            Ok(Box::pin(stream::once(async move {
                reduce::execute_reduce(input, agg).await
            })))
        }
    }
}

// ---------------------------------------------------------------------------
// Materialization helpers
// ---------------------------------------------------------------------------

/// Helper: materialize a stream into a single SFrameRows batch.
pub async fn materialize(stream: BatchStream) -> Result<SFrameRows> {
    let mut stream = stream;
    let mut result: Option<SFrameRows> = None;

    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        match &mut result {
            None => result = Some(batch),
            Some(existing) => existing.append(&batch)?,
        }
    }

    Ok(result.unwrap_or_else(|| SFrameRows::empty(&[])))
}

/// Helper: materialize synchronously using a tokio runtime.
pub fn materialize_sync(stream: BatchStream) -> Result<SFrameRows> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| SFrameError::Format(format!("Failed to create tokio runtime: {}", e)))?;
    rt.block_on(materialize(stream))
}

/// Materialize at most `limit` rows from a stream, then stop pulling.
///
/// This enables efficient `head(n)` operations: only enough batches are
/// consumed to fill the requested row count. Remaining batches are never
/// read from the source.
pub async fn materialize_head(mut stream: BatchStream, limit: usize) -> Result<SFrameRows> {
    let mut result: Option<SFrameRows> = None;
    let mut remaining = limit;

    while remaining > 0 {
        match stream.next().await {
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

/// Synchronous version of [`materialize_head`].
pub fn materialize_head_sync(stream: BatchStream, limit: usize) -> Result<SFrameRows> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| SFrameError::Format(format!("Failed to create tokio runtime: {}", e)))?;
    rt.block_on(materialize_head(stream, limit))
}

/// Consume a stream batch-by-batch synchronously, calling `callback` for
/// each batch. This avoids collecting the entire stream into memory.
pub fn for_each_batch_sync<F>(stream: BatchStream, mut callback: F) -> Result<()>
where
    F: FnMut(SFrameRows) -> Result<()>,
{
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| SFrameError::Format(format!("Failed to create tokio runtime: {}", e)))?;
    rt.block_on(async move {
        let mut stream = stream;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            callback(batch)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sframe_storage::sframe_reader::SFrameReader;
    use sframe_types::flex_type::{FlexType, FlexTypeEnum};

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../samples", manifest)
    }

    #[tokio::test]
    async fn test_materialized_source() {
        let rows = vec![
            vec![FlexType::Integer(1), FlexType::String("a".into())],
            vec![FlexType::Integer(2), FlexType::String("b".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let node = PlannerNode::materialized(batch);
        let stream = compile(&node).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.row(0), rows[0]);
        assert_eq!(result.row(1), rows[1]);
    }

    #[tokio::test]
    async fn test_project() {
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
        let stream = compile(&projected).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(
            result.row(0),
            vec![FlexType::Integer(1), FlexType::String("a".into())]
        );
    }

    #[tokio::test]
    async fn test_filter() {
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
        let stream = compile(&filtered).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.row(0), vec![FlexType::Integer(3)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(4)]);
    }

    #[tokio::test]
    async fn test_transform() {
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
        let stream = compile(&transformed).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.row(0), vec![FlexType::Float(15.0)]);
        assert_eq!(result.row(1), vec![FlexType::Float(30.0)]);
    }

    #[tokio::test]
    async fn test_append() {
        let rows1 = vec![vec![FlexType::Integer(1)]];
        let rows2 = vec![vec![FlexType::Integer(2)]];
        let dtypes = [FlexTypeEnum::Integer];

        let src1 = PlannerNode::materialized(SFrameRows::from_rows(&rows1, &dtypes).unwrap());
        let src2 = PlannerNode::materialized(SFrameRows::from_rows(&rows2, &dtypes).unwrap());
        let appended = PlannerNode::append(src1, src2);

        let stream = compile(&appended).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.row(0), vec![FlexType::Integer(1)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(2)]);
    }

    #[tokio::test]
    async fn test_range() {
        let node = PlannerNode::range(0, 2, 5);
        let stream = compile(&node).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_rows(), 5);
        assert_eq!(result.row(0), vec![FlexType::Integer(0)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(2)]);
        assert_eq!(result.row(2), vec![FlexType::Integer(4)]);
        assert_eq!(result.row(3), vec![FlexType::Integer(6)]);
        assert_eq!(result.row(4), vec![FlexType::Integer(8)]);
    }

    #[tokio::test]
    async fn test_sframe_source() {
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
        let stream = compile(&node).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_rows(), 11536);
        assert_eq!(result.num_columns(), 12);
    }

    #[tokio::test]
    async fn test_sframe_source_filter_project() {
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

        let stream = compile(&projected).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_rows(), 3020);
        assert_eq!(result.num_columns(), 2);

        for i in 0..result.num_rows() {
            let row = result.row(i);
            assert!(matches!(&row[0], FlexType::String(_)));
            assert!(matches!(&row[1], FlexType::String(_)));
        }
    }

    #[tokio::test]
    async fn test_materialize_head_partial() {
        let node = PlannerNode::range(0, 1, 10);
        let stream = compile(&node).unwrap();
        let result = materialize_head(stream, 3).await.unwrap();

        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.row(0), vec![FlexType::Integer(0)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(1)]);
        assert_eq!(result.row(2), vec![FlexType::Integer(2)]);
    }

    #[tokio::test]
    async fn test_materialize_head_exceeds_total() {
        let node = PlannerNode::range(0, 1, 5);
        let stream = compile(&node).unwrap();
        let result = materialize_head(stream, 100).await.unwrap();

        assert_eq!(result.num_rows(), 5);
    }

    #[tokio::test]
    async fn test_materialize_head_zero() {
        let node = PlannerNode::range(0, 1, 10);
        let stream = compile(&node).unwrap();
        let result = materialize_head(stream, 0).await.unwrap();

        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_for_each_batch_sync() {
        let node = PlannerNode::range(0, 1, 10);
        let stream = compile(&node).unwrap();

        let mut total_rows = 0usize;
        for_each_batch_sync(stream, |batch| {
            total_rows += batch.num_rows();
            Ok(())
        })
        .unwrap();

        assert_eq!(total_rows, 10);
    }

    #[tokio::test]
    async fn test_chained_operations() {
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
        let stream = compile(&transformed).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.row(0), vec![FlexType::Integer(12)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(14)]);
        assert_eq!(result.row(2), vec![FlexType::Integer(16)]);
        assert_eq!(result.row(3), vec![FlexType::Integer(18)]);
    }

    // --- Fan-out / broadcast tests ---

    #[tokio::test]
    async fn test_shared_source_broadcast() {
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

        let stream = compile(&appended).unwrap();
        let result = materialize(stream).await.unwrap();

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

    #[tokio::test]
    async fn test_logical_filter_shared_source() {
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

        let stream = compile(&filtered).unwrap();
        let result = materialize(stream).await.unwrap();

        // Only even values pass: 2, 4
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.row(0), vec![FlexType::Integer(2)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(4)]);
    }

    // --- Rebatch tests ---

    #[tokio::test]
    async fn test_rebatch_normalizes_batch_sizes() {
        // Create a stream with variable batch sizes, rebatch to target=3.
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

        let input: BatchStream = Box::pin(stream::iter(vec![Ok(b1), Ok(b2), Ok(b3)]));
        let rebatched = rebatch::rebatch(input, 3);
        let result = materialize(rebatched).await.unwrap();

        // 7 total rows, rebatched to target=3 → batches of 3, 3, 1
        assert_eq!(result.num_rows(), 7);
        for i in 0..7 {
            assert_eq!(
                result.row(i),
                vec![FlexType::Integer((i + 1) as i64)]
            );
        }
    }

    #[tokio::test]
    async fn test_rebatch_batch_boundaries() {
        // Verify exact batch sizes: 5 rows, target=2 → [2, 2, 1]
        let rows: Vec<Vec<FlexType>> = (0..5).map(|i| vec![FlexType::Integer(i)]).collect();
        let batch =
            SFrameRows::from_rows(&rows, &[FlexTypeEnum::Integer]).unwrap();
        let input: BatchStream = Box::pin(stream::once(async { Ok(batch) }));

        let rebatched = rebatch::rebatch(input, 2);

        let mut batch_sizes = vec![];
        let mut rebatched = rebatched;
        while let Some(Ok(b)) = rebatched.next().await {
            batch_sizes.push(b.num_rows());
        }
        assert_eq!(batch_sizes, vec![2, 2, 1]);
    }
}
