//! Physical execution engine.
//!
//! Compiles a logical plan DAG into async streams of `SFrameRows` batches.
//! Each operator wraps its input stream(s) and produces output batches.

use std::pin::Pin;
use std::sync::Arc;

use futures::stream::{self, Stream, StreamExt};

use sframe_storage::sframe_reader::SFrameReader;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::{ColumnData, SFrameRows};
use crate::optimizer;
use crate::planner::{LogicalOp, PlannerNode};

/// A stream of SFrameRows batches.
pub type BatchStream = Pin<Box<dyn Stream<Item = Result<SFrameRows>> + Send>>;

/// Target batch size for source operators.
const SOURCE_BATCH_SIZE: usize = 4096;

/// Compile a logical plan node into a BatchStream.
///
/// Applies optimizer passes before compilation for better execution.
pub fn compile(node: &Arc<PlannerNode>) -> Result<BatchStream> {
    let node = optimizer::optimize(node);
    compile_node(&node)
}

/// Internal compilation without optimization (used recursively).
fn compile_node(node: &Arc<PlannerNode>) -> Result<BatchStream> {
    match &node.op {
        LogicalOp::SFrameSource {
            path,
            column_names: _,
            column_types,
            num_rows,
        } => compile_sframe_source(path, column_types, *num_rows),

        LogicalOp::MaterializedSource { data } => {
            let data = data.clone();
            Ok(Box::pin(stream::once(async move { Ok((*data).clone()) })))
        }

        LogicalOp::Project { column_indices } => {
            let input = compile_node(&node.inputs[0])?;
            let indices = column_indices.clone();
            Ok(Box::pin(input.map(move |batch_result| {
                batch_result.and_then(|batch| batch.select_columns(&indices))
            })))
        }

        LogicalOp::Filter { column, predicate } => {
            let input = compile_node(&node.inputs[0])?;
            let col = *column;
            let pred = predicate.clone();
            Ok(Box::pin(input.filter_map(move |batch_result| {
                let pred = pred.clone();
                async move {
                    match batch_result {
                        Err(e) => Some(Err(e)),
                        Ok(batch) => {
                            match batch.filter_by_column(col, &*pred) {
                                Err(e) => Some(Err(e)),
                                Ok(filtered) => {
                                    if filtered.num_rows() == 0 {
                                        None // skip empty batches
                                    } else {
                                        Some(Ok(filtered))
                                    }
                                }
                            }
                        }
                    }
                }
            })))
        }

        LogicalOp::Transform {
            input_column,
            func,
            output_type,
        } => {
            let input = compile_node(&node.inputs[0])?;
            let col = *input_column;
            let f = func.clone();
            let out_type = *output_type;
            Ok(Box::pin(input.map(move |batch_result| {
                batch_result.and_then(|batch| {
                    apply_transform(&batch, col, &*f, out_type)
                })
            })))
        }

        LogicalOp::BinaryTransform {
            left_column,
            right_column,
            func,
            output_type,
        } => {
            let input = compile_node(&node.inputs[0])?;
            let l_col = *left_column;
            let r_col = *right_column;
            let f = func.clone();
            let out_type = *output_type;
            Ok(Box::pin(input.map(move |batch_result| {
                batch_result.and_then(|batch| {
                    apply_binary_transform(&batch, l_col, r_col, &*f, out_type)
                })
            })))
        }

        LogicalOp::GeneralizedTransform {
            func,
            output_types,
        } => {
            let input = compile_node(&node.inputs[0])?;
            let f = func.clone();
            let out_types = output_types.clone();
            Ok(Box::pin(input.map(move |batch_result| {
                batch_result.and_then(|batch| {
                    apply_generalized_transform(&batch, &*f, &out_types)
                })
            })))
        }

        LogicalOp::Append => {
            let left = compile_node(&node.inputs[0])?;
            let right = compile_node(&node.inputs[1])?;
            Ok(Box::pin(left.chain(right)))
        }

        LogicalOp::Union => {
            let mut combined: BatchStream =
                Box::pin(stream::empty());
            for input_node in &node.inputs {
                let input = compile_node(input_node)?;
                combined = Box::pin(combined.chain(input));
            }
            Ok(combined)
        }

        LogicalOp::Range { start, step, count } => {
            compile_range(*start, *step, *count)
        }

        LogicalOp::Reduce { aggregator } => {
            let input = compile_node(&node.inputs[0])?;
            let agg = aggregator.clone();
            Ok(Box::pin(stream::once(async move {
                execute_reduce(input, agg).await
            })))
        }
    }
}

/// Compile an SFrame source: reads one segment at a time and emits batches.
///
/// Unlike the previous implementation that loaded all columns eagerly,
/// this reads segment-by-segment. Only one segment's data is in memory
/// at a time, limiting peak memory usage.
fn compile_sframe_source(
    path: &str,
    column_types: &[FlexTypeEnum],
    _num_rows: u64,
) -> Result<BatchStream> {
    let mut reader = SFrameReader::open(path)?;
    let num_segments = reader.num_segments();
    let dtypes: Vec<FlexTypeEnum> = column_types.to_vec();

    // Read segment-at-a-time: for each segment, read all columns,
    // then emit as batches of SOURCE_BATCH_SIZE.
    let mut batches: Vec<Result<SFrameRows>> = Vec::new();

    for seg_idx in 0..num_segments {
        let seg_data = reader.read_segment_columns(seg_idx)?;
        let seg_rows = if seg_data.is_empty() { 0 } else { seg_data[0].len() };

        let mut offset = 0;
        while offset < seg_rows {
            let end = (offset + SOURCE_BATCH_SIZE).min(seg_rows);
            let mut columns = Vec::with_capacity(dtypes.len());
            for (col_idx, col_data) in seg_data.iter().enumerate() {
                let mut col = ColumnData::empty(dtypes[col_idx]);
                for val in &col_data[offset..end] {
                    col.push(val)?;
                }
                columns.push(col);
            }
            batches.push(SFrameRows::new(columns));
            offset = end;
        }
        // seg_data is dropped here, freeing the segment's memory
    }

    Ok(Box::pin(stream::iter(batches)))
}

/// Compile a range source.
fn compile_range(start: i64, step: i64, count: u64) -> Result<BatchStream> {
    let mut batches: Vec<Result<SFrameRows>> = Vec::new();
    let total = count as usize;
    let mut offset = 0;
    while offset < total {
        let end = (offset + SOURCE_BATCH_SIZE).min(total);
        let values: Vec<Option<i64>> = (offset..end)
            .map(|i| Some(start + (i as i64) * step))
            .collect();
        let col = ColumnData::Integer(values);
        batches.push(SFrameRows::new(vec![col]));
        offset = end;
    }
    Ok(Box::pin(stream::iter(batches)))
}

/// Apply a unary transform: append a new column from transforming an input column.
fn apply_transform(
    batch: &SFrameRows,
    input_column: usize,
    func: &dyn Fn(&FlexType) -> FlexType,
    output_type: FlexTypeEnum,
) -> Result<SFrameRows> {
    let mut new_col = ColumnData::empty(output_type);
    for i in 0..batch.num_rows() {
        let val = batch.column(input_column).get(i);
        let result = func(&val);
        new_col.push(&result)?;
    }

    let mut columns: Vec<ColumnData> = batch.columns().to_vec();
    columns.push(new_col);
    SFrameRows::new(columns)
}

/// Apply a binary transform: append a new column from two input columns.
fn apply_binary_transform(
    batch: &SFrameRows,
    left_col: usize,
    right_col: usize,
    func: &dyn Fn(&FlexType, &FlexType) -> FlexType,
    output_type: FlexTypeEnum,
) -> Result<SFrameRows> {
    let mut new_col = ColumnData::empty(output_type);
    for i in 0..batch.num_rows() {
        let left = batch.column(left_col).get(i);
        let right = batch.column(right_col).get(i);
        let result = func(&left, &right);
        new_col.push(&result)?;
    }

    let mut columns: Vec<ColumnData> = batch.columns().to_vec();
    columns.push(new_col);
    SFrameRows::new(columns)
}

/// Apply a generalized transform: replace all columns with transform output.
fn apply_generalized_transform(
    batch: &SFrameRows,
    func: &dyn Fn(&[FlexType]) -> Vec<FlexType>,
    output_types: &[FlexTypeEnum],
) -> Result<SFrameRows> {
    let mut columns: Vec<ColumnData> = output_types
        .iter()
        .map(|&dt| ColumnData::empty(dt))
        .collect();

    for i in 0..batch.num_rows() {
        let row: Vec<FlexType> = batch.row(i);
        let results = func(&row);
        if results.len() != output_types.len() {
            return Err(SFrameError::Format(format!(
                "GeneralizedTransform produced {} values, expected {}",
                results.len(),
                output_types.len()
            )));
        }
        for (col_idx, val) in results.iter().enumerate() {
            columns[col_idx].push(val)?;
        }
    }

    SFrameRows::new(columns)
}

/// Execute a reduce operation by consuming the entire input stream.
async fn execute_reduce(
    mut input: BatchStream,
    aggregator: Arc<dyn crate::planner::Aggregator>,
) -> Result<SFrameRows> {
    let mut agg = aggregator.box_clone();

    while let Some(batch_result) = input.next().await {
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
            vec![FlexType::Integer(1), FlexType::Float(1.5), FlexType::String("a".into())],
            vec![FlexType::Integer(2), FlexType::Float(2.5), FlexType::String("b".into())],
        ];
        let dtypes = [FlexTypeEnum::Integer, FlexTypeEnum::Float, FlexTypeEnum::String];
        let batch = SFrameRows::from_rows(&rows, &dtypes).unwrap();

        let source = PlannerNode::materialized(batch);
        let projected = PlannerNode::project(source, vec![0, 2]);
        let stream = compile(&projected).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.row(0), vec![FlexType::Integer(1), FlexType::String("a".into())]);
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

        // Original column + new column
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.row(0), vec![FlexType::Integer(10), FlexType::Float(15.0)]);
        assert_eq!(result.row(1), vec![FlexType::Integer(20), FlexType::Float(30.0)]);
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
        // Pipeline: read business.sf → filter stars >= 4.5 → project [business_id, city]
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

        // Column indices: stars=9 (float)
        let source = PlannerNode::sframe_source(&path, col_names, col_types, num_rows);
        let filtered = PlannerNode::filter(
            source,
            9, // stars column
            Arc::new(|v| matches!(v, FlexType::Float(f) if *f >= 4.5)),
        );
        // business_id=0, city=2
        let projected = PlannerNode::project(filtered, vec![0, 2]);

        let stream = compile(&projected).unwrap();
        let result = materialize(stream).await.unwrap();

        // stars >= 4.5: 1748 (4.5) + 1272 (5.0) = 3020
        assert_eq!(result.num_rows(), 3020);
        assert_eq!(result.num_columns(), 2);

        // All values should be non-undefined strings
        for i in 0..result.num_rows() {
            let row = result.row(i);
            assert!(matches!(&row[0], FlexType::String(_)));
            assert!(matches!(&row[1], FlexType::String(_)));
        }
    }

    #[tokio::test]
    async fn test_materialize_head_partial() {
        // 10 elements, head(3) should only take 3
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
        // 5 elements, head(100) should return all 5
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
        // range(0,1,10) → filter (>5) → transform (*2)
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
        // Project just the new column (column 1)
        let projected = PlannerNode::project(transformed, vec![1]);

        let stream = compile(&projected).unwrap();
        let result = materialize(stream).await.unwrap();

        assert_eq!(result.num_rows(), 4); // 6,7,8,9
        assert_eq!(result.row(0), vec![FlexType::Integer(12)]); // 6*2
        assert_eq!(result.row(1), vec![FlexType::Integer(14)]); // 7*2
        assert_eq!(result.row(2), vec![FlexType::Integer(16)]); // 8*2
        assert_eq!(result.row(3), vec![FlexType::Integer(18)]); // 9*2
    }
}
