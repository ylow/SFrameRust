//! Hash join algorithm.
//!
//! Supports INNER, LEFT, RIGHT, and FULL outer joins.
//! Materializes both sides, builds a hash table on the right, probes with the left.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::{ColumnData, SFrameRows};
use crate::execute::{BatchCo, BatchCommand, BatchIterator, BatchResponse};

/// Join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Join key specification: pairs of (left_column, right_column) indices.
#[derive(Debug, Clone)]
pub struct JoinOn {
    pub pairs: Vec<(usize, usize)>,
}

impl JoinOn {
    /// Single-column join.
    pub fn new(left: usize, right: usize) -> Self {
        JoinOn {
            pairs: vec![(left, right)],
        }
    }

    /// Multi-column join.
    pub fn multi(pairs: Vec<(usize, usize)>) -> Self {
        JoinOn { pairs }
    }

    fn left_columns(&self) -> Vec<usize> {
        self.pairs.iter().map(|&(l, _)| l).collect()
    }

    fn right_columns(&self) -> Vec<usize> {
        self.pairs.iter().map(|&(_, r)| r).collect()
    }
}

/// Perform a hash join of two streams.
///
/// Returns a BatchIterator of result batches. For small inputs, uses an in-memory
/// hash join. For large inputs, will use GRACE hash partitioned join.
///
/// Output schema: all left columns followed by all right columns (except join key columns).
pub fn join(
    mut left_stream: BatchIterator,
    mut right_stream: BatchIterator,
    on: &JoinOn,
    join_type: JoinType,
) -> Result<BatchIterator> {
    // Materialize both sides to estimate sizes
    let left = materialize_stream(&mut left_stream)?;
    let right = materialize_stream(&mut right_stream)?;

    let left_cells = left.num_rows() * left.num_columns().max(1);
    let right_cells = right.num_rows() * right.num_columns().max(1);
    let budget = sframe_config::global().join_buffer_num_cells;

    let smaller_cells = left_cells.min(right_cells);

    if smaller_cells <= budget {
        // In-memory fast path
        let result = in_memory_join(left, right, on, join_type)?;
        Ok(crate::execute::BatchIterator::new(move |co: BatchCo| async move {
            let cmd = co.yield_(BatchResponse::Ready).await;
            if matches!(cmd, BatchCommand::NextBatch) {
                co.yield_(BatchResponse::Batch(Ok(result))).await;
            }
        }))
    } else {
        // GRACE hash join
        grace_hash_join(left, right, on, join_type, budget)
    }
}

/// In-memory hash join on already-materialized inputs.
fn in_memory_join(
    left: SFrameRows,
    right: SFrameRows,
    on: &JoinOn,
    join_type: JoinType,
) -> Result<SFrameRows> {
    let left_rows = left.num_rows();
    let right_rows = right.num_rows();

    if left_rows == 0 && right_rows == 0 {
        return Ok(SFrameRows::empty(&[]));
    }

    let left_key_cols = on.left_columns();
    let right_key_cols = on.right_columns();

    // Build hash table on the right side
    let mut right_index: HashMap<Vec<FlexType>, Vec<usize>> = HashMap::new();
    for i in 0..right_rows {
        let key: Vec<FlexType> = right_key_cols.iter().map(|&c| right.column(c).get(i)).collect();
        right_index.entry(key).or_default().push(i);
    }

    // Determine output schema: all left cols + right cols (excluding join key columns)
    let left_dtypes = left.dtypes();
    let right_dtypes = right.dtypes();
    let right_key_set: std::collections::HashSet<usize> = right_key_cols.iter().copied().collect();
    let right_output_cols: Vec<usize> = (0..right.num_columns())
        .filter(|c| !right_key_set.contains(c))
        .collect();

    let mut output_dtypes: Vec<FlexTypeEnum> = left_dtypes.clone();
    for &col in &right_output_cols {
        output_dtypes.push(right_dtypes[col]);
    }

    let mut output_cols: Vec<ColumnData> = output_dtypes
        .iter()
        .map(|&dt| ColumnData::empty(dt))
        .collect();

    let mut right_matched = vec![false; right_rows];

    // Probe: for each left row, find matching right rows
    for left_idx in 0..left_rows {
        let key: Vec<FlexType> = left_key_cols.iter().map(|&c| left.column(c).get(left_idx)).collect();
        let matches = right_index.get(&key);

        if let Some(right_indices) = matches {
            for &right_idx in right_indices {
                right_matched[right_idx] = true;
                emit_row(
                    &mut output_cols,
                    &left,
                    left_idx,
                    &right,
                    Some(right_idx),
                    &right_output_cols,
                )?;
            }
        } else if join_type == JoinType::Left || join_type == JoinType::Full {
            // Left row with no match — NULL-pad right side
            emit_row(
                &mut output_cols,
                &left,
                left_idx,
                &right,
                None,
                &right_output_cols,
            )?;
        } else if join_type == JoinType::Inner {
            // Skip unmatched left rows for inner join
        }
    }

    // For RIGHT or FULL join: emit unmatched right rows
    if join_type == JoinType::Right || join_type == JoinType::Full {
        for (right_idx, &matched) in right_matched.iter().enumerate().take(right_rows) {
            if !matched {
                emit_right_only(
                    &mut output_cols,
                    &left,
                    &right,
                    right_idx,
                    &right_output_cols,
                )?;
            }
        }
    }

    Ok(SFrameRows::new(output_cols).unwrap_or_else(|_| {
        SFrameRows::empty(&output_dtypes)
    }))
}

fn materialize_stream(stream: &mut BatchIterator) -> Result<SFrameRows> {
    let mut result: Option<SFrameRows> = None;
    while let Some(batch_result) = stream.next_batch() {
        let batch = batch_result?;
        match &mut result {
            None => result = Some(batch),
            Some(existing) => existing.append(&batch)?,
        }
    }
    Ok(result.unwrap_or_else(|| SFrameRows::empty(&[])))
}

fn emit_row(
    output_cols: &mut [ColumnData],
    left: &SFrameRows,
    left_idx: usize,
    right: &SFrameRows,
    right_idx: Option<usize>,
    right_output_cols: &[usize],
) -> Result<()> {
    let left_ncols = left.num_columns();

    // Left columns
    for (col, out_col) in output_cols.iter_mut().enumerate().take(left_ncols) {
        let val = left.column(col).get(left_idx);
        out_col.push(&val)?;
    }

    // Right columns (excluding join key)
    for (out_offset, &right_col) in right_output_cols.iter().enumerate() {
        let val = match right_idx {
            Some(ri) => right.column(right_col).get(ri),
            None => FlexType::Undefined,
        };
        output_cols[left_ncols + out_offset].push(&val)?;
    }

    Ok(())
}

fn emit_right_only(
    output_cols: &mut [ColumnData],
    left: &SFrameRows,
    right: &SFrameRows,
    right_idx: usize,
    right_output_cols: &[usize],
) -> Result<()> {
    let left_ncols = left.num_columns();

    // NULL-pad left columns
    for out_col in output_cols.iter_mut().take(left_ncols) {
        out_col.push(&FlexType::Undefined)?;
    }

    // Right columns (excluding join key)
    for (out_offset, &right_col) in right_output_cols.iter().enumerate() {
        let val = right.column(right_col).get(right_idx);
        output_cols[left_ncols + out_offset].push(&val)?;
    }

    Ok(())
}

/// GRACE hash join: partition both sides by hash(key), then join each partition.
fn grace_hash_join(
    left: SFrameRows,
    right: SFrameRows,
    on: &JoinOn,
    join_type: JoinType,
    budget: usize,
) -> Result<BatchIterator> {
    let left_key_cols = on.left_columns();
    let right_key_cols = on.right_columns();

    // Determine number of partitions
    let smaller_cells =
        left.num_rows().min(right.num_rows()) * left.num_columns().max(right.num_columns()).max(1);
    let num_partitions = (smaller_cells / budget.max(1)).max(2);

    // Partition both sides
    let left_partitions = partition_rows(&left, &left_key_cols, num_partitions)?;
    let right_partitions = partition_rows(&right, &right_key_cols, num_partitions)?;

    // Drop the full materialized inputs
    drop(left);
    drop(right);

    // Process each partition: do in-memory join and yield batches
    let on_owned = on.clone();
    Ok(crate::execute::BatchIterator::new(move |co: BatchCo| async move {
        let mut cmd = co.yield_(BatchResponse::Ready).await;
        for idx in 0..left_partitions.len() {
            match cmd {
                BatchCommand::NextBatch => {
                    let left_part = left_partitions[idx].clone();
                    let right_part = right_partitions[idx].clone();

                    let result = match (left_part, right_part) {
                        (Some(lp), Some(rp)) => in_memory_join(lp, rp, &on_owned, join_type),
                        (Some(lp), None) => {
                            match join_type {
                                JoinType::Left | JoinType::Full => {
                                    in_memory_join(lp, SFrameRows::empty(&[]), &on_owned, join_type)
                                }
                                _ => Ok(SFrameRows::empty(&[])),
                            }
                        }
                        (None, Some(rp)) => {
                            match join_type {
                                JoinType::Right | JoinType::Full => {
                                    in_memory_join(SFrameRows::empty(&[]), rp, &on_owned, join_type)
                                }
                                _ => Ok(SFrameRows::empty(&[])),
                            }
                        }
                        (None, None) => Ok(SFrameRows::empty(&[])),
                    };

                    cmd = co.yield_(BatchResponse::Batch(result)).await;
                }
                BatchCommand::SkipBatch => {
                    cmd = co.yield_(BatchResponse::Skipped).await;
                }
                BatchCommand::Start => unreachable!(),
            }
        }
    }))
}

/// Partition rows by hash(key) into N buckets.
/// Returns Vec<Option<SFrameRows>> — None for empty partitions.
fn partition_rows(
    batch: &SFrameRows,
    key_cols: &[usize],
    num_partitions: usize,
) -> Result<Vec<Option<SFrameRows>>> {
    let mut partition_indices: Vec<Vec<usize>> = vec![Vec::new(); num_partitions];

    for row_idx in 0..batch.num_rows() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &col in key_cols {
            batch.column(col).get(row_idx).hash(&mut hasher);
        }
        let partition = (hasher.finish() as usize) % num_partitions;
        partition_indices[partition].push(row_idx);
    }

    let mut partitions = Vec::with_capacity(num_partitions);
    for indices in partition_indices {
        if indices.is_empty() {
            partitions.push(None);
        } else {
            partitions.push(Some(batch.take(&indices)?));
        }
    }
    Ok(partitions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execute::BatchIterator;

    fn make_stream(batch: SFrameRows) -> BatchIterator {
        BatchIterator::new(move |co: BatchCo| async move {
            let cmd = co.yield_(BatchResponse::Ready).await;
            if matches!(cmd, BatchCommand::NextBatch) {
                co.yield_(BatchResponse::Batch(Ok(batch))).await;
            }
        })
    }

    /// Consume a join result stream into a single batch for testing.
    fn collect_stream(mut stream: BatchIterator) -> SFrameRows {
        let mut result: Option<SFrameRows> = None;
        while let Some(batch_result) = stream.next_batch() {
            let batch = batch_result.unwrap();
            if batch.num_rows() == 0 {
                continue;
            }
            match &mut result {
                None => result = Some(batch),
                Some(existing) => existing.append(&batch).unwrap(),
            }
        }
        result.unwrap_or_else(|| SFrameRows::empty(&[]))
    }

    #[test]
    fn test_inner_join() {
        // Left: id, name
        let left_rows = vec![
            vec![FlexType::Integer(1), FlexType::String("alice".into())],
            vec![FlexType::Integer(2), FlexType::String("bob".into())],
            vec![FlexType::Integer(3), FlexType::String("charlie".into())],
        ];
        let left = SFrameRows::from_rows(
            &left_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        // Right: id, score
        let right_rows = vec![
            vec![FlexType::Integer(1), FlexType::Float(90.0)],
            vec![FlexType::Integer(3), FlexType::Float(85.0)],
            vec![FlexType::Integer(4), FlexType::Float(70.0)],
        ];
        let right = SFrameRows::from_rows(
            &right_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let result_stream = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::new(0, 0),
            JoinType::Inner,
        )
        .unwrap();
        let result = collect_stream(result_stream);

        // Should have 2 matched rows (ids 1 and 3)
        assert_eq!(result.num_rows(), 2);
        // Output: left_id, left_name, right_score (right_id excluded as join key)
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_left_join() {
        let left_rows = vec![
            vec![FlexType::Integer(1), FlexType::String("alice".into())],
            vec![FlexType::Integer(2), FlexType::String("bob".into())],
        ];
        let left = SFrameRows::from_rows(
            &left_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        let right_rows = vec![
            vec![FlexType::Integer(1), FlexType::Float(90.0)],
        ];
        let right = SFrameRows::from_rows(
            &right_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let result_stream = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::new(0, 0),
            JoinType::Left,
        )
        .unwrap();
        let result = collect_stream(result_stream);

        // Both left rows should appear
        assert_eq!(result.num_rows(), 2);

        // Row for id=2 should have Undefined score
        let mut found_null = false;
        for i in 0..result.num_rows() {
            let row = result.row(i);
            if matches!(&row[0], FlexType::Integer(2)) {
                assert_eq!(row[2], FlexType::Undefined);
                found_null = true;
            }
        }
        assert!(found_null, "Expected NULL-padded row for id=2");
    }

    #[test]
    fn test_full_join() {
        let left_rows = vec![
            vec![FlexType::Integer(1), FlexType::String("alice".into())],
        ];
        let left = SFrameRows::from_rows(
            &left_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        )
        .unwrap();

        let right_rows = vec![
            vec![FlexType::Integer(2), FlexType::Float(90.0)],
        ];
        let right = SFrameRows::from_rows(
            &right_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        )
        .unwrap();

        let result_stream = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::new(0, 0),
            JoinType::Full,
        )
        .unwrap();
        let result = collect_stream(result_stream);

        // Both rows should appear (no match between them)
        assert_eq!(result.num_rows(), 2);
    }

    #[test]
    fn test_multi_column_join() {
        // Left: dept, region, name
        let left_rows = vec![
            vec![FlexType::String("eng".into()), FlexType::String("us".into()), FlexType::String("alice".into())],
            vec![FlexType::String("eng".into()), FlexType::String("eu".into()), FlexType::String("bob".into())],
            vec![FlexType::String("sales".into()), FlexType::String("us".into()), FlexType::String("charlie".into())],
        ];
        let left = SFrameRows::from_rows(
            &left_rows,
            &[FlexTypeEnum::String, FlexTypeEnum::String, FlexTypeEnum::String],
        ).unwrap();

        // Right: dept, region, budget
        let right_rows = vec![
            vec![FlexType::String("eng".into()), FlexType::String("us".into()), FlexType::Float(100.0)],
            vec![FlexType::String("eng".into()), FlexType::String("eu".into()), FlexType::Float(80.0)],
            vec![FlexType::String("hr".into()), FlexType::String("us".into()), FlexType::Float(50.0)],
        ];
        let right = SFrameRows::from_rows(
            &right_rows,
            &[FlexTypeEnum::String, FlexTypeEnum::String, FlexTypeEnum::Float],
        ).unwrap();

        // Join on (dept, region)
        let result_stream = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::multi(vec![(0, 0), (1, 1)]),
            JoinType::Inner,
        ).unwrap();
        let result = collect_stream(result_stream);

        // eng+us → alice, eng+eu → bob. sales+us has no match.
        assert_eq!(result.num_rows(), 2);
        // Output: left.dept, left.region, left.name, right.budget
        // (right.dept and right.region excluded as join keys)
        assert_eq!(result.num_columns(), 4);
    }

    #[test]
    fn test_join_returns_stream() {
        // Test that join returns a BatchIterator, not SFrameRows
        let left_rows = vec![
            vec![FlexType::Integer(1), FlexType::String("alice".into())],
            vec![FlexType::Integer(2), FlexType::String("bob".into())],
        ];
        let left = SFrameRows::from_rows(
            &left_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        ).unwrap();

        let right_rows = vec![
            vec![FlexType::Integer(1), FlexType::Float(90.0)],
        ];
        let right = SFrameRows::from_rows(
            &right_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        ).unwrap();

        let mut result_stream = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::new(0, 0),
            JoinType::Inner,
        ).unwrap();

        // Consume the stream and collect all rows
        let mut total_rows = 0;
        while let Some(batch_result) = result_stream.next_batch() {
            let batch = batch_result.unwrap();
            total_rows += batch.num_rows();
        }
        assert_eq!(total_rows, 1);
    }

    #[test]
    fn test_join_large_dataset_partitioned() {
        // Test with enough data to trigger GRACE partitioning
        // (when join_buffer_num_cells is set low)
        let n = 1000;
        let left_rows: Vec<Vec<FlexType>> = (0..n)
            .map(|i| vec![FlexType::Integer(i), FlexType::String(format!("left_{i}").into())])
            .collect();
        let left = SFrameRows::from_rows(
            &left_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        ).unwrap();

        let right_rows: Vec<Vec<FlexType>> = (0..n)
            .map(|i| vec![FlexType::Integer(i), FlexType::Float(i as f64 * 10.0)])
            .collect();
        let right = SFrameRows::from_rows(
            &right_rows,
            &[FlexTypeEnum::Integer, FlexTypeEnum::Float],
        ).unwrap();

        let mut result_stream = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::new(0, 0),
            JoinType::Inner,
        ).unwrap();

        let mut total_rows = 0;
        while let Some(batch_result) = result_stream.next_batch() {
            let batch = batch_result.unwrap();
            total_rows += batch.num_rows();
        }
        assert_eq!(total_rows, n as usize);
    }
}
