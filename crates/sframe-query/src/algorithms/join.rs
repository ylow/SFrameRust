//! Hash join algorithm.
//!
//! Supports INNER, LEFT, RIGHT, and FULL outer joins.
//! Materializes both sides, builds a hash table on the right, probes with the left.

use std::collections::HashMap;

use futures::StreamExt;

use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::{ColumnData, SFrameRows};
use crate::execute::BatchStream;

/// Join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Join key specification: (left_column, right_column).
#[derive(Debug, Clone)]
pub struct JoinOn {
    pub left_column: usize,
    pub right_column: usize,
}

impl JoinOn {
    pub fn new(left: usize, right: usize) -> Self {
        JoinOn {
            left_column: left,
            right_column: right,
        }
    }
}

/// Perform a hash join of two streams.
///
/// Output schema: all left columns followed by all right columns (except join key column).
pub async fn join(
    mut left_stream: BatchStream,
    mut right_stream: BatchStream,
    on: &JoinOn,
    join_type: JoinType,
) -> Result<SFrameRows> {
    // Materialize both sides
    let left = materialize_stream(&mut left_stream).await?;
    let right = materialize_stream(&mut right_stream).await?;

    let left_rows = left.num_rows();
    let right_rows = right.num_rows();

    if left_rows == 0 && right_rows == 0 {
        return Ok(SFrameRows::empty(&[]));
    }

    // Build hash table on the right side
    let mut right_index: HashMap<FlexTypeHashKey, Vec<usize>> = HashMap::new();
    for i in 0..right_rows {
        let key = FlexTypeHashKey(right.column(on.right_column).get(i));
        right_index.entry(key).or_default().push(i);
    }

    // Determine output schema: all left cols + right cols (excluding join key)
    let left_dtypes = left.dtypes();
    let right_dtypes = right.dtypes();
    let right_output_cols: Vec<usize> = (0..right.num_columns())
        .filter(|&c| c != on.right_column)
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
        let key = FlexTypeHashKey(left.column(on.left_column).get(left_idx));
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
        for right_idx in 0..right_rows {
            if !right_matched[right_idx] {
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

async fn materialize_stream(stream: &mut BatchStream) -> Result<SFrameRows> {
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
    for col in 0..left_ncols {
        let val = left.column(col).get(left_idx);
        output_cols[col].push(&val)?;
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
    for col in 0..left_ncols {
        output_cols[col].push(&FlexType::Undefined)?;
    }

    // Right columns (excluding join key)
    for (out_offset, &right_col) in right_output_cols.iter().enumerate() {
        let val = right.column(right_col).get(right_idx);
        output_cols[left_ncols + out_offset].push(&val)?;
    }

    Ok(())
}

/// Wrapper for FlexType to implement Hash + Eq for join keys.
#[derive(Clone, Debug)]
struct FlexTypeHashKey(FlexType);

impl PartialEq for FlexTypeHashKey {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (FlexType::Integer(a), FlexType::Integer(b)) => a == b,
            (FlexType::Float(a), FlexType::Float(b)) => a.to_bits() == b.to_bits(),
            (FlexType::String(a), FlexType::String(b)) => a == b,
            (FlexType::Undefined, FlexType::Undefined) => true,
            _ => false,
        }
    }
}

impl Eq for FlexTypeHashKey {}

impl std::hash::Hash for FlexTypeHashKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(&self.0).hash(state);
        match &self.0 {
            FlexType::Integer(i) => i.hash(state),
            FlexType::Float(f) => f.to_bits().hash(state),
            FlexType::String(s) => s.hash(state),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn make_stream(batch: SFrameRows) -> BatchStream {
        Box::pin(stream::once(async { Ok(batch) }))
    }

    #[tokio::test]
    async fn test_inner_join() {
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

        let result = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::new(0, 0),
            JoinType::Inner,
        )
        .await
        .unwrap();

        // Should have 2 matched rows (ids 1 and 3)
        assert_eq!(result.num_rows(), 2);
        // Output: left_id, left_name, right_score (right_id excluded as join key)
        assert_eq!(result.num_columns(), 3);
    }

    #[tokio::test]
    async fn test_left_join() {
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

        let result = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::new(0, 0),
            JoinType::Left,
        )
        .await
        .unwrap();

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

    #[tokio::test]
    async fn test_full_join() {
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

        let result = join(
            make_stream(left),
            make_stream(right),
            &JoinOn::new(0, 0),
            JoinType::Full,
        )
        .await
        .unwrap();

        // Both rows should appear (no match between them)
        assert_eq!(result.num_rows(), 2);
    }
}
