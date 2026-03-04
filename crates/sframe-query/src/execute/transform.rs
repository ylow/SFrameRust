//! Transform operators: unary, binary, and generalized row transforms.

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::{ColumnData, SFrameRows};

/// Apply a unary transform: produce a single output column from an input column.
pub(super) fn apply_transform(
    batch: &SFrameRows,
    input_column: usize,
    func: &(dyn Fn(&FlexType) -> FlexType + Send + Sync),
    output_type: FlexTypeEnum,
) -> Result<SFrameRows> {
    let col = batch.column(input_column);
    let results = col.map(func);
    let new_col = ColumnData::from_flex_vec(results, output_type);
    SFrameRows::new(vec![new_col])
}

/// Apply a binary transform: produce a single output column from two input columns.
pub(super) fn apply_binary_transform(
    batch: &SFrameRows,
    left_col: usize,
    right_col: usize,
    func: &(dyn Fn(&FlexType, &FlexType) -> FlexType + Send + Sync),
    output_type: FlexTypeEnum,
) -> Result<SFrameRows> {
    let n = batch.num_rows();
    let results: Vec<FlexType> = (0..n)
        .map(|i| {
            let left = batch.column(left_col).get(i);
            let right = batch.column(right_col).get(i);
            func(&left, &right)
        })
        .collect();

    let new_col = ColumnData::from_flex_vec(results, output_type);
    SFrameRows::new(vec![new_col])
}

/// Apply a generalized transform: replace all columns with transform output.
pub(super) fn apply_generalized_transform(
    batch: &SFrameRows,
    func: &(dyn Fn(&[FlexType]) -> Vec<FlexType> + Send + Sync),
    output_types: &[FlexTypeEnum],
) -> Result<SFrameRows> {
    let n = batch.num_rows();
    let num_out_cols = output_types.len();

    // Collect results row-major, then transpose to column-major.
    let all_results: Vec<Vec<FlexType>> = (0..n)
        .map(|i| {
            let row: Vec<FlexType> = batch.row(i);
            func(&row)
        })
        .collect();

    // Transpose: build per-column Vec<FlexType>, then convert in bulk.
    let mut col_vecs: Vec<Vec<FlexType>> = (0..num_out_cols)
        .map(|_| Vec::with_capacity(n))
        .collect();

    for results in all_results {
        if results.len() != num_out_cols {
            return Err(SFrameError::Format(format!(
                "GeneralizedTransform produced {} values, expected {}",
                results.len(),
                num_out_cols
            )));
        }
        for (col_idx, val) in results.into_iter().enumerate() {
            col_vecs[col_idx].push(val);
        }
    }

    let columns: Vec<ColumnData> = col_vecs
        .into_iter()
        .zip(output_types.iter())
        .map(|(data, &dtype)| ColumnData::from_flex_vec(data, dtype))
        .collect();

    SFrameRows::new(columns)
}
