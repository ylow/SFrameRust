//! Filter operators: predicate-based and logical (mask-based) filtering.

use sframe_types::error::{Result, SFrameError};

use crate::batch::{ColumnData, SFrameRows};

/// Apply logical filter: keep rows from `data` where `mask` column 0 is truthy.
pub(super) fn logical_filter_batch(data: &SFrameRows, mask: &SFrameRows) -> Result<SFrameRows> {
    if data.num_rows() != mask.num_rows() {
        return Err(SFrameError::Format(format!(
            "LogicalFilter: data has {} rows but mask has {} rows",
            data.num_rows(),
            mask.num_rows()
        )));
    }
    let indices = mask.column(0).truthy_indices();

    let columns: Vec<ColumnData> = data.columns().iter()
        .map(|col| col.gather(&indices))
        .collect();

    Ok(SFrameRows::new(columns)?)
}
