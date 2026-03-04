//! Range operator: generate a sequence of integers.

use futures::stream;

use sframe_types::error::Result;

use crate::batch::{ColumnData, SFrameRows};

use super::BatchStream;

/// Compile a range source.
pub(super) fn compile_range(start: i64, step: i64, count: u64) -> Result<BatchStream> {
    let batch_size = sframe_config::global().source_batch_size;
    let mut batches: Vec<Result<SFrameRows>> = Vec::new();
    let total = count as usize;
    let mut offset = 0;
    while offset < total {
        let end = (offset + batch_size).min(total);
        let values: Vec<Option<i64>> = (offset..end)
            .map(|i| Some(start + (i as i64) * step))
            .collect();
        let col = ColumnData::Integer(values);
        batches.push(SFrameRows::new(vec![col]));
        offset = end;
    }
    Ok(Box::pin(stream::iter(batches)))
}
