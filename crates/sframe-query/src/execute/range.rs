//! Range operator: generate a sequence of integers.

use sframe_types::error::Result;

use crate::batch::{ColumnData, SFrameRows};

use super::batch_iter::{BatchCommand, BatchCo, BatchIterator, BatchResponse};

/// Compile a range source.
pub(super) fn compile_range(start: i64, step: i64, count: u64) -> Result<BatchIterator> {
    let batch_size = sframe_config::global().source_batch_size();
    let total = count as usize;

    Ok(BatchIterator::new(move |co: BatchCo| async move {
        let mut cmd = co.yield_(BatchResponse::Ready).await;
        let mut offset = 0;
        while offset < total {
            let end = (offset + batch_size).min(total);
            match cmd {
                BatchCommand::NextBatch => {
                    let values: Vec<Option<i64>> = (offset..end)
                        .map(|i| Some(start + (i as i64) * step))
                        .collect();
                    let col = ColumnData::Integer(values);
                    let batch = SFrameRows::new(vec![col]);
                    cmd = co.yield_(BatchResponse::Batch(batch)).await;
                }
                BatchCommand::SkipBatch => {
                    cmd = co.yield_(BatchResponse::Skipped).await;
                }
                BatchCommand::Start => unreachable!(),
            }
            offset = end;
        }
    }))
}
