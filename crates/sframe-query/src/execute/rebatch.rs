//! Rebatch adapter: normalize batch sizes to a target row count.
//!
//! Inserted between operators with different rate IDs at a lockstep
//! multi-input operator. Buffers incoming batches and re-emits them
//! at a fixed target size (except the last batch which may be smaller).

use std::collections::VecDeque;

use sframe_types::error::Result;

use crate::batch::SFrameRows;

use super::batch_iter::{BatchCommand, BatchCo, BatchIterator, BatchResponse};

/// Wrap a BatchIterator to emit batches of exactly `target_size` rows
/// (except the final batch which may be smaller).
pub(super) fn rebatch(mut input: BatchIterator, target_size: usize) -> BatchIterator {
    BatchIterator::new(move |co: BatchCo| async move {
        let mut cmd = co.yield_(BatchResponse::Ready).await;
        let mut buffer: VecDeque<SFrameRows> = VecDeque::new();
        let mut buffer_rows: usize = 0;
        let mut input_exhausted = false;

        loop {
            match cmd {
                BatchCommand::NextBatch => {
                    // Fill buffer until we have enough rows or input is exhausted.
                    let mut got_error = false;
                    while buffer_rows < target_size && !input_exhausted {
                        match input.next_batch() {
                            None => {
                                input_exhausted = true;
                                break;
                            }
                            Some(Err(e)) => {
                                cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                                got_error = true;
                                break;
                            }
                            Some(Ok(batch)) => {
                                buffer_rows += batch.num_rows();
                                buffer.push_back(batch);
                            }
                        }
                    }

                    if got_error {
                        continue; // re-match cmd in outer loop
                    }

                    if buffer_rows == 0 {
                        return; // done
                    }

                    // Emit exactly target_size rows (or all remaining if input exhausted).
                    let emit_count = buffer_rows.min(target_size);
                    match drain_rows(&mut buffer, &mut buffer_rows, emit_count) {
                        Ok(batch) => {
                            cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await;
                        }
                        Err(e) => {
                            cmd = co.yield_(BatchResponse::Batch(Err(e))).await;
                        }
                    }
                }
                BatchCommand::SkipBatch => {
                    // Drain buffer first, then propagate skip to input.
                    if buffer_rows > 0 {
                        let drain_count = buffer_rows.min(target_size);
                        let _ = drain_rows(&mut buffer, &mut buffer_rows, drain_count);
                        cmd = co.yield_(BatchResponse::Skipped).await;
                    } else if !input_exhausted {
                        match input.skip_batch() {
                            None => {
                                return;
                            }
                            Some(()) => {
                                cmd = co.yield_(BatchResponse::Skipped).await;
                            }
                        }
                    } else {
                        return;
                    }
                }
                BatchCommand::Start => unreachable!(),
            }
        }
    })
}

/// Drain exactly `count` rows from the front of the buffer.
///
/// Consumes complete batches when possible. If a batch has more rows
/// than needed, splits it: takes the first `count` rows and pushes the
/// remainder back to the front of the buffer.
fn drain_rows(
    buffer: &mut VecDeque<SFrameRows>,
    buffer_rows: &mut usize,
    count: usize,
) -> Result<SFrameRows> {
    let mut remaining = count;
    let mut result: Option<SFrameRows> = None;

    while remaining > 0 {
        let batch = buffer.pop_front().expect("buffer underflow");
        let batch_rows = batch.num_rows();
        *buffer_rows -= batch_rows;

        if batch_rows <= remaining {
            // Consume entire batch.
            remaining -= batch_rows;
            match &mut result {
                None => result = Some(batch),
                Some(existing) => existing.append(&batch)?,
            }
        } else {
            // Split: take first `remaining` rows, push rest back.
            let take_indices: Vec<usize> = (0..remaining).collect();
            let rest_indices: Vec<usize> = (remaining..batch_rows).collect();

            let taken = batch.take(&take_indices)?;
            let leftover = batch.take(&rest_indices)?;

            let leftover_rows = leftover.num_rows();
            buffer.push_front(leftover);
            *buffer_rows += leftover_rows;

            match &mut result {
                None => result = Some(taken),
                Some(existing) => existing.append(&taken)?,
            }
            remaining = 0;
        }
    }

    Ok(result.unwrap_or_else(|| SFrameRows::empty(&[])))
}
