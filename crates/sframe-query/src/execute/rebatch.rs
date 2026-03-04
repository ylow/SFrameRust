//! Rebatch adapter: normalize batch sizes to a target row count.
//!
//! Inserted between operators with different rate IDs at a lockstep
//! multi-input operator. Buffers incoming batches and re-emits them
//! at a fixed target size (except the last batch which may be smaller).

use std::collections::VecDeque;

use futures::stream;
use futures::StreamExt;

use sframe_types::error::Result;

use crate::batch::SFrameRows;

use super::BatchStream;

/// Internal state for the rebatch stream.
struct RebatchState {
    input: BatchStream,
    /// Queued batches not yet emitted.
    buffer: VecDeque<SFrameRows>,
    /// Total rows across all batches in `buffer`.
    buffer_rows: usize,
    /// Target output batch size.
    target_size: usize,
    /// Whether the input stream is exhausted.
    input_exhausted: bool,
}

/// Wrap a stream to emit batches of exactly `target_size` rows
/// (except the final batch which may be smaller).
pub(super) fn rebatch(input: BatchStream, target_size: usize) -> BatchStream {
    let state = RebatchState {
        input,
        buffer: VecDeque::new(),
        buffer_rows: 0,
        target_size,
        input_exhausted: false,
    };

    Box::pin(stream::unfold(state, |mut state| async move {
        // Fill buffer until we have enough rows or input is exhausted.
        while state.buffer_rows < state.target_size && !state.input_exhausted {
            match state.input.next().await {
                None => {
                    state.input_exhausted = true;
                    break;
                }
                Some(Err(e)) => return Some((Err(e), state)),
                Some(Ok(batch)) => {
                    state.buffer_rows += batch.num_rows();
                    state.buffer.push_back(batch);
                }
            }
        }

        if state.buffer_rows == 0 {
            return None; // done
        }

        // Emit exactly target_size rows (or all remaining if input exhausted).
        let emit_count = state.buffer_rows.min(state.target_size);
        match drain_rows(&mut state.buffer, &mut state.buffer_rows, emit_count) {
            Ok(batch) => Some((Ok(batch), state)),
            Err(e) => Some((Err(e), state)),
        }
    }))
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
