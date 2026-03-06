//! BatchIterator: synchronous pull-based batch producer using genawaiter.
//!
//! Each operator is a genawaiter coroutine that yields `BatchResponse` values
//! and receives `BatchCommand` via `resume_with`. The `BatchIterator` wrapper
//! provides a uniform `next_batch()` / `skip_batch()` API.

use std::future::Future;
use std::pin::Pin;

use genawaiter::rc::{Co, Gen};
use genawaiter::GeneratorState;

use sframe_types::error::{Result, SFrameError};

use crate::batch::SFrameRows;

// ---------------------------------------------------------------------------
// Protocol types
// ---------------------------------------------------------------------------

/// Commands sent into the generator via `resume_with`.
#[derive(Debug, Clone, Copy)]
pub enum BatchCommand {
    /// Initial resume. Dropped by genawaiter (never seen by generator body).
    Start,
    /// Request the next batch of data.
    NextBatch,
    /// Skip the next batch without producing data.
    SkipBatch,
}

/// Responses yielded by the generator.
#[derive(Debug)]
pub enum BatchResponse {
    /// Generator initialized and ready for commands.
    Ready,
    /// A batch of data (or an error).
    Batch(Result<SFrameRows>),
    /// Batch was skipped successfully.
    Skipped,
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Convenience alias for the coroutine handle passed to producer functions.
pub type BatchCo = Co<BatchResponse, BatchCommand>;

/// The concrete generator type used by `BatchIterator`.
type InnerGen = Gen<BatchResponse, BatchCommand, Pin<Box<dyn Future<Output = ()>>>>;

// ---------------------------------------------------------------------------
// BatchIterator
// ---------------------------------------------------------------------------

/// A synchronous, pull-based batch iterator backed by a genawaiter generator.
///
/// Wraps the `resume_with` protocol so callers see a simple
/// `next_batch()` / `skip_batch()` interface.
pub struct BatchIterator {
    gen: InnerGen,
    done: bool,
}

impl BatchIterator {
    /// Create a new `BatchIterator` from a producer function.
    ///
    /// The producer receives a [`BatchCo`] handle and should:
    /// 1. Yield `BatchResponse::Ready` as its first yield.
    /// 2. Read the command returned by that yield's `.await` to get the first
    ///    real command (NextBatch or SkipBatch).
    /// 3. Respond with `Batch(...)` or `Skipped` as appropriate.
    /// 4. Return when there are no more batches to produce.
    pub fn new<F, Fut>(producer: F) -> Self
    where
        F: FnOnce(BatchCo) -> Fut,
        Fut: Future<Output = ()> + 'static,
    {
        let mut gen: InnerGen = Gen::new(|co| {
            Box::pin(producer(co)) as Pin<Box<dyn Future<Output = ()>>>
        });

        // Bootstrap: first resume_with value is dropped by genawaiter.
        // The generator runs to its first yield (Ready).
        let state = gen.resume_with(BatchCommand::Start);
        debug_assert!(
            matches!(state, GeneratorState::Yielded(BatchResponse::Ready)),
            "BatchIterator: producer must yield Ready as first response"
        );

        BatchIterator { gen, done: false }
    }

    /// Request the next batch from the producer.
    ///
    /// Returns `Some(Ok(batch))` for a successful batch, `Some(Err(e))` for
    /// an error, or `None` when the producer is exhausted.
    pub fn next_batch(&mut self) -> Option<Result<SFrameRows>> {
        if self.done {
            return None;
        }

        match self.gen.resume_with(BatchCommand::NextBatch) {
            GeneratorState::Yielded(BatchResponse::Batch(result)) => Some(result),
            GeneratorState::Yielded(BatchResponse::Skipped) => {
                Some(Err(SFrameError::Format(
                    "BatchIterator protocol violation: got Skipped in response to NextBatch".into(),
                )))
            }
            GeneratorState::Yielded(BatchResponse::Ready) => {
                Some(Err(SFrameError::Format(
                    "BatchIterator protocol violation: got Ready after initialization".into(),
                )))
            }
            GeneratorState::Complete(()) => {
                self.done = true;
                None
            }
        }
    }

    /// Ask the producer to skip the next batch without producing data.
    ///
    /// Returns `Some(())` if the batch was skipped, or `None` if the
    /// producer is exhausted.
    pub fn skip_batch(&mut self) -> Option<()> {
        if self.done {
            return None;
        }

        match self.gen.resume_with(BatchCommand::SkipBatch) {
            GeneratorState::Yielded(BatchResponse::Skipped) => Some(()),
            GeneratorState::Yielded(BatchResponse::Batch(_)) => {
                // Producer responded with Batch to a SkipBatch command.
                // Treat as skipped (the data is discarded).
                Some(())
            }
            GeneratorState::Yielded(BatchResponse::Ready) => {
                self.done = true;
                None
            }
            GeneratorState::Complete(()) => {
                self.done = true;
                None
            }
        }
    }

    /// Whether the producer has been exhausted.
    pub fn is_done(&self) -> bool {
        self.done
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sframe_types::error::SFrameError;
    use sframe_types::flex_type::FlexTypeEnum;

    /// Helper: create a simple SFrameRows batch with integer values.
    fn make_batch(values: &[i64]) -> SFrameRows {
        let rows: Vec<Vec<sframe_types::flex_type::FlexType>> = values
            .iter()
            .map(|&v| vec![sframe_types::flex_type::FlexType::Integer(v)])
            .collect();
        SFrameRows::from_rows(&rows, &[FlexTypeEnum::Integer]).unwrap()
    }

    #[test]
    fn test_simple_producer() {
        let batch = make_batch(&[1, 2, 3]);

        let iter = BatchIterator::new(|co: BatchCo| async move {
            let cmd = co.yield_(BatchResponse::Ready).await;
            if let BatchCommand::NextBatch = cmd {
                co.yield_(BatchResponse::Batch(Ok(batch))).await;
            }
        });

        let mut iter = iter;

        // First call should return the batch.
        let result = iter.next_batch();
        assert!(result.is_some());
        let batch = result.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);

        // Second call should return None (producer is done).
        let result = iter.next_batch();
        assert!(result.is_none());
        assert!(iter.is_done());
    }

    #[test]
    fn test_skip_batch() {
        let batches: Vec<SFrameRows> = (0..3).map(|i| make_batch(&[i * 10, i * 10 + 1])).collect();

        let mut iter = BatchIterator::new(|co: BatchCo| async move {
            let mut cmd = co.yield_(BatchResponse::Ready).await;
            for batch in batches {
                match cmd {
                    BatchCommand::NextBatch => {
                        cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await;
                    }
                    BatchCommand::SkipBatch => {
                        cmd = co.yield_(BatchResponse::Skipped).await;
                    }
                    _ => break,
                }
            }
        });

        // Skip the first batch.
        let skipped = iter.skip_batch();
        assert_eq!(skipped, Some(()));

        // Get the second batch.
        let result = iter.next_batch().unwrap().unwrap();
        assert_eq!(result.num_rows(), 2);
        assert_eq!(
            result.row(0),
            vec![sframe_types::flex_type::FlexType::Integer(10)]
        );

        // Get the third batch.
        let result = iter.next_batch().unwrap().unwrap();
        assert_eq!(result.num_rows(), 2);
        assert_eq!(
            result.row(0),
            vec![sframe_types::flex_type::FlexType::Integer(20)]
        );

        // Should be done now.
        assert!(iter.next_batch().is_none());
        assert!(iter.is_done());
    }

    #[test]
    fn test_empty_producer() {
        let mut iter = BatchIterator::new(|co: BatchCo| async move {
            // Yield Ready then immediately return (no batches).
            co.yield_(BatchResponse::Ready).await;
        });

        // Should return None immediately.
        let result = iter.next_batch();
        assert!(result.is_none());
        assert!(iter.is_done());
    }

    #[test]
    fn test_error_propagation() {
        let mut iter = BatchIterator::new(|co: BatchCo| async move {
            let cmd = co.yield_(BatchResponse::Ready).await;
            if matches!(cmd, BatchCommand::NextBatch) {
                co.yield_(BatchResponse::Batch(Err(SFrameError::Format(
                    "test error".to_string(),
                ))))
                .await;
            }
        });

        let result = iter.next_batch();
        assert!(result.is_some());
        let err = result.unwrap().unwrap_err();
        assert!(
            matches!(err, SFrameError::Format(ref msg) if msg == "test error"),
            "Expected Format error, got: {err:?}"
        );
    }

    #[test]
    fn test_multi_batch_producer() {
        let num_batches = 5;

        let mut iter = BatchIterator::new(|co: BatchCo| async move {
            let mut cmd = co.yield_(BatchResponse::Ready).await;
            for i in 0..num_batches {
                match cmd {
                    BatchCommand::NextBatch => {
                        let batch = make_batch(&[i as i64]);
                        cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await;
                    }
                    BatchCommand::SkipBatch => {
                        cmd = co.yield_(BatchResponse::Skipped).await;
                    }
                    _ => break,
                }
            }
        });

        // Verify all 5 batches arrive in order.
        for i in 0..num_batches {
            let result = iter.next_batch();
            assert!(
                result.is_some(),
                "Expected batch {i}, got None"
            );
            let batch = result.unwrap().unwrap();
            assert_eq!(batch.num_rows(), 1);
            assert_eq!(
                batch.row(0),
                vec![sframe_types::flex_type::FlexType::Integer(i as i64)]
            );
        }

        // Should be done.
        assert!(iter.next_batch().is_none());
        assert!(iter.is_done());
    }
}
