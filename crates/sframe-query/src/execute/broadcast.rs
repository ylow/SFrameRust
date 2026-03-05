//! Fan-out broadcast: single source BatchIterator -> N consumer BatchIterators.
//!
//! When a plan node is shared by multiple downstream operators, compile it
//! once and broadcast its output to all consumers. Each consumer gets its
//! own BatchIterator backed by shared state (`Rc<RefCell<...>>`).
//!
//! Consumers drive the source on demand. Batches are buffered and freed
//! once all consumers have advanced past them.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use sframe_types::error::{Result, SFrameError};

use crate::batch::SFrameRows;

use super::batch_iter::{BatchCommand, BatchCo, BatchIterator, BatchResponse};

/// Shared inner state for broadcast.
struct SharedSourceInner {
    /// The underlying source.
    source: BatchIterator,
    /// Buffer of batches read from the source so far.
    /// Each entry is Arc-wrapped so consumers can cheaply clone.
    buffer: std::collections::VecDeque<Arc<Result<SFrameRows>>>,
    /// The global batch index corresponding to buffer[0].
    /// Entries before this have been consumed by all consumers and freed.
    base_index: usize,
    /// Per-consumer cursor positions (global batch index).
    cursors: Vec<usize>,
    /// Whether the source is exhausted.
    source_done: bool,
}

impl SharedSourceInner {
    /// Ensure the buffer has a batch at global index `idx`.
    /// If not, drive the source forward.
    fn ensure_batch(&mut self, idx: usize) -> bool {
        let needed = idx + 1;
        while self.base_index + self.buffer.len() < needed {
            if self.source_done {
                return false;
            }
            match self.source.next_batch() {
                None => {
                    self.source_done = true;
                    return false;
                }
                Some(result) => {
                    self.buffer.push_back(Arc::new(result));
                }
            }
        }
        true
    }

    /// Get a batch by global index.
    fn get_batch(&self, idx: usize) -> &Arc<Result<SFrameRows>> {
        &self.buffer[idx - self.base_index]
    }

    /// Free buffer entries that all consumers have advanced past.
    fn gc(&mut self) {
        let min_cursor = self.cursors.iter().copied().min().unwrap_or(0);
        while self.base_index < min_cursor {
            self.buffer.pop_front();
            self.base_index += 1;
        }
    }
}

/// Manages broadcast from one source BatchIterator to N consumers.
pub(super) struct BroadcastState {
    inner: Rc<RefCell<SharedSourceInner>>,
    num_consumers: usize,
    next_subscriber: usize,
}

impl BroadcastState {
    /// Create a broadcast from `source` to `num_consumers` consumers.
    ///
    /// `_buffer` is accepted for API compatibility but not used (the
    /// shared buffer grows on demand and entries are freed when all
    /// consumers have advanced past them).
    pub fn new(source: BatchIterator, num_consumers: usize, _buffer: usize) -> Self {
        assert!(num_consumers >= 2, "broadcast requires at least 2 consumers");

        let inner = Rc::new(RefCell::new(SharedSourceInner {
            source,
            buffer: std::collections::VecDeque::new(),
            base_index: 0,
            cursors: vec![0; num_consumers],
            source_done: false,
        }));

        BroadcastState {
            inner,
            num_consumers,
            next_subscriber: 0,
        }
    }

    /// Hand out the next consumer BatchIterator.
    ///
    /// Each call returns a new `BatchIterator` backed by the shared source.
    /// Panics if called more times than `num_consumers`.
    pub fn subscribe(&mut self) -> BatchIterator {
        assert!(
            self.next_subscriber < self.num_consumers,
            "subscriber already taken"
        );
        let consumer_id = self.next_subscriber;
        self.next_subscriber += 1;

        let inner = Rc::clone(&self.inner);

        BatchIterator::new(move |co: BatchCo| async move {
            let mut cmd = co.yield_(BatchResponse::Ready).await;
            let mut cursor: usize = 0;

            loop {
                match cmd {
                    BatchCommand::NextBatch => {
                        let has_batch = {
                            let mut shared = inner.borrow_mut();
                            shared.ensure_batch(cursor)
                        };

                        if !has_batch {
                            return;
                        }

                        let arc_result = {
                            let shared = inner.borrow();
                            Arc::clone(shared.get_batch(cursor))
                        };
                        cursor += 1;

                        // Update cursor and GC
                        {
                            let mut shared = inner.borrow_mut();
                            shared.cursors[consumer_id] = cursor;
                            shared.gc();
                        }

                        match &*arc_result {
                            Ok(batch) => {
                                cmd = co.yield_(BatchResponse::Batch(Ok(batch.clone()))).await;
                            }
                            Err(e) => {
                                cmd = co.yield_(BatchResponse::Batch(Err(
                                    SFrameError::Format(e.to_string()),
                                ))).await;
                            }
                        }
                    }
                    BatchCommand::SkipBatch => {
                        // Still need to ensure the batch exists (to keep cursors aligned
                        // with other consumers who may need it).
                        let has_batch = {
                            let mut shared = inner.borrow_mut();
                            shared.ensure_batch(cursor)
                        };

                        if !has_batch {
                            return;
                        }

                        cursor += 1;

                        // Update cursor and GC
                        {
                            let mut shared = inner.borrow_mut();
                            shared.cursors[consumer_id] = cursor;
                            shared.gc();
                        }

                        cmd = co.yield_(BatchResponse::Skipped).await;
                    }
                    BatchCommand::Start => unreachable!(),
                }
            }
        })
    }
}
