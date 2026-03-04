//! Fan-out broadcast: single source stream → N consumer streams.
//!
//! When a plan node is shared by multiple downstream operators, compile it
//! once and broadcast its output to all consumers. Each consumer gets its
//! own bounded channel and sees every batch.

use std::sync::Arc;

use futures::stream;

use sframe_types::error::SFrameError;

use crate::batch::SFrameRows;

use super::BatchStream;

/// Channel message: Arc-wrapped batch (shared across consumers) or error string.
type BroadcastMsg = Result<Arc<SFrameRows>, String>;

/// Manages broadcast from one source stream to N consumers.
pub(super) struct BroadcastState {
    receivers: Vec<Option<std::sync::mpsc::Receiver<BroadcastMsg>>>,
    next_subscriber: usize,
}

impl BroadcastState {
    /// Create a broadcast from `source` to `num_consumers` consumers.
    ///
    /// Spawns a background driver thread that pulls batches from the source
    /// and sends them (as `Arc<SFrameRows>`) to all consumer channels.
    /// `buffer` is the bounded channel capacity per consumer.
    pub fn new(source: BatchStream, num_consumers: usize, buffer: usize) -> Self {
        assert!(num_consumers >= 2, "broadcast requires at least 2 consumers");

        let mut senders = Vec::with_capacity(num_consumers);
        let mut receivers = Vec::with_capacity(num_consumers);

        for _ in 0..num_consumers {
            let (tx, rx) = std::sync::mpsc::sync_channel::<BroadcastMsg>(buffer);
            senders.push(tx);
            receivers.push(Some(rx));
        }

        // Driver thread: pull from source, fan out to all consumers.
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(_) => return,
            };
            rt.block_on(async {
                use futures::StreamExt;
                let mut source = source;
                while let Some(batch_result) = source.next().await {
                    match batch_result {
                        Ok(batch) => {
                            let shared = Arc::new(batch);
                            for tx in &senders {
                                if tx.send(Ok(shared.clone())).is_err() {
                                    return; // consumer dropped
                                }
                            }
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            for tx in &senders {
                                let _ = tx.send(Err(msg.clone()));
                            }
                            return; // stop on error
                        }
                    }
                }
                // source exhausted; drop senders → receivers see channel closed
            });
        });

        BroadcastState {
            receivers,
            next_subscriber: 0,
        }
    }

    /// Hand out the next consumer stream.
    ///
    /// Each call returns a new `BatchStream` backed by its own channel.
    /// Panics if called more times than `num_consumers`.
    pub fn subscribe(&mut self) -> BatchStream {
        let rx = self.receivers[self.next_subscriber]
            .take()
            .expect("subscriber already taken");
        self.next_subscriber += 1;

        let s = stream::unfold(rx, |rx| async move {
            match rx.recv() {
                Ok(Ok(arc_batch)) => {
                    // Clone the batch out of the Arc for this consumer.
                    let batch = SFrameRows::clone(&arc_batch);
                    Some((Ok(batch), rx))
                }
                Ok(Err(msg)) => {
                    Some((Err(SFrameError::Format(msg)), rx))
                }
                Err(_) => None, // channel closed, source exhausted
            }
        });

        Box::pin(s)
    }
}
