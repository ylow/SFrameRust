# Block-Level Source Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change the source reader from segment-level to batch-level reading so that `head(10)` through a filter reads ~4K rows instead of ~2M rows.

**Architecture:** Replace the background thread's full-segment reads with chunked reads using the existing `read_columns_block_range`. Simplify `unfold_prefetch` since chunks are already batch-sized. Repurpose `source_prefetch_segments` to control prefetch batch count.

**Tech Stack:** Rust, futures::stream, std::sync::mpsc, SegmentReader

---

### Task 1: Rewrite the background thread to read in chunks

**Files:**
- Modify: `crates/sframe-query/src/execute/source.rs`

**Step 1: Add a helper function for chunked segment reading**

Add this function after `read_columns_block_range` (after line 377):

```rust
/// Read a segment in chunks, sending each chunk through the channel.
///
/// Opens the segment once, then reads row ranges of `chunk_size` rows
/// via `read_columns_block_range`, sending each chunk through `tx`.
/// Stops if the receiver is dropped.
fn read_segment_chunked(
    vfs: &dyn VirtualFileSystem,
    segment_path: &str,
    column_types: &[FlexTypeEnum],
    column_indices: Option<&[usize]>,
    local_begin: u64,
    local_end: u64,
    chunk_size: u64,
    tx: &std::sync::mpsc::SyncSender<Result<Vec<Vec<FlexType>>>>,
) -> bool {
    let open_result = (|| -> Result<()> {
        let file = vfs.open_read(segment_path)?;
        let file_size = file.size()?;
        let mut seg_reader = SegmentReader::open(
            Box::new(file),
            file_size,
            column_types.to_vec(),
        )?;

        let mut offset = local_begin;
        while offset < local_end {
            let chunk_end = (offset + chunk_size).min(local_end);
            let result = read_columns_block_range(
                &mut seg_reader,
                offset,
                chunk_end,
                column_indices,
            );
            if tx.send(result).is_err() {
                return Ok(()); // receiver dropped
            }
            offset = chunk_end;
        }
        Ok(())
    })();

    match open_result {
        Ok(()) => true,
        Err(e) => tx.send(Err(e)).is_ok(),
    }
}
```

Returns `true` if the channel is still open, `false` if the receiver was dropped.

**Step 2: Rewrite `compile_sframe_source` to use chunked reading**

Replace the background thread spawn (lines 120-133) and channel setup (lines 115-116) with:

```rust
    let n_prefetch = sframe_config::global().source_prefetch_segments.max(1);

    // Channel carries batch-sized chunks, not full segments.
    let (tx, rx) =
        std::sync::mpsc::sync_channel::<Result<Vec<Vec<FlexType>>>>(n_prefetch);
    let dtypes_bg = dtypes.clone();
    let vfs_bg = vfs.clone();
    let chunk_size = batch_size as u64;

    std::thread::spawn(move || {
        for slice in slices {
            if !read_segment_chunked(
                &*vfs_bg,
                &slice.segment_path,
                &dtypes_bg,
                None,
                slice.local_begin,
                slice.local_end,
                chunk_size,
                &tx,
            ) {
                break;
            }
        }
    });
```

**Step 3: Rewrite `compile_sframe_source_projected` the same way**

Replace its background thread spawn (lines 186-200) with:

```rust
    let n_prefetch = sframe_config::global().source_prefetch_segments.max(1);
    let chunk_size = batch_size as u64;

    let (tx, rx) =
        std::sync::mpsc::sync_channel::<Result<Vec<Vec<FlexType>>>>(n_prefetch);
    let vfs_bg = vfs.clone();

    std::thread::spawn(move || {
        for slice in slices {
            if !read_segment_chunked(
                &*vfs_bg,
                &slice.segment_path,
                &all_col_types,
                Some(&proj_indices),
                slice.local_begin,
                slice.local_end,
                chunk_size,
                &tx,
            ) {
                break;
            }
        }
    });
```

**Step 4: Simplify `PrefetchSourceState` and `unfold_prefetch`**

Chunks are already batch-sized, so no more `seg_data` slicing. Replace:

```rust
/// State for chunk-prefetching source reading.
struct PrefetchSourceState {
    rx: std::sync::mpsc::Receiver<Result<Vec<Vec<FlexType>>>>,
    dtypes: Vec<FlexTypeEnum>,
}
```

Replace `unfold_prefetch`:

```rust
/// Shared unfold logic for prefetched chunk reading.
fn unfold_prefetch(
    state: PrefetchSourceState,
) -> impl Stream<Item = Result<SFrameRows>> {
    stream::unfold(state, |state| async move {
        match state.rx.recv() {
            Err(_) => None, // channel closed, all chunks done
            Ok(Ok(data)) => {
                if data.is_empty() || data.first().map(|c| c.is_empty()).unwrap_or(true) {
                    // Empty chunk — yield empty batch (shouldn't normally happen)
                    Some((Ok(SFrameRows::empty(&state.dtypes)), state))
                } else {
                    let batch = columns_to_batch(&data, &state.dtypes);
                    Some((batch, state))
                }
            }
            Ok(Err(e)) => Some((Err(e), state)),
        }
    })
}

/// Convert column vectors to an SFrameRows batch.
fn columns_to_batch(
    data: &[Vec<FlexType>],
    dtypes: &[FlexTypeEnum],
) -> Result<SFrameRows> {
    let columns: Vec<ColumnData> = data.iter()
        .zip(dtypes.iter())
        .map(|(col_data, &dtype)| ColumnData::from_flex_slice(col_data, dtype))
        .collect();
    SFrameRows::new(columns)
}
```

**Step 5: Update PrefetchSourceState construction in both compile functions**

In `compile_sframe_source` (around line 135):
```rust
    let state = PrefetchSourceState { rx, dtypes };
```

In `compile_sframe_source_projected` (around line 202):
```rust
    let state = PrefetchSourceState { rx, dtypes: projected_dtypes };
```

**Step 6: Remove dead code**

Remove these functions/items that are no longer used:
- `read_segment_row_range` (lines 269-296)
- `read_segment_columns_projected_row_range` (lines 299-326)
- `slice_columns_to_batch` (lines 251-262)

**Step 7: Run tests**

Run: `cargo test --workspace 2>&1 | tail -5`
Expected: All tests pass

**Step 8: Commit**

```
git add crates/sframe-query/src/execute/source.rs
git commit -m "feat: block-level source streaming for early termination

Background thread reads batch-sized chunks via read_columns_block_range
instead of full segments. head(10) through a filter now reads ~4K rows
instead of ~2M rows."
```

---

### Task 2: Verify with the user's benchmark

**Files:** None (verification only)

**Step 1: Run workspace tests**

Run: `cargo test --workspace 2>&1 | tail -5`
Expected: All tests pass, 0 failures

**Step 2: Run clippy**

Run: `cargo clippy --workspace 2>&1 | grep -E "warning|error" | grep -v "generated" | head -20`
Expected: No new warnings from changed code

**Step 3: Check for dead code in source.rs**

Run: `cargo clippy -p sframe-query 2>&1 | grep "source.rs"`
Expected: No unused function warnings

**Step 4: Commit any cleanup if needed**
