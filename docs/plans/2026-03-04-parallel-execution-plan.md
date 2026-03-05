# Parallel Execution Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace unbounded per-worker materialization with streaming segment consumers, making parallel execution memory-bounded.

**Architecture:** Plans always execute sequentially. Data parallelism is orchestrated above the plan level: split plan into N sub-plans, each drives a sequential stream through a segment consumer writing to CacheFs, then assemble the segments into an SFrame.

**Tech Stack:** Rust, rayon (parallelism), tokio (async stream driving), sframe-storage (SegmentWriter, SFrame metadata), sframe-io (CacheFs, VFS)

**Design doc:** `docs/plans/2026-03-04-parallel-execution-design.md`

---

### Task 1: Add `ColumnData::to_flex_vec()`

Converts typed column storage to `Vec<FlexType>` so batch data can be fed to `SegmentWriter::write_column_block()`.

**Files:**
- Modify: `crates/sframe-query/src/batch.rs` (add method after `filter_indices` at ~line 307)

**Step 1: Write the test**

Add to the existing `mod tests` block at the bottom of `batch.rs`:

```rust
#[test]
fn test_to_flex_vec_integer() {
    let col = ColumnData::Integer(vec![Some(1), None, Some(3)]);
    let result = col.to_flex_vec();
    assert_eq!(result, vec![
        FlexType::Integer(1),
        FlexType::Undefined,
        FlexType::Integer(3),
    ]);
}

#[test]
fn test_to_flex_vec_float() {
    let col = ColumnData::Float(vec![Some(1.5), None]);
    let result = col.to_flex_vec();
    assert_eq!(result, vec![FlexType::Float(1.5), FlexType::Undefined]);
}

#[test]
fn test_to_flex_vec_string() {
    let col = ColumnData::String(vec![Some("hello".into()), None]);
    let result = col.to_flex_vec();
    assert_eq!(result, vec![FlexType::String("hello".into()), FlexType::Undefined]);
}

#[test]
fn test_to_flex_vec_flexible() {
    let col = ColumnData::Flexible(vec![FlexType::Integer(1), FlexType::String("x".into())]);
    let result = col.to_flex_vec();
    assert_eq!(result, vec![FlexType::Integer(1), FlexType::String("x".into())]);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p sframe-query --lib batch::tests::test_to_flex_vec -- --nocapture`
Expected: Compilation error — `to_flex_vec` method not found.

**Step 3: Implement `to_flex_vec()`**

Add this method to the `impl ColumnData` block (after `filter_indices`, around line 307):

```rust
/// Convert this column to a `Vec<FlexType>`.
///
/// Used when feeding column data to `SegmentWriter::write_column_block()`.
pub fn to_flex_vec(&self) -> Vec<FlexType> {
    macro_rules! convert_opt {
        ($v:expr, $variant:ident) => {
            $v.iter().map(|val| match val {
                Some(x) => FlexType::$variant(x.clone()),
                None => FlexType::Undefined,
            }).collect()
        };
    }
    match self {
        ColumnData::Integer(v) => convert_opt!(v, Integer),
        ColumnData::Float(v) => convert_opt!(v, Float),
        ColumnData::String(v) => convert_opt!(v, String),
        ColumnData::Vector(v) => convert_opt!(v, Vector),
        ColumnData::List(v) => convert_opt!(v, List),
        ColumnData::Dict(v) => convert_opt!(v, Dict),
        ColumnData::DateTime(v) => convert_opt!(v, DateTime),
        ColumnData::Flexible(v) => v.clone(),
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p sframe-query --lib batch::tests::test_to_flex_vec`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add crates/sframe-query/src/batch.rs
git commit -m "feat: add ColumnData::to_flex_vec() for segment writer integration"
```

---

### Task 2: Add `assemble_sframe_from_segments()`

Public function in sframe-storage that writes SFrame metadata (sidx, frame_idx, dir_archive.ini) given pre-built segment files.

**Files:**
- Modify: `crates/sframe-storage/src/sframe_writer.rs` (add public function at end, before tests)

**Step 1: Write the test**

Create a new test in `crates/sframe-storage/tests/assemble_segments.rs`:

```rust
//! Test: manually write segment files, assemble SFrame metadata, read back.

use sframe_storage::segment_writer::SegmentWriter;
use sframe_storage::sframe_reader::SFrameReader;
use sframe_storage::sframe_writer::assemble_sframe_from_segments;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

#[test]
fn test_assemble_from_two_segments() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().to_str().unwrap();

    // Write segment 0: rows [1, 2, 3]
    let seg0_name = "seg.0000".to_string();
    let seg0_path = format!("{}/{}", base_path, seg0_name);
    let seg0_sizes = {
        let file = std::fs::File::create(&seg0_path).unwrap();
        let mut sw = SegmentWriter::new(std::io::BufWriter::new(file), 2);
        sw.write_column_block(0, &[
            FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3),
        ], FlexTypeEnum::Integer).unwrap();
        sw.write_column_block(1, &[
            FlexType::String("a".into()), FlexType::String("b".into()), FlexType::String("c".into()),
        ], FlexTypeEnum::String).unwrap();
        sw.finish().unwrap()
    };

    // Write segment 1: rows [4, 5]
    let seg1_name = "seg.0001".to_string();
    let seg1_path = format!("{}/{}", base_path, seg1_name);
    let seg1_sizes = {
        let file = std::fs::File::create(&seg1_path).unwrap();
        let mut sw = SegmentWriter::new(std::io::BufWriter::new(file), 2);
        sw.write_column_block(0, &[
            FlexType::Integer(4), FlexType::Integer(5),
        ], FlexTypeEnum::Integer).unwrap();
        sw.write_column_block(1, &[
            FlexType::String("d".into()), FlexType::String("e".into()),
        ], FlexTypeEnum::String).unwrap();
        sw.finish().unwrap()
    };

    // Assemble metadata
    let segment_files = vec![seg0_name, seg1_name];
    let all_segment_sizes = vec![seg0_sizes, seg1_sizes];
    let vfs = sframe_io::local_fs::LocalFileSystem;
    assemble_sframe_from_segments(
        &vfs,
        base_path,
        &["id", "name"],
        &[FlexTypeEnum::Integer, FlexTypeEnum::String],
        &segment_files,
        &all_segment_sizes,
        5,
    ).unwrap();

    // Read back and verify
    let reader = SFrameReader::open(base_path).unwrap();
    assert_eq!(reader.num_rows(), 5);
    assert_eq!(reader.column_names(), &["id", "name"]);

    let col0 = reader.segment_readers[0].read_column(0).unwrap();
    assert_eq!(col0, vec![
        FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3),
    ]);
    let col0_seg1 = reader.segment_readers[1].read_column(0).unwrap();
    assert_eq!(col0_seg1, vec![FlexType::Integer(4), FlexType::Integer(5)]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p sframe-storage --test assemble_segments`
Expected: Compilation error — `assemble_sframe_from_segments` not found.

**Step 3: Implement `assemble_sframe_from_segments()`**

Add to `crates/sframe-storage/src/sframe_writer.rs`, before the `fn segment_filename` helper (around line 768):

```rust
/// Assemble SFrame metadata for pre-built segment files.
///
/// Writes dir_archive.ini, .sidx, .frame_idx, and objects.bin into
/// `base_path`, referencing the given segment files. The segment files
/// must already exist at `{base_path}/{segment_file}`.
///
/// This is the complement to building segments with `SegmentWriter`
/// directly — it creates the metadata that `SFrameReader` needs to
/// open the SFrame.
pub fn assemble_sframe_from_segments(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    column_names: &[&str],
    column_types: &[FlexTypeEnum],
    segment_files: &[String],
    all_segment_sizes: &[Vec<u64>],
    total_rows: u64,
) -> Result<()> {
    let hash = generate_hash(base_path);
    let data_prefix = format!("m_{}", hash);

    vfs.mkdir_p(base_path)?;

    // Rename/copy segment files to canonical names expected by dir_archive.
    // The caller may have used arbitrary file names; we rename them to
    // the data_prefix convention so the sidx/frame_idx references are correct.
    let canonical_files: Vec<String> = segment_files
        .iter()
        .enumerate()
        .map(|(i, original)| {
            let canonical = segment_filename(&data_prefix, i);
            // Only rename if names differ
            if *original != canonical {
                let src = format!("{}/{}", base_path, original);
                let dst = format!("{}/{}", base_path, canonical);
                // Read and rewrite via VFS (works for both local and CacheFs)
                if let Ok(data) = vfs.read_to_bytes(&src) {
                    let _ = vfs.write_bytes(&dst, &data);
                    let _ = vfs.remove(&src);
                }
            }
            canonical
        })
        .collect();

    // Write .sidx
    let sidx_file = format!("{}.sidx", data_prefix);
    let sidx_path = format!("{}/{}", base_path, sidx_file);
    write_sidx(vfs, &sidx_path, &canonical_files, column_types, all_segment_sizes)?;

    // Write .frame_idx
    let frame_idx_file = format!("{}.frame_idx", data_prefix);
    let frame_idx_path = format!("{}/{}", base_path, frame_idx_file);
    let num_segments = segment_files.len();
    write_frame_idx(
        vfs,
        &frame_idx_path,
        column_names,
        &sidx_file,
        total_rows,
        num_segments,
        &std::collections::HashMap::new(),
    )?;

    // Write dir_archive.ini
    write_dir_archive_ini(vfs, base_path, &data_prefix)?;

    // Write empty objects.bin (legacy)
    vfs.write_string(&format!("{}/objects.bin", base_path), "")?;

    Ok(())
}
```

**Wait — the rename/copy approach is fragile.** Simpler alternative: have the caller name segment files using the canonical convention upfront (pass in the `data_prefix`), or just have `assemble_sframe_from_segments` accept arbitrary segment file names and reference them directly in the sidx. Looking at `build_sidx_content`, it takes `segment_files: &[String]` and uses them as-is in the JSON. So we do NOT need to rename. We just need the sidx, frame_idx, and dir_archive to be consistently prefixed. Let me simplify:

```rust
pub fn assemble_sframe_from_segments(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    column_names: &[&str],
    column_types: &[FlexTypeEnum],
    segment_files: &[String],
    all_segment_sizes: &[Vec<u64>],
    total_rows: u64,
) -> Result<()> {
    let hash = generate_hash(base_path);
    let data_prefix = format!("m_{}", hash);

    vfs.mkdir_p(base_path)?;

    // Write .sidx — references segment_files as-is
    let sidx_file = format!("{}.sidx", data_prefix);
    let sidx_path = format!("{}/{}", base_path, sidx_file);
    write_sidx(vfs, &sidx_path, segment_files, column_types, all_segment_sizes)?;

    // Write .frame_idx
    let frame_idx_file = format!("{}.frame_idx", data_prefix);
    let frame_idx_path = format!("{}/{}", base_path, frame_idx_file);
    write_frame_idx(
        vfs,
        &frame_idx_path,
        column_names,
        &sidx_file,
        total_rows,
        segment_files.len(),
        &std::collections::HashMap::new(),
    )?;

    // Write dir_archive.ini
    write_dir_archive_ini(vfs, base_path, &data_prefix)?;

    // Write empty objects.bin
    vfs.write_string(&format!("{}/objects.bin", base_path), "")?;

    Ok(())
}
```

The sidx JSON stores segment file names as provided. `SFrameReader` reads them from the sidx and opens `{base_path}/{segment_file}`. So the caller just needs to use consistent file names between writing segments and calling this function.

**Step 4: Run test to verify it passes**

Run: `cargo test -p sframe-storage --test assemble_segments`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/sframe-storage/src/sframe_writer.rs crates/sframe-storage/tests/assemble_segments.rs
git commit -m "feat: add assemble_sframe_from_segments() for pre-built segment metadata"
```

---

### Task 3: Add `consume_to_segment()`

Drives a `BatchStream` synchronously and writes each batch into a segment file via `SegmentWriter`. This is the core consumer building block.

**Files:**
- Create: `crates/sframe-query/src/execute/consumer.rs`
- Modify: `crates/sframe-query/src/execute/mod.rs` (add `mod consumer;` and re-export)

**Step 1: Write the test**

Add to the bottom of the new `consumer.rs` file:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use crate::batch::{ColumnData, SFrameRows};
    use sframe_storage::segment_reader::SegmentReader;

    #[test]
    fn test_consume_to_segment_basic() {
        // Create two batches
        let batch1 = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1), Some(2)]),
            ColumnData::String(vec![Some("a".into()), Some("b".into())]),
        ]).unwrap();
        let batch2 = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(3)]),
            ColumnData::String(vec![Some("c".into())]),
        ]).unwrap();

        let input: super::super::BatchStream =
            Box::pin(stream::iter(vec![Ok(batch1), Ok(batch2)]));

        let dir = tempfile::tempdir().unwrap();
        let seg_path = dir.path().join("test.segment");
        let file = std::fs::File::create(&seg_path).unwrap();
        let seg_writer = SegmentWriter::new(
            std::io::BufWriter::new(file),
            2, // num_columns
        );
        let dtypes = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];

        let (segment_sizes, total_rows) =
            consume_to_segment(input, seg_writer, &dtypes).unwrap();

        assert_eq!(total_rows, 3);
        assert_eq!(segment_sizes[0], 3); // 3 integers
        assert_eq!(segment_sizes[1], 3); // 3 strings

        // Read back and verify
        let file = std::fs::File::open(&seg_path).unwrap();
        let file_size = file.metadata().unwrap().len();
        let readable = sframe_io::local_fs::LocalReadableFile::new(
            std::io::BufReader::new(file), file_size,
        );
        let mut reader = SegmentReader::open(
            Box::new(readable),
            file_size,
            dtypes,
        ).unwrap();

        let col0 = reader.read_column(0).unwrap();
        assert_eq!(col0, vec![
            FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3),
        ]);
        let col1 = reader.read_column(1).unwrap();
        assert_eq!(col1, vec![
            FlexType::String("a".into()), FlexType::String("b".into()), FlexType::String("c".into()),
        ]);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p sframe-query --lib execute::consumer::tests`
Expected: Compilation error — module and function don't exist.

**Step 3: Implement `consume_to_segment()`**

Create `crates/sframe-query/src/execute/consumer.rs`:

```rust
//! Stream consumers that drive a BatchStream to a destination.

use std::io::Write;

use futures::StreamExt;

use sframe_storage::segment_writer::SegmentWriter;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexTypeEnum;

use super::BatchStream;

/// Drive a `BatchStream` and write all output into a single segment file.
///
/// Each batch's columns are written as blocks into the `SegmentWriter`.
/// Blocks from different columns are interleaved in the file (all columns
/// for batch 0, then all columns for batch 1, etc.), which is correct
/// because `SegmentReader` uses the block index footer for seeking.
///
/// Returns `(segment_sizes, total_rows)` where `segment_sizes` is the
/// per-column element count vector from `SegmentWriter::finish()`.
pub fn consume_to_segment<W: Write>(
    stream: BatchStream,
    mut seg_writer: SegmentWriter<W>,
    dtypes: &[FlexTypeEnum],
) -> Result<(Vec<u64>, u64)> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| SFrameError::Format(format!("Failed to create tokio runtime: {}", e)))?;

    let mut total_rows: u64 = 0;

    rt.block_on(async {
        let mut stream = stream;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let n = batch.num_rows();
            if n == 0 {
                continue;
            }
            for (col_idx, col) in batch.columns().iter().enumerate() {
                let values = col.to_flex_vec();
                seg_writer.write_column_block(col_idx, &values, dtypes[col_idx])?;
            }
            total_rows += n as u64;
        }
        Ok::<(), sframe_types::error::SFrameError>(())
    })?;

    let segment_sizes = seg_writer.finish()?;
    Ok((segment_sizes, total_rows))
}
```

Add to `crates/sframe-query/src/execute/mod.rs` after line 18 (`mod transform;`):

```rust
mod consumer;
```

And add the public re-export alongside the existing ones (around line 36):

```rust
pub use consumer::consume_to_segment;
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p sframe-query --lib execute::consumer::tests`
Expected: PASS. If `LocalReadableFile` is not public, adjust the test to use the VFS `open_read` instead:

```rust
let readable = sframe_io::local_fs::LocalFileSystem.open_read(seg_path.to_str().unwrap()).unwrap();
```

**Step 5: Commit**

```bash
git add crates/sframe-query/src/execute/consumer.rs crates/sframe-query/src/execute/mod.rs
git commit -m "feat: add consume_to_segment() stream consumer"
```

---

### Task 4: Simplify `compile()` — remove parallel path

**Files:**
- Modify: `crates/sframe-query/src/execute/mod.rs:49-60`

**Step 1: Simplify `compile()`**

Replace the current `compile()` function (lines 49-60) with:

```rust
/// Compile a logical plan node into a BatchStream.
///
/// Always compiles single-threaded. Data parallelism is handled by
/// the caller via `execute_parallel()` in the parallel module.
pub fn compile(node: &Arc<PlannerNode>) -> Result<BatchStream> {
    let node = optimizer::optimize(node);
    compile_single_threaded(&node)
}
```

**Step 2: Run all existing tests**

Run: `cargo test -p sframe-query`
Expected: All existing tests PASS. The tests in `execute::tests` and `parallel::tests` don't rely on the parallel path being triggered from `compile()` — they test the plan structure checks and the execution operators independently.

**Step 3: Commit**

```bash
git add crates/sframe-query/src/execute/mod.rs
git commit -m "refactor: make compile() always sequential, remove parallel dispatch"
```

---

### Task 5: Rewrite `execute_parallel()` with segment consumers

Replace the old `compile_parallel()` (which materialized everything) with the new `execute_parallel()` that writes segments to CacheFs.

**Files:**
- Modify: `crates/sframe-query/src/execute/parallel.rs`
- Modify: `crates/sframe-query/src/execute/mod.rs` (update re-exports)

**Step 1: Write the integration test**

Add to the bottom of `parallel.rs`, inside the existing `mod tests` block:

```rust
#[test]
fn test_execute_parallel_roundtrip() {
    // Use the sample business.sf
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = format!("{}/../../samples/business.sf", manifest);
    let reader = sframe_storage::sframe_reader::SFrameReader::open(&path).unwrap();
    let col_names: Vec<String> = reader.column_names().to_vec();
    let col_types: Vec<FlexTypeEnum> = reader
        .group_index
        .columns
        .iter()
        .map(|c| c.dtype)
        .collect();
    let num_rows = reader.num_rows();

    let source = PlannerNode::sframe_source(&path, col_names.clone(), col_types.clone(), num_rows);

    // Execute in parallel
    let result_path = execute_parallel(&source, num_rows, &col_names, &col_types).unwrap();

    // Read back from CacheFs and verify
    let cache_fs = sframe_io::cache_fs::global_cache_fs();
    let vfs = sframe_io::vfs::ArcCacheFsVfs(cache_fs.clone());
    let result_reader = sframe_storage::sframe_reader::SFrameReader::open_with_fs(
        &vfs, &result_path,
    ).unwrap();

    assert_eq!(result_reader.num_rows(), num_rows);
    assert_eq!(result_reader.column_names(), reader.column_names());

    // Verify first segment has data
    let col0 = result_reader.segment_readers[0].read_column(0).unwrap();
    assert!(!col0.is_empty());

    // Clean up
    cache_fs.remove_dir(&result_path).unwrap();
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p sframe-query --lib execute::parallel::tests::test_execute_parallel_roundtrip`
Expected: Compilation error — `execute_parallel` not found.

**Step 3: Rewrite `parallel.rs`**

Replace the `compile_parallel` function and update imports. Keep `parallel_slice_row_count`, `check_sliceable`, and existing tests. Replace `compile_parallel` with `execute_parallel`:

```rust
//! Data-parallel query execution via input slicing.
//!
//! Divides a plan's input rows across N worker threads, where each
//! worker runs the full operator pipeline on its slice independently,
//! writing output to a segment file on CacheFs. The segments are
//! assembled into an SFrame that can be read back as a source.

use std::sync::Arc;

use rayon::prelude::*;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::vfs::ArcCacheFsVfs;
use sframe_storage::segment_writer::SegmentWriter;
use sframe_storage::sframe_writer::assemble_sframe_from_segments;
use sframe_types::error::Result;
use sframe_types::flex_type::FlexTypeEnum;

use crate::planner::{clone_plan_with_row_range, LogicalOp, PlannerNode};

use super::consumer::consume_to_segment;

/// Minimum row count to justify parallel execution.
const MIN_ROWS_FOR_PARALLEL: u64 = 10_000;

// ... keep parallel_slice_row_count() and check_sliceable() unchanged ...

/// Execute a plan in parallel by slicing input rows across workers.
///
/// Each worker compiles and runs its slice sequentially, writing output
/// to a segment file on CacheFs. After all workers finish, the segments
/// are assembled into an SFrame on CacheFs.
///
/// Returns the CacheFs SFrame path. The caller is responsible for
/// cleanup via `cache_fs.remove_dir()` when the result is no longer needed.
pub fn execute_parallel(
    plan: &Arc<PlannerNode>,
    total_rows: u64,
    column_names: &[String],
    dtypes: &[FlexTypeEnum],
) -> Result<String> {
    let n_workers = rayon::current_num_threads().max(1);

    // Build N plans with row-range-scoped sources
    let worker_plans: Vec<Arc<PlannerNode>> = (0..n_workers)
        .filter_map(|i| {
            let begin = (i as u64 * total_rows) / n_workers as u64;
            let end = ((i as u64 + 1) * total_rows) / n_workers as u64;
            if begin >= end {
                return None;
            }
            Some(clone_plan_with_row_range(plan, begin, end))
        })
        .collect();

    let cache_fs = global_cache_fs();
    let base_path = cache_fs.alloc_dir();
    let vfs = Arc::new(ArcCacheFsVfs(cache_fs.clone()));

    vfs.mkdir_p(&base_path)?;

    // Each worker writes a segment file and returns metadata.
    let worker_results: Vec<Result<(String, Vec<u64>, u64)>> = worker_plans
        .into_par_iter()
        .enumerate()
        .map(|(i, plan)| {
            let seg_name = format!("seg.{:04}", i);
            let seg_path = format!("{}/{}", base_path, seg_name);
            let file = vfs.open_write(&seg_path)?;
            let seg_writer = SegmentWriter::new(file, dtypes.len());

            let stream = super::compile_single_threaded(&plan)?;
            let (segment_sizes, row_count) = consume_to_segment(stream, seg_writer, dtypes)?;

            Ok((seg_name, segment_sizes, row_count))
        })
        .collect();

    // Collect results, propagating any worker error.
    let mut segment_files = Vec::new();
    let mut all_segment_sizes = Vec::new();
    let mut total_written: u64 = 0;
    for result in worker_results {
        let (seg_name, sizes, rows) = result?;
        segment_files.push(seg_name);
        all_segment_sizes.push(sizes);
        total_written += rows;
    }

    // Assemble SFrame metadata
    let col_name_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
    assemble_sframe_from_segments(
        &*vfs,
        &base_path,
        &col_name_refs,
        dtypes,
        &segment_files,
        &all_segment_sizes,
        total_written,
    )?;

    Ok(base_path)
}
```

Update visibility in `mod.rs`: change `parallel` from `mod parallel;` to `pub mod parallel;` (or add `pub use parallel::{parallel_slice_row_count, execute_parallel};` to the re-exports).

**Step 4: Run tests**

Run: `cargo test -p sframe-query --lib execute::parallel::tests`
Expected: All tests PASS (existing `check_sliceable` tests + new roundtrip test).

**Step 5: Run the full workspace test suite**

Run: `cargo test --workspace`
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add crates/sframe-query/src/execute/parallel.rs crates/sframe-query/src/execute/mod.rs
git commit -m "feat: rewrite execute_parallel to stream through segment consumers on CacheFs"
```

---

### Task 6: Verify end-to-end with full test suite

No code changes. Verify everything works together.

**Step 1: Run the full workspace tests**

Run: `cargo test --workspace`
Expected: All tests PASS.

**Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings.

**Step 3: Remove dead code**

Check if the old `compile_parallel` references or unused imports remain. Clean up any dead code flagged by clippy or compiler warnings.

**Step 4: Final commit if any cleanup was needed**

```bash
git add -u
git commit -m "chore: clean up dead code from parallel execution rewrite"
```

---

## Notes for Implementer

**Key files to read before starting:**
- Design doc: `docs/plans/2026-03-04-parallel-execution-design.md`
- Current parallel.rs: `crates/sframe-query/src/execute/parallel.rs`
- SegmentWriter API: `crates/sframe-storage/src/segment_writer.rs`
- SFrame metadata format: `crates/sframe-storage/src/sframe_writer.rs` (the `build_*` functions)
- CacheFs API: `crates/sframe-io/src/cache_fs.rs` (alloc_dir, remove_dir, CacheGuard)
- VFS trait: `crates/sframe-io/src/vfs.rs` (ArcCacheFsVfs)

**What is NOT in scope (deferred to callers):**
- Updating `SFrame::save()`, `SFrame::materialize()`, or `SArray` to use `execute_parallel()`. That's a separate task where each caller decides its own parallelism strategy.

**Test data:**
- `samples/business.sf` — 11,536 rows, 12 columns. Used for integration tests.

**VFS note:**
- `ArcCacheFsVfs` routes `open_write` through `open_cache_write`, which buffers in memory first and may spill to disk based on cache capacity settings. This is the right VFS for CacheFs segment writing.

**SegmentWriter generic parameter:**
- `SegmentWriter<W: Write>` is generic over the writer. For CacheFs, use `SegmentWriter<Box<dyn WritableFile>>` since `vfs.open_write()` returns `Box<dyn WritableFile>`. `WritableFile` implements `Write`.
