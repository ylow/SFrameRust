# EC Sort Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement External Columnar Sort (ec_sort) — a sort algorithm that separates key sorting from value permutation, enabling efficient sorting of wide SFrames.

**Architecture:** Uses existing external sort to sort key columns and generate a forward_map (SArray of integers on CacheFs). Then `permute_sframe` scatters value columns into buckets via a new `ScatterWriter`, then permutes each bucket in-memory and writes the sorted output. See `docs/plans/2026-03-07-ec-sort-design.md` for full design.

**Tech Stack:** Rust, existing sframe-storage/sframe-io/sframe-query crates, CacheFs, rayon for parallel permute phase.

---

## Task 1: Make `estimate_bytes_per_value` pub(crate)

**Files:**
- Modify: `crates/sframe-storage/src/segment_writer.rs:184`

**Step 1: Change visibility**

Change `fn estimate_bytes_per_value` from private to `pub(crate)`:

```rust
pub(crate) fn estimate_bytes_per_value(dtype: FlexTypeEnum) -> usize {
```

Also export the block sizing constants needed by ScatterWriter:

```rust
pub(crate) const TARGET_BLOCK_SIZE: usize = 64 * 1024;
pub(crate) const MIN_ROWS_PER_BLOCK: usize = 8;
pub(crate) const MAX_ROWS_PER_BLOCK: usize = 256 * 1024;
```

**Step 2: Verify it compiles**

Run: `cargo build -p sframe-storage`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add crates/sframe-storage/src/segment_writer.rs
git commit -m "refactor: make estimate_bytes_per_value and block sizing constants pub(crate)"
```

---

## Task 2: ScatterWriter — struct and constructor

**Files:**
- Create: `crates/sframe-storage/src/scatter_writer.rs`
- Modify: `crates/sframe-storage/src/lib.rs` (add `pub mod scatter_writer;`)

The ScatterWriter manages M segment files. It supports writing values to
specific (column, segment) pairs. Columns must be written in order (0, 1, 2, ...)
so that blocks in each segment file are in column order, matching the existing
segment format.

**Step 1: Write the struct and constructor**

```rust
//! Scatter writer for distributing column data across multiple segments.
//!
//! Used by ec_sort to scatter values into buckets based on a forward map.
//! Columns must be written in sequential order (0, 1, 2, ...) so that
//! blocks in each segment file are in column order.

use std::io::Write;

use sframe_io::vfs::{VirtualFileSystem, WritableFile};
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::segment_writer::{
    SegmentWriter, estimate_bytes_per_value,
    TARGET_BLOCK_SIZE, MIN_ROWS_PER_BLOCK, MAX_ROWS_PER_BLOCK,
};
use crate::sframe_writer::segment_filename;

/// Result of finishing a ScatterWriter.
pub struct ScatterResult {
    /// Segment file names (relative to base_path).
    pub segment_files: Vec<String>,
    /// Per-segment, per-column row counts: `all_segment_sizes[seg][col]`.
    pub all_segment_sizes: Vec<Vec<u64>>,
}

/// Writes column data to M segment files, distributing values across segments
/// based on caller-determined bucket assignment.
///
/// Columns must be written in order: all values for column 0 first, then
/// column 1, etc. This ensures blocks in each segment file are in column
/// order, compatible with the existing SegmentReader.
pub struct ScatterWriter {
    segment_writers: Vec<SegmentWriter<Box<dyn WritableFile>>>,
    segment_files: Vec<String>,
    column_types: Vec<FlexTypeEnum>,
    num_segments: usize,
    num_columns: usize,

    /// Per-segment buffer for the active column.
    buffers: Vec<Vec<FlexType>>,
    /// Rows per block for the active column (based on type estimate).
    rows_per_block: usize,
    /// Type of the active column.
    active_dtype: FlexTypeEnum,
    /// Index of the active column (for validation).
    active_column: Option<usize>,

    /// Per-segment, per-column row counts for metadata.
    /// Layout: column_counts[segment][column] = count.
    column_counts: Vec<Vec<u64>>,
}

impl ScatterWriter {
    /// Create a new ScatterWriter with `num_segments` segment files.
    ///
    /// Segment files are created at `{base_path}/{data_prefix}.NNNN`.
    pub fn new(
        vfs: &dyn VirtualFileSystem,
        base_path: &str,
        data_prefix: &str,
        column_types: &[FlexTypeEnum],
        num_segments: usize,
    ) -> Result<Self> {
        let num_columns = column_types.len();

        let mut segment_writers = Vec::with_capacity(num_segments);
        let mut segment_files = Vec::with_capacity(num_segments);

        for seg_idx in 0..num_segments {
            let seg_file = segment_filename(data_prefix, seg_idx);
            let seg_path = format!("{base_path}/{seg_file}");
            let file = vfs.open_write(&seg_path)?;
            segment_writers.push(SegmentWriter::new(file, num_columns));
            segment_files.push(seg_file);
        }

        let column_counts = vec![vec![0u64; num_columns]; num_segments];

        Ok(ScatterWriter {
            segment_writers,
            segment_files,
            column_types: column_types.to_vec(),
            num_segments,
            num_columns,
            buffers: Vec::new(),
            rows_per_block: 0,
            active_dtype: FlexTypeEnum::Undefined,
            active_column: None,
            column_counts,
        })
    }
}
```

**Step 2: Add module to lib.rs**

Add to `crates/sframe-storage/src/lib.rs`:

```rust
pub mod scatter_writer;
```

**Step 3: Verify it compiles**

Run: `cargo build -p sframe-storage`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add crates/sframe-storage/src/scatter_writer.rs crates/sframe-storage/src/lib.rs
git commit -m "feat(scatter_writer): add ScatterWriter struct and constructor"
```

---

## Task 3: ScatterWriter — write and flush methods

**Files:**
- Modify: `crates/sframe-storage/src/scatter_writer.rs`

**Step 1: Add the write/flush/finish methods**

```rust
impl ScatterWriter {
    // ... (constructor from Task 2)

    /// Begin writing a new column. Must be called in order (0, 1, 2, ...).
    /// If a previous column was active, its remaining buffers are flushed.
    fn begin_column(&mut self, column_id: usize) -> Result<()> {
        // Flush previous column if any
        if let Some(prev_col) = self.active_column {
            if column_id <= prev_col {
                return Err(SFrameError::Format(format!(
                    "ScatterWriter: columns must be written in order, got {column_id} after {prev_col}"
                )));
            }
            self.flush_active_column()?;
        }

        let dtype = self.column_types[column_id];
        let est = estimate_bytes_per_value(dtype).max(1);
        let rpb = (TARGET_BLOCK_SIZE / est).clamp(MIN_ROWS_PER_BLOCK, MAX_ROWS_PER_BLOCK);

        self.buffers = (0..self.num_segments).map(|_| Vec::new()).collect();
        self.rows_per_block = rpb;
        self.active_dtype = dtype;
        self.active_column = Some(column_id);
        Ok(())
    }

    /// Write a single value to a specific segment for the active column.
    pub fn write_to_segment(&mut self, column_id: usize, segment_id: usize, value: FlexType) -> Result<()> {
        // Auto-begin column if needed
        if self.active_column != Some(column_id) {
            self.begin_column(column_id)?;
        }

        self.buffers[segment_id].push(value);

        if self.buffers[segment_id].len() >= self.rows_per_block {
            self.flush_segment_buffer(column_id, segment_id)?;
        }

        Ok(())
    }

    /// Flush remaining values for the active column across all segments.
    pub fn flush_column(&mut self, column_id: usize) -> Result<()> {
        if self.active_column != Some(column_id) {
            return Err(SFrameError::Format(format!(
                "ScatterWriter: flush_column({column_id}) but active column is {:?}",
                self.active_column
            )));
        }
        self.flush_active_column()?;
        self.active_column = None;
        Ok(())
    }

    /// Finish all segment writers and return metadata.
    pub fn finish(mut self) -> Result<ScatterResult> {
        // Flush any remaining active column
        if self.active_column.is_some() {
            self.flush_active_column()?;
        }

        let mut all_segment_sizes = Vec::with_capacity(self.num_segments);
        for seg_writer in self.segment_writers {
            let sizes = seg_writer.finish()?;
            all_segment_sizes.push(sizes);
        }

        Ok(ScatterResult {
            segment_files: self.segment_files,
            all_segment_sizes,
        })
    }

    /// Flush the buffer for one segment of the active column.
    fn flush_segment_buffer(&mut self, column_id: usize, segment_id: usize) -> Result<()> {
        let buf = std::mem::take(&mut self.buffers[segment_id]);
        if buf.is_empty() {
            return Ok(());
        }
        let count = buf.len() as u64;
        self.segment_writers[segment_id].write_column_block(
            column_id,
            &buf,
            self.active_dtype,
        )?;
        self.column_counts[segment_id][column_id] += count;
        Ok(())
    }

    /// Flush all segment buffers for the active column.
    fn flush_active_column(&mut self) -> Result<()> {
        if let Some(col) = self.active_column {
            for seg in 0..self.num_segments {
                self.flush_segment_buffer(col, seg)?;
            }
        }
        Ok(())
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p sframe-storage`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add crates/sframe-storage/src/scatter_writer.rs
git commit -m "feat(scatter_writer): add write_to_segment, flush_column, finish methods"
```

---

## Task 4: ScatterWriter — unit tests

**Files:**
- Modify: `crates/sframe-storage/src/scatter_writer.rs` (add `#[cfg(test)]` module)

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_reader::SegmentReader;
    use std::io::Cursor;

    /// Helper: create an in-memory VFS-like setup for testing.
    /// Instead of VFS, write to temp files.
    fn test_scatter_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().to_str().unwrap();
        let vfs = sframe_io::local_fs::LocalFileSystem;
        std::fs::create_dir_all(base_path).unwrap();

        let column_types = &[FlexTypeEnum::Integer, FlexTypeEnum::String];
        let num_segments = 3;
        let data_prefix = "m_test";

        let mut writer = ScatterWriter::new(
            &vfs, base_path, data_prefix, column_types, num_segments,
        ).unwrap();

        // Write column 0 (integers): values 0-8, distributed across 3 segments
        for i in 0..9 {
            let seg = i % 3;  // round-robin distribution
            writer.write_to_segment(0, seg, FlexType::Integer(i as i64)).unwrap();
        }
        writer.flush_column(0).unwrap();

        // Write column 1 (strings): same distribution
        for i in 0..9 {
            let seg = i % 3;
            writer.write_to_segment(1, seg, FlexType::String(format!("s{i}").into())).unwrap();
        }
        writer.flush_column(1).unwrap();

        let result = writer.finish().unwrap();

        assert_eq!(result.segment_files.len(), 3);
        assert_eq!(result.all_segment_sizes.len(), 3);

        // Each segment should have 3 rows per column
        for seg_sizes in &result.all_segment_sizes {
            assert_eq!(seg_sizes[0], 3); // 3 integers
            assert_eq!(seg_sizes[1], 3); // 3 strings
        }

        // Verify segment 0 contains values 0, 3, 6 for column 0
        let seg0_path = format!("{base_path}/{}", result.segment_files[0]);
        let file = std::fs::File::open(&seg0_path).unwrap();
        let file_size = file.metadata().unwrap().len();
        let mut reader = SegmentReader::open(
            Box::new(file), file_size, column_types.to_vec(),
        ).unwrap();

        let col0_values = reader.read_column(0).unwrap();
        assert_eq!(col0_values, vec![
            FlexType::Integer(0), FlexType::Integer(3), FlexType::Integer(6)
        ]);

        let col1_values = reader.read_column(1).unwrap();
        assert_eq!(col1_values, vec![
            FlexType::String("s0".into()),
            FlexType::String("s3".into()),
            FlexType::String("s6".into()),
        ]);
    }

    #[test]
    fn test_scatter_empty_segments() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().to_str().unwrap();
        let vfs = sframe_io::local_fs::LocalFileSystem;
        std::fs::create_dir_all(base_path).unwrap();

        let column_types = &[FlexTypeEnum::Integer];
        let mut writer = ScatterWriter::new(
            &vfs, base_path, "m_test", column_types, 3,
        ).unwrap();

        // Write all values to segment 1 only
        for i in 0..5 {
            writer.write_to_segment(0, 1, FlexType::Integer(i)).unwrap();
        }
        writer.flush_column(0).unwrap();

        let result = writer.finish().unwrap();

        // Segment 0 and 2 should have 0 rows
        assert_eq!(result.all_segment_sizes[0][0], 0);
        assert_eq!(result.all_segment_sizes[1][0], 5);
        assert_eq!(result.all_segment_sizes[2][0], 0);
    }

    #[test]
    fn test_scatter_column_order_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().to_str().unwrap();
        let vfs = sframe_io::local_fs::LocalFileSystem;
        std::fs::create_dir_all(base_path).unwrap();

        let column_types = &[FlexTypeEnum::Integer, FlexTypeEnum::Integer];
        let mut writer = ScatterWriter::new(
            &vfs, base_path, "m_test", column_types, 2,
        ).unwrap();

        // Write column 1 first — should work (no column 0 yet)
        writer.write_to_segment(1, 0, FlexType::Integer(1)).unwrap();
        writer.flush_column(1).unwrap();

        // Try to write column 0 after column 1 — should fail
        let result = writer.write_to_segment(0, 0, FlexType::Integer(0));
        assert!(result.is_err());
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p sframe-storage scatter_writer`
Expected: All tests pass

**Step 3: Commit**

```bash
git add crates/sframe-storage/src/scatter_writer.rs
git commit -m "test(scatter_writer): add unit tests for ScatterWriter"
```

---

## Task 5: ec_sort module — scaffold and permute_sframe signature

**Files:**
- Create: `crates/sframe/src/ec_sort.rs`
- Modify: `crates/sframe/src/lib.rs` (add `mod ec_sort;`)

**Step 1: Create the module with public API signatures**

```rust
//! External Columnar Sort (EC Sort).
//!
//! Sorts SFrames by separating key sorting from value column permutation.
//! For wide SFrames with many or large value columns, this avoids the
//! memory overhead of sorting entire rows.
//!
//! See `docs/plans/2026-03-07-ec-sort-design.md` for the full design.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_query::algorithms::sort::{self, SortKey, SortOrder};
use sframe_query::execute::compile;
use sframe_storage::scatter_writer::{ScatterResult, ScatterWriter};
use sframe_storage::segment_reader::SegmentReader;
use sframe_storage::segment_writer::BufferedSegmentWriter;
use sframe_storage::sframe_writer::{assemble_sframe_from_segments, generate_hash, segment_filename};
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::sarray::SArray;
use crate::sframe::{AnonymousStore, SFrame, SFrameBuilder};

const CHUNK_SIZE: usize = 8192;

/// Permute an SFrame's rows according to a forward map.
///
/// `forward_map[i] = j` means input row `i` is written to output row `j`.
/// The forward_map must contain a permutation of `[0..N)` where N is
/// the number of rows in the input SFrame.
pub(crate) fn permute_sframe(
    input: &SFrame,
    forward_map: &SArray,
) -> Result<SFrame> {
    todo!("Implemented in Tasks 6-8")
}

/// External Columnar Sort.
///
/// Sorts `input` by the given key columns using the EC Sort algorithm:
/// 1. Sort key columns with row numbers → inverse_map + sorted keys
/// 2. Invert the permutation → forward_map
/// 3. Permute value columns using forward_map
/// 4. Assemble sorted keys + sorted values
pub(crate) fn ec_sort(
    input: &SFrame,
    key_column_indices: &[usize],
    sort_orders: &[bool],
) -> Result<SFrame> {
    todo!("Implemented in Task 9")
}
```

**Step 2: Add module to lib.rs**

In `crates/sframe/src/lib.rs`, add:

```rust
mod ec_sort;
```

(Keep it private for now; it's called from within the crate.)

**Step 3: Verify it compiles**

Run: `cargo build -p sframe`
Expected: SUCCESS (functions are `todo!()` but that's fine)

**Step 4: Commit**

```bash
git add crates/sframe/src/ec_sort.rs crates/sframe/src/lib.rs
git commit -m "feat(ec_sort): scaffold ec_sort module with API signatures"
```

---

## Task 6: Scatter phase — column size estimation and bucket count

**Files:**
- Modify: `crates/sframe/src/ec_sort.rs`

This adds helper functions that `permute_sframe` will use.

**Step 1: Implement column size estimation and bucket count**

```rust
use sframe_storage::segment_writer::estimate_bytes_per_value;

/// Estimate bytes per value for each column.
///
/// Uses the type-based estimate from segment_writer as a starting point.
/// For a more accurate estimate, we could inspect block metadata, but
/// the type-based estimate is sufficient for bucket sizing.
fn estimate_column_bytes(column_types: &[FlexTypeEnum]) -> Vec<usize> {
    column_types
        .iter()
        .map(|&dt| estimate_bytes_per_value(dt))
        .collect()
}

/// Determine the number of buckets for the scatter phase.
///
/// Ensures each bucket's largest column fits in `budget_per_thread` bytes.
fn compute_num_buckets(
    num_rows: u64,
    column_bytes: &[usize],
    budget_per_thread: usize,
) -> usize {
    if num_rows == 0 || column_bytes.is_empty() {
        return 1;
    }

    let max_col_bytes = column_bytes
        .iter()
        .map(|&b| b as u64 * num_rows)
        .max()
        .unwrap_or(0);

    let half_budget = (budget_per_thread / 2).max(1) as u64;
    let mut num_buckets = ((max_col_bytes + half_budget - 1) / half_budget) as usize;
    num_buckets = num_buckets.max(1);
    num_buckets *= rayon::current_num_threads().max(1);

    // Don't create more buckets than rows
    if num_buckets as u64 > num_rows {
        num_buckets = 1;
    }

    num_buckets
}
```

**Step 2: Write a test for bucket count computation**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_num_buckets() {
        // Small dataset: should get 1 bucket (fewer rows than buckets)
        let n = compute_num_buckets(10, &[8], 1024 * 1024);
        assert_eq!(n, 1);

        // Large dataset with small values
        let n = compute_num_buckets(
            1_000_000,
            &[8, 32],  // int and string columns
            4 * 1024 * 1024, // 4MB budget
        );
        assert!(n >= 1);
        // With 32 bytes/value * 1M rows = 32MB max column
        // half_budget = 2MB, so 32MB/2MB = 16 buckets * num_threads
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p sframe ec_sort::tests`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/sframe/src/ec_sort.rs
git commit -m "feat(ec_sort): add column size estimation and bucket count computation"
```

---

## Task 7: Scatter phase implementation

**Files:**
- Modify: `crates/sframe/src/ec_sort.rs`

This is the core of the scatter phase. For each column, read the forward_map
and column data in lockstep, and scatter values to the appropriate bucket
segment via ScatterWriter.

**Step 1: Implement the scatter function**

Key design: For each value column, create a 2-column SFrame combining
`[forward_map, column]` and stream it. This keeps the forward_map and column
data synchronized without manual reader management.

```rust
use sframe_query::execute::compile;

/// Scatter phase: distribute column values into M bucket segments
/// based on the forward_map.
///
/// The forward_map is also scattered as the last column (N-th column)
/// so the permute phase knows the per-bucket ordering.
///
/// Returns the ScatterResult with segment file locations and row counts.
fn scatter_columns(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
    input: &SFrame,
    forward_map: &SArray,
    num_buckets: usize,
    rows_per_bucket: u64,
) -> Result<ScatterResult> {
    let num_input_cols = input.num_columns();
    // Output has num_input_cols + 1 columns (last is per-bucket forward_map)
    let mut output_types: Vec<FlexTypeEnum> = input.column_types();
    output_types.push(FlexTypeEnum::Integer); // forward_map column

    let mut writer = ScatterWriter::new(
        vfs, base_path, data_prefix, &output_types, num_buckets,
    )?;

    // Scatter each input column using the forward_map
    for col_idx in 0..num_input_cols {
        scatter_one_column(
            &mut writer, input, forward_map,
            col_idx, num_buckets, rows_per_bucket,
        )?;
    }

    // Scatter the forward_map itself as the last column.
    // Store the raw forward_map value (the permute phase subtracts bucket_start).
    scatter_forward_map_column(
        &mut writer, forward_map,
        num_input_cols, num_buckets, rows_per_bucket,
    )?;

    writer.finish()
}

/// Scatter one value column into buckets.
fn scatter_one_column(
    writer: &mut ScatterWriter,
    input: &SFrame,
    forward_map: &SArray,
    col_idx: usize,
    num_buckets: usize,
    rows_per_bucket: u64,
) -> Result<()> {
    // Build a 2-column SFrame: [forward_map, value_column]
    let value_col = input.columns()[col_idx].clone();
    let combined = SFrame::new_with_columns(
        vec![forward_map.clone(), value_col],
        vec!["__fmap__".to_string(), "__val__".to_string()],
    );

    let mut stream = combined.compile_stream()?;
    while let Some(batch_result) = stream.next_batch() {
        let batch = batch_result?;
        let fmap_col = batch.column(0);
        let val_col = batch.column(1);
        for r in 0..batch.num_rows() {
            let output_row = match fmap_col.get(r) {
                FlexType::Integer(v) => v as u64,
                _ => return Err(SFrameError::Type("forward_map must be Integer".into())),
            };
            let seg = (output_row / rows_per_bucket) as usize;
            let seg = seg.min(num_buckets - 1);
            writer.write_to_segment(col_idx, seg, val_col.get(r))?;
        }
    }
    writer.flush_column(col_idx)?;
    Ok(())
}

/// Scatter the forward_map values as the last column.
fn scatter_forward_map_column(
    writer: &mut ScatterWriter,
    forward_map: &SArray,
    output_col_idx: usize,
    num_buckets: usize,
    rows_per_bucket: u64,
) -> Result<()> {
    let fmap_sf = SFrame::new_with_columns(
        vec![forward_map.clone()],
        vec!["__fmap__".to_string()],
    );

    let mut stream = fmap_sf.compile_stream()?;
    while let Some(batch_result) = stream.next_batch() {
        let batch = batch_result?;
        let fmap_col = batch.column(0);
        for r in 0..batch.num_rows() {
            let output_row = match fmap_col.get(r) {
                FlexType::Integer(v) => v as u64,
                _ => return Err(SFrameError::Type("forward_map must be Integer".into())),
            };
            let seg = (output_row / rows_per_bucket) as usize;
            let seg = seg.min(num_buckets - 1);
            writer.write_to_segment(
                output_col_idx, seg, FlexType::Integer(output_row as i64),
            )?;
        }
    }
    writer.flush_column(output_col_idx)?;
    Ok(())
}
```

**Step 2: Write a test for the scatter phase**

```rust
#[cfg(test)]
mod tests {
    // ... (existing tests)

    #[test]
    fn test_scatter_columns() {
        // Create a small input SFrame: 2 columns, 6 rows
        let col0 = SArray::from_vec(
            (0..6).map(|i| FlexType::Integer(i * 10)).collect(),
            FlexTypeEnum::Integer,
        ).unwrap();
        let col1 = SArray::from_vec(
            (0..6).map(|i| FlexType::String(format!("r{i}").into())).collect(),
            FlexTypeEnum::String,
        ).unwrap();
        let input = SFrame::new_with_columns(
            vec![col0, col1],
            vec!["a".to_string(), "b".to_string()],
        );

        // Forward map: reverse order [5, 4, 3, 2, 1, 0]
        let forward_map = SArray::from_vec(
            (0..6).rev().map(|i| FlexType::Integer(i)).collect(),
            FlexTypeEnum::Integer,
        ).unwrap();

        let cache_fs = global_cache_fs();
        let base_path = cache_fs.alloc_dir();
        let vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
        vfs.mkdir_p(&base_path).unwrap();

        let result = scatter_columns(
            &*vfs, &base_path, "m_test",
            &input, &forward_map,
            2,   // 2 buckets
            3,   // 3 rows per bucket
        ).unwrap();

        assert_eq!(result.segment_files.len(), 2);

        // Forward map is [5,4,3,2,1,0]:
        // Row 0 → output 5 → bucket 1 (5/3=1)
        // Row 1 → output 4 → bucket 1 (4/3=1)
        // Row 2 → output 3 → bucket 1 (3/3=1)
        // Row 3 → output 2 → bucket 0 (2/3=0)
        // Row 4 → output 1 → bucket 0 (1/3=0)
        // Row 5 → output 0 → bucket 0 (0/3=0)
        // Bucket 0 gets rows [3,4,5], bucket 1 gets rows [0,1,2]

        // Verify bucket 0 has values from input rows 3,4,5
        let seg0_path = format!("{base_path}/{}", result.segment_files[0]);
        let file = vfs.open_read(&seg0_path).unwrap();
        let file_size = file.size().unwrap();
        let col_types = vec![FlexTypeEnum::Integer, FlexTypeEnum::String, FlexTypeEnum::Integer];
        let mut reader = SegmentReader::open(Box::new(file), file_size, col_types).unwrap();
        let col0_vals = reader.read_column(0).unwrap();
        assert_eq!(col0_vals, vec![
            FlexType::Integer(30), FlexType::Integer(40), FlexType::Integer(50)
        ]);

        cache_fs.remove_dir(&base_path).ok();
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p sframe ec_sort::tests::test_scatter`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/sframe/src/ec_sort.rs
git commit -m "feat(ec_sort): implement scatter phase"
```

---

## Task 8: Permute phase implementation

**Files:**
- Modify: `crates/sframe/src/ec_sort.rs`

For each bucket, read the per-bucket forward_map and value columns from the
scatter output, permute in memory, and write to output segments.

**Step 1: Implement the permute function**

```rust
/// Permute phase: for each bucket, permute column values in memory
/// using the per-bucket forward_map, and write output segments.
///
/// Returns the output SFrame with permuted rows.
fn permute_buckets(
    vfs: Arc<dyn VirtualFileSystem>,
    scatter_base_path: &str,
    scatter_result: &ScatterResult,
    output_base_path: &str,
    output_data_prefix: &str,
    input_column_types: &[FlexTypeEnum],
    input_column_names: &[String],
    column_bytes: &[usize],
    num_rows: u64,
) -> Result<SFrame> {
    let num_buckets = scatter_result.segment_files.len();
    let num_input_cols = input_column_types.len();
    let budget_per_thread = sframe_config::global().sort_max_memory
        / rayon::current_num_threads().max(1);

    // Scatter output has num_input_cols + 1 columns (last is forward_map)
    let mut scatter_col_types: Vec<FlexTypeEnum> = input_column_types.to_vec();
    scatter_col_types.push(FlexTypeEnum::Integer);

    // Process buckets in parallel
    let results: Vec<Result<(String, Vec<u64>, u64)>> = (0..num_buckets)
        .into_par_iter()
        .map(|bucket_id| {
            permute_one_bucket(
                &*vfs,
                scatter_base_path,
                &scatter_result.segment_files[bucket_id],
                &scatter_col_types,
                output_base_path,
                output_data_prefix,
                bucket_id,
                input_column_types,
                column_bytes,
                budget_per_thread,
            )
        })
        .collect();

    // Assemble output SFrame from segments
    let mut segment_files = Vec::new();
    let mut all_segment_sizes = Vec::new();
    let mut total_rows = 0u64;

    for result in results {
        let (seg_file, sizes, rows) = result?;
        if rows > 0 {
            segment_files.push(seg_file);
            all_segment_sizes.push(sizes);
            total_rows += rows;
        }
    }

    if total_rows == 0 {
        // Return empty SFrame
        let dtypes = input_column_types.to_vec();
        let batch = sframe_query::batch::SFrameRows::empty(&dtypes);
        let plan = sframe_query::planner::PlannerNode::materialized(batch);
        let columns: Vec<SArray> = dtypes
            .iter()
            .enumerate()
            .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(0), i))
            .collect();
        return Ok(SFrame::new_with_columns(columns, input_column_names.to_vec()));
    }

    let col_name_refs: Vec<&str> = input_column_names.iter().map(|s| s.as_str()).collect();
    assemble_sframe_from_segments(
        &*vfs,
        output_base_path,
        &col_name_refs,
        input_column_types,
        &segment_files,
        &all_segment_sizes,
        total_rows,
        &HashMap::new(),
    )?;

    let cache_fs = global_cache_fs();
    let store: Arc<dyn Send + Sync> = Arc::new(AnonymousStore {
        path: output_base_path.to_string(),
        cache_fs: cache_fs.clone(),
    });
    let plan = sframe_query::planner::PlannerNode::sframe_source_cached(
        output_base_path,
        input_column_names.to_vec(),
        input_column_types.to_vec(),
        total_rows,
        store,
    );

    let columns: Vec<SArray> = input_column_types
        .iter()
        .enumerate()
        .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(total_rows), i))
        .collect();

    Ok(SFrame::new_with_columns(columns, input_column_names.to_vec()))
}

/// Permute a single bucket: read scatter segment, permute in memory, write output.
fn permute_one_bucket(
    vfs: &dyn VirtualFileSystem,
    scatter_base_path: &str,
    scatter_seg_file: &str,
    scatter_col_types: &[FlexTypeEnum],
    output_base_path: &str,
    output_data_prefix: &str,
    bucket_id: usize,
    input_col_types: &[FlexTypeEnum],
    column_bytes: &[usize],
    budget_per_thread: usize,
) -> Result<(String, Vec<u64>, u64)> {
    let num_input_cols = input_col_types.len();
    let fmap_col_idx = num_input_cols; // last column

    // Open the scatter segment
    let seg_path = format!("{scatter_base_path}/{scatter_seg_file}");
    let file = vfs.open_read(&seg_path)?;
    let file_size = file.size()?;
    let mut seg_reader = SegmentReader::open(
        Box::new(file), file_size, scatter_col_types.to_vec(),
    )?;

    let num_rows = seg_reader.column_len(fmap_col_idx);
    if num_rows == 0 {
        // Empty bucket — write empty segment
        let seg_file = segment_filename(output_data_prefix, bucket_id);
        let seg_path = format!("{output_base_path}/{seg_file}");
        let file = vfs.open_write(&seg_path)?;
        let seg_writer = BufferedSegmentWriter::new(file, input_col_types);
        let sizes = seg_writer.finish()?;
        return Ok((seg_file, sizes, 0));
    }

    // Read the per-bucket forward_map
    let fmap_values = seg_reader.read_column(fmap_col_idx)?;

    // Compute bucket_start: minimum forward_map value in this bucket
    let bucket_start = fmap_values.iter()
        .filter_map(|v| if let FlexType::Integer(i) = v { Some(*i as u64) } else { None })
        .min()
        .unwrap_or(0);

    // Build the permutation: for each input row r, output position is fmap[r] - bucket_start
    let permutation: Vec<usize> = fmap_values.iter()
        .map(|v| match v {
            FlexType::Integer(i) => (*i as u64 - bucket_start) as usize,
            _ => 0,
        })
        .collect();

    let num_rows_usize = num_rows as usize;

    // Create output segment
    let seg_file = segment_filename(output_data_prefix, bucket_id);
    let seg_path = format!("{output_base_path}/{seg_file}");
    let file = vfs.open_write(&seg_path)?;
    let mut out_writer = BufferedSegmentWriter::new(file, input_col_types);

    // Process columns in groups that fit in memory
    let mut col_start = 0;
    while col_start < num_input_cols {
        let mut col_end = col_start + 1;
        let mut mem_estimate = column_bytes[col_start] * num_rows_usize;

        while col_end < num_input_cols {
            let next_mem = column_bytes[col_end] * num_rows_usize;
            if mem_estimate + next_mem < budget_per_thread {
                mem_estimate += next_mem;
                col_end += 1;
            } else {
                break;
            }
        }

        // Read and permute this group of columns
        for col_idx in col_start..col_end {
            let values = seg_reader.read_column(col_idx)?;

            // Permute into output order
            let mut permuted = vec![FlexType::Undefined; num_rows_usize];
            for (input_row, value) in values.into_iter().enumerate() {
                let output_pos = permutation[input_row];
                permuted[output_pos] = value;
            }

            // Write in chunks to the output segment
            for chunk in permuted.chunks(CHUNK_SIZE) {
                out_writer.write_column_block(
                    col_idx, chunk, input_col_types[col_idx],
                )?;
            }
        }

        col_start = col_end;
    }

    let sizes = out_writer.finish()?;
    Ok((seg_file, sizes, num_rows))
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p sframe`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add crates/sframe/src/ec_sort.rs
git commit -m "feat(ec_sort): implement permute phase"
```

---

## Task 9: permute_sframe — connect scatter + permute

**Files:**
- Modify: `crates/sframe/src/ec_sort.rs`

**Step 1: Implement permute_sframe**

Replace the `todo!()` in `permute_sframe`:

```rust
pub(crate) fn permute_sframe(
    input: &SFrame,
    forward_map: &SArray,
) -> Result<SFrame> {
    let num_rows = input.num_rows()?;
    if num_rows == 0 {
        return input.head(0);
    }

    let column_types = input.column_types();
    let column_names = input.column_names().to_vec();
    let column_bytes = estimate_column_bytes(&column_types);

    let budget = sframe_config::global().sort_max_memory
        / rayon::current_num_threads().max(1);
    let num_buckets = compute_num_buckets(num_rows, &column_bytes, budget);
    let rows_per_bucket = (num_rows + num_buckets as u64 - 1) / num_buckets as u64;

    eprintln!(
        "[sframe] ec_sort permute: {num_rows} rows, {num_buckets} buckets, \
         {rows_per_bucket} rows/bucket"
    );

    // Allocate scratch directory for scatter output
    let cache_fs = global_cache_fs();
    let scatter_path = cache_fs.alloc_dir();
    let vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
    vfs.mkdir_p(&scatter_path)?;
    let scatter_prefix = format!("m_{}", generate_hash(&scatter_path));

    // Phase 1: Scatter
    eprintln!("[sframe] ec_sort: scatter phase...");
    let scatter_result = scatter_columns(
        &*vfs, &scatter_path, &scatter_prefix,
        input, forward_map,
        num_buckets, rows_per_bucket,
    )?;

    // Allocate output directory
    let output_path = cache_fs.alloc_dir();
    vfs.mkdir_p(&output_path)?;
    let output_prefix = format!("m_{}", generate_hash(&output_path));

    // Phase 2: Permute
    eprintln!("[sframe] ec_sort: permute phase...");
    let result = permute_buckets(
        vfs.clone(),
        &scatter_path,
        &scatter_result,
        &output_path,
        &output_prefix,
        &column_types,
        &column_names,
        &column_bytes,
        num_rows,
    )?;

    // Clean up scatter scratch
    cache_fs.remove_dir(&scatter_path).ok();

    Ok(result)
}
```

**Step 2: Write a test for permute_sframe**

```rust
#[cfg(test)]
mod tests {
    // ...

    #[test]
    fn test_permute_sframe_reverse() {
        // Input: 6 rows, 2 columns
        let col0 = SArray::from_vec(
            (0..6).map(|i| FlexType::Integer(i * 10)).collect(),
            FlexTypeEnum::Integer,
        ).unwrap();
        let col1 = SArray::from_vec(
            (0..6).map(|i| FlexType::String(format!("r{i}").into())).collect(),
            FlexTypeEnum::String,
        ).unwrap();
        let input = SFrame::new_with_columns(
            vec![col0, col1],
            vec!["a".to_string(), "b".to_string()],
        );

        // Forward map: reverse [5, 4, 3, 2, 1, 0]
        // Row 0 → output pos 5, row 1 → pos 4, ..., row 5 → pos 0
        let forward_map = SArray::from_vec(
            (0..6).rev().map(|i| FlexType::Integer(i)).collect(),
            FlexTypeEnum::Integer,
        ).unwrap();

        let result = permute_sframe(&input, &forward_map).unwrap();

        // Verify: output should be reversed
        let rows = result.iter_rows().unwrap();
        assert_eq!(rows.len(), 6);
        // Output pos 0 ← input row 5 (value 50, "r5")
        assert_eq!(rows[0][0], FlexType::Integer(50));
        assert_eq!(rows[0][1], FlexType::String("r5".into()));
        // Output pos 5 ← input row 0 (value 0, "r0")
        assert_eq!(rows[5][0], FlexType::Integer(0));
        assert_eq!(rows[5][1], FlexType::String("r0".into()));
    }

    #[test]
    fn test_permute_sframe_identity() {
        let col0 = SArray::from_vec(
            vec![FlexType::Integer(10), FlexType::Integer(20), FlexType::Integer(30)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let input = SFrame::new_with_columns(
            vec![col0],
            vec!["x".to_string()],
        );

        // Identity permutation
        let forward_map = SArray::from_vec(
            vec![FlexType::Integer(0), FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();

        let result = permute_sframe(&input, &forward_map).unwrap();
        let rows = result.iter_rows().unwrap();
        assert_eq!(rows[0][0], FlexType::Integer(10));
        assert_eq!(rows[1][0], FlexType::Integer(20));
        assert_eq!(rows[2][0], FlexType::Integer(30));
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p sframe ec_sort::tests::test_permute_sframe`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/sframe/src/ec_sort.rs
git commit -m "feat(ec_sort): implement permute_sframe connecting scatter and permute phases"
```

---

## Task 10: ec_sort — forward map generation and top-level function

**Files:**
- Modify: `crates/sframe/src/ec_sort.rs`

**Step 1: Implement forward map generation and ec_sort**

Replace the `todo!()` in `ec_sort`:

```rust
use sframe_query::planner::PlannerNode;

/// Generate the forward map by sorting key columns with row numbers.
///
/// 1. Create SFrame: [row_number(0..N), key_columns...]
/// 2. Sort by key columns → sorted result has row_number column = inverse_map
/// 3. Permute [0..N) by inverse_map → forward_map
fn generate_forward_map(
    input: &SFrame,
    key_column_indices: &[usize],
    sort_orders: &[bool],
) -> Result<(SFrame, SArray)> {
    let num_rows = input.num_rows()?;
    let column_names = input.column_names();

    // Build key column names and sort keys
    let mut sort_input_names = vec!["__row_number__".to_string()];
    let mut sort_input_cols: Vec<SArray> = Vec::new();

    // Add row number column (0..N)
    let row_number_plan = PlannerNode::range(0, 1, num_rows);
    let row_number = SArray::from_plan(row_number_plan, FlexTypeEnum::Integer, Some(num_rows), 0);
    sort_input_cols.push(row_number);

    // Add key columns
    for &key_idx in key_column_indices {
        sort_input_names.push(column_names[key_idx].clone());
        sort_input_cols.push(input.columns()[key_idx].clone());
    }

    let sort_input = SFrame::new_with_columns(sort_input_cols, sort_input_names);

    // Sort by key columns (indices 1..=num_keys in the sort_input SFrame)
    let sort_keys: Vec<(&str, SortOrder)> = key_column_indices
        .iter()
        .zip(sort_orders.iter())
        .map(|(&key_idx, &ascending)| {
            let name = column_names[key_idx].as_str();
            let order = if ascending { SortOrder::Ascending } else { SortOrder::Descending };
            (name, order)
        })
        .collect();

    eprintln!("[sframe] ec_sort: sorting key columns to generate inverse map...");
    let sorted = sort_input.sort(&sort_keys)?;

    // Extract inverse_map (row_number column from sorted result)
    let inverse_map = sorted.column("__row_number__")?.clone();

    // Extract sorted key columns
    let sorted_key_names: Vec<&str> = key_column_indices
        .iter()
        .map(|&i| column_names[i].as_str())
        .collect();
    let sorted_keys = sorted.select(&sorted_key_names)?;

    // Generate forward_map by permuting [0..N) with inverse_map
    // forward_map[inverse_map[i]] = i
    // Equivalently: permute_sframe([0..N), inverse_map)
    eprintln!("[sframe] ec_sort: generating forward map...");
    let range_plan = PlannerNode::range(0, 1, num_rows);
    let range_col = SArray::from_plan(range_plan, FlexTypeEnum::Integer, Some(num_rows), 0);
    let range_sf = SFrame::new_with_columns(
        vec![range_col],
        vec!["__idx__".to_string()],
    );

    let forward_map_sf = permute_sframe(&range_sf, &inverse_map)?;
    let forward_map = forward_map_sf.columns()[0].clone();

    Ok((sorted_keys, forward_map))
}

pub(crate) fn ec_sort(
    input: &SFrame,
    key_column_indices: &[usize],
    sort_orders: &[bool],
) -> Result<SFrame> {
    let num_rows = input.num_rows()?;
    if num_rows == 0 {
        return input.head(0);
    }

    let column_names = input.column_names().to_vec();
    let num_columns = column_names.len();

    // Step 1+2: Generate sorted key columns and forward_map
    let (sorted_keys, forward_map) = generate_forward_map(
        input, key_column_indices, sort_orders,
    )?;

    // Step 3: Identify value columns (non-key columns)
    let key_set: std::collections::HashSet<usize> =
        key_column_indices.iter().copied().collect();
    let value_column_indices: Vec<usize> = (0..num_columns)
        .filter(|i| !key_set.contains(i))
        .collect();

    if value_column_indices.is_empty() {
        // All columns are key columns — sorted_keys is the result
        // Reorder columns to match original order
        return reorder_columns(&sorted_keys, &column_names, key_column_indices, &[]);
    }

    // Select value columns from input
    let value_names: Vec<&str> = value_column_indices
        .iter()
        .map(|&i| column_names[i].as_str())
        .collect();
    let value_sf = input.select(&value_names)?;

    // Step 3: Permute value columns
    eprintln!("[sframe] ec_sort: permuting value columns...");
    let sorted_values = permute_sframe(&value_sf, &forward_map)?;

    // Step 4: Assemble final SFrame in original column order
    reorder_columns(&sorted_keys, &column_names, key_column_indices, &value_column_indices)
        .and_then(|_| {
            // Actually, build the final SFrame by combining key and value columns
            let mut final_columns: Vec<SArray> = vec![
                SArray::from_vec(vec![], FlexTypeEnum::Undefined).unwrap(); num_columns
            ];

            for (ki, &orig_idx) in key_column_indices.iter().enumerate() {
                final_columns[orig_idx] = sorted_keys.columns()[ki].clone();
            }
            for (vi, &orig_idx) in value_column_indices.iter().enumerate() {
                final_columns[orig_idx] = sorted_values.columns()[vi].clone();
            }

            Ok(SFrame::new_with_columns(final_columns, column_names.clone()))
        })
}

/// Reassemble columns in original order from sorted keys and sorted values.
fn reorder_columns(
    sorted_keys: &SFrame,
    column_names: &[String],
    key_column_indices: &[usize],
    value_column_indices: &[usize],
) -> Result<SFrame> {
    // This is a placeholder; the actual assembly happens in ec_sort
    // by placing columns at their original indices.
    Ok(sorted_keys.clone())
}
```

Note: The `reorder_columns` helper above is simplified. The actual assembly
in `ec_sort` places key and value columns at their original indices directly.
Clean this up during implementation — the `and_then` closure in `ec_sort`
contains the real logic.

**Step 2: Write tests for ec_sort**

```rust
#[cfg(test)]
mod tests {
    // ...

    #[test]
    fn test_ec_sort_basic() {
        // Create SFrame: key column "id" + value column "name"
        let ids = SArray::from_vec(
            vec![FlexType::Integer(3), FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let names = SArray::from_vec(
            vec![
                FlexType::String("three".into()),
                FlexType::String("one".into()),
                FlexType::String("two".into()),
            ],
            FlexTypeEnum::String,
        ).unwrap();
        let input = SFrame::new_with_columns(
            vec![ids, names],
            vec!["id".to_string(), "name".to_string()],
        );

        let result = ec_sort(&input, &[0], &[true]).unwrap();

        let rows = result.iter_rows().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][0], FlexType::Integer(1));
        assert_eq!(rows[0][1], FlexType::String("one".into()));
        assert_eq!(rows[1][0], FlexType::Integer(2));
        assert_eq!(rows[1][1], FlexType::String("two".into()));
        assert_eq!(rows[2][0], FlexType::Integer(3));
        assert_eq!(rows[2][1], FlexType::String("three".into()));
    }

    #[test]
    fn test_ec_sort_descending() {
        let ids = SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(3), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let vals = SArray::from_vec(
            vec![FlexType::Integer(10), FlexType::Integer(30), FlexType::Integer(20)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let input = SFrame::new_with_columns(
            vec![ids, vals],
            vec!["id".to_string(), "val".to_string()],
        );

        let result = ec_sort(&input, &[0], &[false]).unwrap();

        let rows = result.iter_rows().unwrap();
        assert_eq!(rows[0][0], FlexType::Integer(3));
        assert_eq!(rows[0][1], FlexType::Integer(30));
        assert_eq!(rows[2][0], FlexType::Integer(1));
        assert_eq!(rows[2][1], FlexType::Integer(10));
    }

    #[test]
    fn test_ec_sort_matches_standard_sort() {
        // Create a dataset with mixed types
        let n = 1000;
        let ids: Vec<FlexType> = (0..n).rev().map(|i| FlexType::Integer(i)).collect();
        let names: Vec<FlexType> = (0..n).rev()
            .map(|i| FlexType::String(format!("name_{i:04}").into()))
            .collect();
        let scores: Vec<FlexType> = (0..n).rev()
            .map(|i| FlexType::Float(i as f64 * 1.5))
            .collect();

        let input = SFrame::new_with_columns(
            vec![
                SArray::from_vec(ids, FlexTypeEnum::Integer).unwrap(),
                SArray::from_vec(names, FlexTypeEnum::String).unwrap(),
                SArray::from_vec(scores, FlexTypeEnum::Float).unwrap(),
            ],
            vec!["id".to_string(), "name".to_string(), "score".to_string()],
        );

        // Sort with ec_sort
        let ec_result = ec_sort(&input, &[0], &[true]).unwrap();

        // Sort with standard sort
        let std_result = input.sort(&[("id", SortOrder::Ascending)]).unwrap();

        // Compare
        let ec_rows = ec_result.iter_rows().unwrap();
        let std_rows = std_result.iter_rows().unwrap();
        assert_eq!(ec_rows.len(), std_rows.len());
        for i in 0..ec_rows.len() {
            assert_eq!(ec_rows[i], std_rows[i], "Mismatch at row {i}");
        }
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p sframe ec_sort::tests`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/sframe/src/ec_sort.rs
git commit -m "feat(ec_sort): implement ec_sort with forward map generation"
```

---

## Task 11: Expose ec_sort in SFrame API

**Files:**
- Modify: `crates/sframe/src/sframe.rs`

**Step 1: Add ec_sort method to SFrame**

Add a public method to SFrame (near the existing `sort` method):

```rust
/// Sort using the External Columnar Sort algorithm.
///
/// More efficient than `sort` for SFrames with many or large value columns.
/// Separates key sorting from value permutation, processing columns
/// independently to bound memory usage.
///
/// `keys` is a list of `(column_name, ascending)` pairs.
pub fn ec_sort(&self, keys: &[(&str, bool)]) -> Result<SFrame> {
    let key_indices: Vec<usize> = keys
        .iter()
        .map(|(name, _)| self.column_index(name))
        .collect::<Result<_>>()?;
    let sort_orders: Vec<bool> = keys.iter().map(|(_, asc)| *asc).collect();

    crate::ec_sort::ec_sort(self, &key_indices, &sort_orders)
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p sframe`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add crates/sframe/src/sframe.rs
git commit -m "feat: expose ec_sort as public SFrame method"
```

---

## Task 12: Integration and edge case tests

**Files:**
- Modify: `crates/sframe/src/ec_sort.rs` (add more tests)

**Step 1: Add integration tests**

```rust
#[cfg(test)]
mod tests {
    // ... (existing tests)

    #[test]
    fn test_ec_sort_single_row() {
        let input = SFrame::new_with_columns(
            vec![
                SArray::from_vec(vec![FlexType::Integer(42)], FlexTypeEnum::Integer).unwrap(),
                SArray::from_vec(vec![FlexType::String("hello".into())], FlexTypeEnum::String).unwrap(),
            ],
            vec!["a".to_string(), "b".to_string()],
        );

        let result = ec_sort(&input, &[0], &[true]).unwrap();
        let rows = result.iter_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], FlexType::Integer(42));
    }

    #[test]
    fn test_ec_sort_already_sorted() {
        let input = SFrame::new_with_columns(
            vec![
                SArray::from_vec(
                    (0..100).map(|i| FlexType::Integer(i)).collect(),
                    FlexTypeEnum::Integer,
                ).unwrap(),
                SArray::from_vec(
                    (0..100).map(|i| FlexType::String(format!("v{i}").into())).collect(),
                    FlexTypeEnum::String,
                ).unwrap(),
            ],
            vec!["id".to_string(), "val".to_string()],
        );

        let result = ec_sort(&input, &[0], &[true]).unwrap();
        let rows = result.iter_rows().unwrap();
        for i in 0..100 {
            assert_eq!(rows[i][0], FlexType::Integer(i as i64));
        }
    }

    #[test]
    fn test_ec_sort_all_identical_keys() {
        let input = SFrame::new_with_columns(
            vec![
                SArray::from_vec(
                    vec![FlexType::Integer(1); 10],
                    FlexTypeEnum::Integer,
                ).unwrap(),
                SArray::from_vec(
                    (0..10).map(|i| FlexType::Integer(i)).collect(),
                    FlexTypeEnum::Integer,
                ).unwrap(),
            ],
            vec!["key".to_string(), "val".to_string()],
        );

        let result = ec_sort(&input, &[0], &[true]).unwrap();
        assert_eq!(result.num_rows().unwrap(), 10);
        // All keys are 1 — values may be in any order (stable not required)
    }

    #[test]
    fn test_ec_sort_multi_key() {
        let col_a = SArray::from_vec(
            vec![FlexType::Integer(2), FlexType::Integer(1), FlexType::Integer(1), FlexType::Integer(2)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let col_b = SArray::from_vec(
            vec![FlexType::Integer(2), FlexType::Integer(2), FlexType::Integer(1), FlexType::Integer(1)],
            FlexTypeEnum::Integer,
        ).unwrap();
        let col_v = SArray::from_vec(
            vec![FlexType::String("d".into()), FlexType::String("b".into()),
                 FlexType::String("a".into()), FlexType::String("c".into())],
            FlexTypeEnum::String,
        ).unwrap();

        let input = SFrame::new_with_columns(
            vec![col_a, col_b, col_v],
            vec!["a".to_string(), "b".to_string(), "v".to_string()],
        );

        let result = ec_sort(&input, &[0, 1], &[true, true]).unwrap();
        let rows = result.iter_rows().unwrap();
        // Expected order: (1,1,"a"), (1,2,"b"), (2,1,"c"), (2,2,"d")
        assert_eq!(rows[0][2], FlexType::String("a".into()));
        assert_eq!(rows[1][2], FlexType::String("b".into()));
        assert_eq!(rows[2][2], FlexType::String("c".into()));
        assert_eq!(rows[3][2], FlexType::String("d".into()));
    }

    #[test]
    fn test_ec_sort_many_columns() {
        // 20 value columns to exercise column grouping in permute phase
        let n = 200;
        let mut cols = Vec::new();
        let mut names = Vec::new();

        // Key column
        let key: Vec<FlexType> = (0..n).rev().map(|i| FlexType::Integer(i)).collect();
        cols.push(SArray::from_vec(key, FlexTypeEnum::Integer).unwrap());
        names.push("key".to_string());

        // 20 value columns
        for c in 0..20 {
            let vals: Vec<FlexType> = (0..n)
                .map(|i| FlexType::String(format!("c{c}_r{i}").into()))
                .collect();
            cols.push(SArray::from_vec(vals, FlexTypeEnum::String).unwrap());
            names.push(format!("col{c}"));
        }

        let input = SFrame::new_with_columns(cols, names);
        let result = ec_sort(&input, &[0], &[true]).unwrap();

        let rows = result.iter_rows().unwrap();
        assert_eq!(rows.len(), n as usize);
        // Verify first and last row
        assert_eq!(rows[0][0], FlexType::Integer(0));
        assert_eq!(rows[0][1], FlexType::String("c0_r199".into()));
        assert_eq!(rows[199][0], FlexType::Integer(199));
        assert_eq!(rows[199][1], FlexType::String("c0_r0".into()));
    }
}
```

**Step 2: Run all tests**

Run: `cargo test -p sframe ec_sort`
Expected: All tests pass

**Step 3: Run full workspace tests to verify no regressions**

Run: `cargo test --workspace`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/sframe/src/ec_sort.rs
git commit -m "test(ec_sort): add integration and edge case tests"
```

---

## Implementation Notes

### Key patterns from existing code to follow:

1. **CacheFs allocation**: `global_cache_fs().alloc_dir()` for temp directories, wrap in `AnonymousStore` for RAII cleanup
2. **VFS pattern**: `Arc::new(ArcCacheFsVfs(cache_fs.clone()))` for CacheFs-backed VFS
3. **Segment assembly**: Use `assemble_sframe_from_segments()` to create metadata for pre-built segments
4. **PlannerNode construction**: Use `PlannerNode::sframe_source_cached()` to create keep-alive plan nodes
5. **SegmentReader from VFS**: `let file = vfs.open_read(path)?; SegmentReader::open(Box::new(file), file.size()?, types)`
6. **Streaming reads**: `SFrame::compile_stream()` → `BatchIterator` with `next_batch()` loop

### Potential issues to watch for:

1. **ReadableFile → ReadSeek**: `Box::new(file)` where `file: Box<dyn ReadableFile>` works because `ReadSeek` has a blanket impl for `T: Read + Seek + Send`
2. **Column ordering in SFrame::new_with_columns**: SArrays from different plans get combined via ColumnUnion; the execution engine handles lockstep consumption
3. **Empty buckets**: Some scatter segments may have 0 rows; skip them during assembly
4. **`segment_writer::estimate_bytes_per_value`** needs to be `pub(crate)` (Task 1)
5. **The `reorder_columns` helper** in Task 10 is a placeholder — the actual column reordering is done inline in `ec_sort`. Clean up during implementation.
