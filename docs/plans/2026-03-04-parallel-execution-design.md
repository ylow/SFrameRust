# Parallel Execution Redesign

Date: 2026-03-04

## Problem

`compile_parallel()` in `execute/parallel.rs:128-134` has each rayon worker
fully materialize its row-range slice into `SFrameRows`. All workers run
concurrently via `par_iter`, so peak memory is the entire dataset held in
memory at once (`nthreads * slice_size = total_data`). The results are
collected into a `Vec<Result<SFrameRows>>` before being re-wrapped as a
stream, defeating the streaming model entirely.

## Design Principle

A single execution plan is always executed sequentially. Data parallelism is
an orchestration concern above the plan level. Consumers are pluggable
functions that take a sequential stream and do something with it.

## Architecture

```
SFrame/SArray API              Decides when to parallelize.
  |                            Calls orchestrator or compile().
  v
Parallel Orchestrator          Splits plan into N sub-plans.
(execute/parallel.rs)          Runs each with segment consumer.
  |                            Assembles SFrame on CacheFs.
  v
Sequential Execution           compile() -> BatchStream (always single-threaded).
(execute/mod.rs)               Segment consumer drives stream -> segment file.
(execute/consumer.rs)
```

Callers (save, materialize, head, etc.) decide independently whether to use
parallel execution or go sequential. The execution layer provides building
blocks; it does not make the parallelism decision.

## Components

### 1. `ColumnData::to_flex_vec()` (batch.rs)

New method on `ColumnData` that converts typed column storage to
`Vec<FlexType>` for the segment writer interface.

- Typed variants (Integer, Float, etc.): maps `Option<T>` to `FlexType`
  (None becomes Undefined).
- `Flexible` variant: clones the inner `Vec<FlexType>` directly.

### 2. `consume_to_segment()` (execute/consumer.rs)

Drives a sequential `BatchStream` and writes output into a single segment
file via `SegmentWriter`.

```
fn consume_to_segment(
    stream: BatchStream,
    seg_writer: SegmentWriter<Box<dyn WritableFile>>,
    dtypes: &[FlexTypeEnum],
) -> Result<(Vec<u64>, u64)>   // (segment_sizes, total_rows)
```

Flow:
1. Create a tokio runtime to drive the async stream.
2. For each batch, for each column: convert `ColumnData` to `Vec<FlexType>`
   via `to_flex_vec()`, write via `seg_writer.write_column_block()`.
3. Uses adaptive block sizing (type-based estimate refined by cumulative
   average bytes-per-value, targeting ~64KB blocks).
4. Call `seg_writer.finish()`.
5. Return segment sizes and total row count.

Blocks from different columns are interleaved in the segment file (batch 0
col 0, batch 0 col 1, batch 1 col 0, ...). This is correct because
`SegmentReader` uses the block index footer to seek by offset. Each column's
blocks are in row order.

### 3. `assemble_sframe_from_segments()` (sframe_writer.rs)

Public function that writes SFrame metadata given pre-built segment files.

```
pub fn assemble_sframe_from_segments(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    column_names: &[&str],
    column_types: &[FlexTypeEnum],
    segment_files: &[String],
    all_segment_sizes: &[Vec<u64>],
    total_rows: u64,
) -> Result<()>
```

Writes: dir_archive.ini, {prefix}.sidx, {prefix}.frame_idx, objects.bin.
Reuses existing `build_sidx_content()`, `build_frame_idx_content()`, and
`build_dir_archive_ini_content()` helpers in `sframe_writer.rs`.

### 4. `execute_parallel()` (execute/parallel.rs)

Splits a plan, runs N workers with segment consumers, assembles the result
into an SFrame on CacheFs.

```
pub fn execute_parallel(
    plan: &Arc<PlannerNode>,
    total_rows: u64,
    column_names: &[String],
    dtypes: &[FlexTypeEnum],
) -> Result<String>   // CacheFs SFrame path
```

Flow:
1. Determine N workers from rayon thread pool.
2. Split plan into N sub-plans via `clone_plan_with_row_range()`.
3. Allocate a CacheFs directory via `cache_fs.alloc_dir()`.
4. `par_iter` over workers. Each worker:
   a. Opens a segment file in the CacheFs directory.
   b. Creates a `SegmentWriter` on that file.
   c. Calls `compile_single_threaded(&sub_plan)` to get a `BatchStream`.
   d. Calls `consume_to_segment()` to drive the stream into the segment.
   e. Returns (segment_file_name, segment_sizes, row_count).
5. Calls `assemble_sframe_from_segments()` with collected metadata.
6. Returns the CacheFs SFrame path.

Lifecycle: The caller holds a `CacheGuard` on the CacheFs directory. When
the guard drops, CacheFs cleans up the intermediate files.

Unchanged: `parallel_slice_row_count()` stays as-is (pure check on plan
structure). `clone_plan_with_row_range()` stays as-is.

### 5. Simplify `compile()`

Remove the `compile_parallel` path. `compile()` becomes:

```
pub fn compile(node: &Arc<PlannerNode>) -> Result<BatchStream> {
    let node = optimizer::optimize(node);
    compile_single_threaded(&node)
}
```

### 6. Caller Decides

The SFrame/SArray API decides independently per operation:

- **`save(path)`**: May call `execute_parallel()` to produce a CacheFs
  SFrame, then copy/stream to the output path. Or may have workers write
  segments directly to the output path (skipping CacheFs) and call
  `assemble_sframe_from_segments()` there.
- **`materialize()`**: May call `execute_parallel()` then read back from the
  CacheFs SFrame.
- **`head(n)`**: Calls sequential `compile()` + `materialize_head_sync()`.
  No parallel path. Already tries `try_slice()` first for O(n) reads.
- **`to_csv()`**, **`for_each_batch()`**, etc.: Caller's choice.

## Memory Model

Peak memory for parallel execution:

- Each worker holds at most one batch (~4096 rows * num_cols) in memory at a
  time, plus the `SegmentWriter` internal buffer.
- CacheFs manages intermediate segment storage with bounded in-memory
  capacity (default 2GB) and automatic disk spill.
- Total: O(batch_size * num_cols * nthreads) + CacheFs budget.

This replaces the current O(total_rows * num_cols) unbounded peak.

## Files Changed

| File | Change |
|------|--------|
| `sframe-query/src/batch.rs` | Add `ColumnData::to_flex_vec()` |
| `sframe-query/src/execute/consumer.rs` | New: `consume_to_segment()` |
| `sframe-query/src/execute/parallel.rs` | Rewrite: `execute_parallel()` returning CacheFs path |
| `sframe-query/src/execute/mod.rs` | Remove `compile_parallel` path from `compile()` |
| `sframe-storage/src/sframe_writer.rs` | Add `assemble_sframe_from_segments()` |
| `sframe/src/sframe.rs` | Callers decide parallel strategy |
| `sframe/src/sarray.rs` | Callers decide parallel strategy |
