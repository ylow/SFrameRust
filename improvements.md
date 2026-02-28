# SFrameRust: Out-of-Core and Bounded-Memory Improvements Plan

## Problem Statement

SFrameRust currently materializes entire datasets into memory for most
operations. While the batch-streaming pipeline (4096-row batches) and the
groupby spill mechanism provide partial bounded-memory behavior, the three
most memory-intensive operations — **sort**, **join**, and **source reading**
— still require O(N) memory where N is the full dataset size. This makes
SFrameRust unsuitable for datasets that exceed available RAM.

The C++ SFrame solves this with external-memory algorithms (EC-Sort, GRACE
hash join, spillable groupby with k-way merge) and truly lazy segment-level
I/O. This plan describes the concrete changes needed to close that gap.

---

## Current State Assessment

### What already works (bounded memory)
- **Groupby**: Hash-partitioned with disk spill via `cache://` and k-way merge
  (`groupby.rs`). This is the most complete out-of-core algorithm.
- **Streaming pipeline**: Filter, project, transform operate batch-by-batch
  without materializing the full dataset.
- **Writer**: `SFrameWriter` buffers per-column and flushes blocks
  incrementally with auto segment splitting.
- **CacheFs**: 2 GiB in-memory cache with disk spill for intermediates.
- **`materialize_head()`**: Short-circuits reading after enough rows.

### What requires O(N) memory (the gaps)
1. **Source reading** (`compile_sframe_source`): Reads ALL segments into
   `Vec<Vec<FlexType>>` before re-batching. Even a 10 GB SFrame is fully
   materialized before the first batch is emitted.
2. **Sort** (`sort.rs`): Materializes the entire input stream, then sorts
   in-memory. No external sort path.
3. **Join** (`join.rs`): Materializes both left and right sides entirely,
   then builds an in-memory hash table. No partitioning.
4. **`materialize()` helper**: Several code paths (SArray `.to_vec()`,
   `.unique()`, `.sort()`) funnel through full materialization.

---

## Improvement 1: Lazy Segment-Level Source Reading ✅ DONE

**Problem**: `compile_sframe_source` in `execute.rs:174-238` reads every
segment file eagerly into memory (parallel `read_segment_independently`),
collects all results, then slices into batches. A 100-segment SFrame loads
all 100 segments before emitting the first batch.

**Solution**: Read segments lazily — open and decode one segment at a time,
emit its batches, then drop it before reading the next segment.

### Design

```
compile_sframe_source(path, types, num_rows)
  → Stream that:
      for each segment_path in segment_files:
          open segment_path via VFS
          for each block in segment:
              decode block → Vec<FlexType>
              yield SFrameRows batch
          drop segment file handle
```

### Changes

- **`execute.rs`**: Replace the `par_iter().map(read_segment_independently)`
  + `stream::iter(batches)` pattern with an `async_stream` that opens one
  segment at a time, reads blocks incrementally, and yields batches as they
  are decoded.

- **`segment_reader.rs`**: Add a `read_column_blocks()` iterator or
  `read_column_block(col, block_idx)` method that decodes a single block
  instead of the full column. The block metadata (offsets, sizes) is already
  in the segment footer — expose it.

- **Parallel variant**: For multi-segment sources, allow a configurable
  concurrency level (e.g. 2-4 segments in flight). Use a bounded channel
  or `futures::stream::buffered(N)` to limit memory to N segments instead
  of all segments.

### Memory bound
- O(batch_size × num_columns) per in-flight segment instead of
  O(total_rows × num_columns).

### Compatibility note
- The current parallel `rayon::par_iter` path can be retained as a fast
  path when the dataset is known to fit in memory (heuristic: estimated
  size < configurable threshold). Otherwise, fall back to the lazy path.

---

## Improvement 2: External Sort (EC-Sort) ✅ DONE

**Problem**: `sort.rs:57-98` materializes the entire input stream into a
single `SFrameRows` and sorts in-memory. Datasets larger than RAM cannot be
sorted.

**Solution**: Implement a two-phase external sort modeled after the C++
EC-Sort algorithm, which is particularly efficient for wide tables because
it avoids shuffling non-key columns during partitioning.

### Algorithm

**Phase 1 — Partition by quantile ranges**:
1. Streaming quantile sketch: Sample a fraction of key-column values to
   estimate quantile cut points. This requires only O(1/epsilon) memory.
   (C++ uses `sframe_query_engine/algorithm/sort.cpp:40-86` with 0.5%
   error tolerance.)
2. Determine P partitions such that each partition fits in the sort memory
   budget (`SFRAME_SORT_BUFFER_SIZE`).
3. Stream the input again. For each batch, scatter rows into partition
   buckets based on which quantile range their key falls into. Write each
   partition to a temporary SFrame via `cache://`.

**Phase 2 — Sort each partition**:
4. For each partition (which fits in memory by construction), read it,
   sort in-memory (using the existing `build_sort_indices`), and write
   the sorted result to the output SFrame.
5. Concatenate the P sorted partitions in order (they are globally ordered
   by the quantile cut points).

### Fast path
- If estimated data size ≤ sort buffer budget, skip external sort and use
  the current in-memory path directly.

### Changes

- **`sort.rs`**: Add `external_sort()` that implements the two-phase
  algorithm. The public `sort()` function checks estimated size and
  dispatches to in-memory or external sort.

- **`sframe-config`**: Add `SFRAME_SORT_BUFFER_SIZE` config
  (default: 2 GB or 50% of available memory). Add
  `SFRAME_SORT_PIVOT_ESTIMATION_SAMPLE_SIZE` (default: 1M).

- **New module `quantile_sketch.rs`** (in `sframe-query`): Streaming
  approximate quantile data structure. The Greenwald-Khanna or t-digest
  algorithm would work. Needs to support `merge()` for parallel sketches.

- **Temporary storage**: Use the existing `CacheFs` + `SFrameWriter` to
  write partition files. Each partition is a temporary SFrame that is
  deleted after being read.

### Memory bound
- O(sort_buffer_size) — configurable, defaults to fit in available RAM.
- Quantile sketch: O(1/epsilon × log(N)) ≈ O(few KB).

---

## Improvement 3: GRACE Hash Join

**Problem**: `join.rs:56-64` materializes both sides of the join into
memory and builds a single hash table on the right side. If either input
exceeds RAM, the join fails or swaps.

**Solution**: Implement the GRACE (General Recursive Algorithm for
Scalable Hashing) hash join, which partitions both inputs by hash(key)
and joins each partition independently.

### Algorithm

**Phase 1 — Partition both sides**:
1. Choose P partitions such that the smaller side of each partition fits
   in the hash join memory budget (`SFRAME_JOIN_BUFFER_NUM_CELLS`).
   P = ceil(estimated_smaller_side / budget).
2. Stream left input: hash each row's join key, write to partition
   `hash(key) % P`. Each partition is a temporary SFrame via `cache://`.
3. Stream right input: same partitioning.

**Phase 2 — Per-partition hash join**:
4. For each partition p = 0..P:
   - Load the smaller side's partition into an in-memory hash table.
   - Stream the larger side's partition and probe the hash table.
   - Emit matched rows (and unmatched for LEFT/RIGHT/FULL joins).
5. Concatenate partition outputs.

### Changes

- **`join.rs`**: Add `grace_hash_join()`. The public `join()` checks
  estimated sizes and dispatches to in-memory or GRACE.

- **`sframe-config`**: Add `SFRAME_JOIN_BUFFER_NUM_CELLS` config
  (default: ~100M cells, where a cell ≈ one FlexType value).

- **Size estimation**: Use `estimate_batch_size()` (already in `sort.rs`)
  applied to the first few batches to estimate per-row cost, then
  extrapolate using `num_rows` metadata from the source.

### Memory bound
- O(join_buffer_size) — only one partition's hash table is in memory at
  a time.

### Compatibility
- The current in-memory hash join is the fast path for small inputs.
  GRACE partitioning is only triggered when estimated sizes exceed the
  budget.

---

## Improvement 4: Streaming Unique / Dedup

**Problem**: `SArray::unique()` likely materializes the entire array to
compute unique values. For sorted input, this could be streaming.

**Solution**: Two strategies:
1. **Hash-partitioned unique** (analogous to groupby spill): Partition by
   hash, deduplicate per partition, merge.
2. **Sort-based unique**: Sort (using the external sort above), then do
   a single streaming pass removing consecutive duplicates.

### Changes
- Implement as a thin wrapper around the external sort + streaming
  dedup pass, or as a partitioned hash set with the same spill mechanism
  as groupby.

---

## Improvement 5: Configuration System for Memory Budgets ✅ DONE

**Problem**: `sframe-config` only has cache capacity settings. The C++
SFrame has a rich set of tunable constants that control memory budgets
for every algorithm.

**Solution**: Extend the configuration system.

### New config keys

| Key | Default | Description |
|-----|---------|-------------|
| `SFRAME_SORT_BUFFER_SIZE` | 2G | Max memory for sort partitions |
| `SFRAME_JOIN_BUFFER_NUM_CELLS` | 100000000 | Max cells for join hash table |
| `SFRAME_GROUPBY_BUFFER_NUM_ROWS` | 1048576 | Already exists as constant; promote to config |
| `SFRAME_SOURCE_PREFETCH_SEGMENTS` | 2 | Segments to prefetch in lazy source |
| `SFRAME_MAX_BLOCKS_IN_CACHE` | 1024 | Decoded block cache size |
| `SFRAME_WRITER_BUFFER_SOFT_LIMIT` | 64M | Writer buffer soft flush threshold |
| `SFRAME_WRITER_BUFFER_HARD_LIMIT` | 128M | Writer buffer hard flush threshold |

### Changes
- **`sframe-config/src/lib.rs`**: Add atomic globals for each new key, with
  environment variable overrides following the existing pattern.
- **Algorithm files**: Replace hardcoded constants with config lookups.

---

## Improvement 6: Block-Level Random Access in Segment Reader ✅ DONE

**Problem**: `SegmentReader::read_column()` decodes ALL blocks for a column.
There is no way to read a single block or a range of blocks.

**Solution**: Expose block-level access using the metadata already present
in the segment footer.

### Changes

- **`segment_reader.rs`**: Add methods:
  ```rust
  fn num_blocks(&self, column: usize) -> usize;
  fn read_block(&mut self, column: usize, block_idx: usize) -> Result<Vec<FlexType>>;
  fn block_iter(&mut self, column: usize) -> impl Iterator<Item = Result<Vec<FlexType>>>;
  ```

- This enables the lazy source reader (Improvement 1) to decode block by
  block instead of column-at-a-time. It also enables future predicate
  pushdown (skip blocks whose min/max statistics don't match the filter).

---

## Improvement 7: Buffer Pool / Arena for Batch Allocation

**Problem**: Every batch allocates fresh `Vec<FlexType>` and `ColumnData`.
For high-throughput pipelines, this causes frequent allocation pressure.

**Solution**: Reusable buffer pool for `ColumnData` vectors.

### Design
- A thread-local free-list of pre-allocated `ColumnData` vectors.
- When a batch is consumed, its column buffers are returned to the pool.
- The pool has a configurable maximum size to cap memory.

This is a lower-priority optimization — the algorithmic changes above
have much larger impact. But it reduces allocation overhead for the
streaming pipeline.

---

## Priority and Ordering

| Priority | Improvement | Impact | Effort | Status |
|----------|-------------|--------|--------|--------|
| **P0** | 1. Lazy source reading | Eliminates the #1 source of unnecessary memory use | Medium | **DONE** |
| **P0** | 2. External sort | Unblocks sorting of large datasets | High | **DONE** |
| **P1** | 3. GRACE hash join | Unblocks joining of large datasets | High | TODO |
| **P1** | 5. Config system | Prerequisite infrastructure for memory budgets | Low | **DONE** |
| **P2** | 6. Block-level random access | Enables lazy source + future predicate pushdown | Medium | **DONE** |
| **P2** | 4. Streaming unique | Dedup of large arrays | Low | TODO |
| **P3** | 7. Buffer pool | Throughput optimization | Medium | TODO |

### Suggested implementation order
1. **Config system** (P1 but low effort, prerequisite for others) — **DONE**
2. **Block-level random access** (P2 but enables #3) — **DONE**
3. **Lazy source reading** (P0, builds on block-level access) — **DONE**
4. **External sort** (P0, needs quantile sketch + temp SFrame writing) — **DONE**
5. **GRACE hash join** (P1, same partitioning infrastructure as sort)
6. **Streaming unique** (P2, reuses sort or groupby spill)
7. **Buffer pool** (P3, performance polish)

### Implementation notes (completed items)

**Config system** (`sframe-query/src/config.rs`): `SFrameConfig` now uses
`LazyLock` to read env vars on first access: `SFRAME_SOURCE_BATCH_SIZE`,
`SFRAME_SORT_BUFFER_SIZE`, `SFRAME_GROUPBY_BUFFER_NUM_ROWS`,
`SFRAME_JOIN_BUFFER_NUM_CELLS`, `SFRAME_SOURCE_PREFETCH_SEGMENTS`.
`parse_byte_size` made public in `sframe-config`.

**Block-level random access** (`sframe-storage/src/segment_reader.rs`):
Added `num_blocks()`, `block_num_elem()`, `read_block()`. `read_column()`
refactored to use `read_block()` internally.

**Lazy source reading** (`sframe-query/src/execute.rs`): Replaced eager
`rayon::par_iter` materialization with `futures::stream::unfold`-based lazy
stream that reads one segment at a time. Memory is O(one segment) instead
of O(all segments).

**External sort** (`sframe/src/external_sort.rs`): 5-phase algorithm using
Greenwald-Khanna quantile sketch (`sframe-query/src/algorithms/quantile_sketch.rs`)
for partition boundaries. `SFrame::sort()` dispatches to `sort_in_memory()`
or `external_sort()` based on `estimate_size()` vs `sort_memory_budget`.
Partitions on primary key; full multi-key sort within each partition.

---

## What This Plan Does NOT Cover

- **Predicate pushdown with block statistics**: Requires storing min/max
  per block in segment metadata. Valuable but orthogonal to bounded-memory.
- **Parallel external sort**: The C++ version parallelizes the partition
  phase across threads. The initial Rust implementation can be single-
  threaded for correctness, then parallelized.
- **S3/HDFS backends**: Stubs exist; full implementation is orthogonal.
- **Memory-mapped I/O**: An alternative to explicit buffering. Could be
  explored but changes the programming model significantly.
