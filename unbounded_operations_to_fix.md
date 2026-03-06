# Unbounded Memory Operations To Fix

Target memory model: `SFRAME_CACHE_CAPACITY + k * nthreads` where k ~ 50-100MB.

## 1. Join: Full Materialization of Both Sides

**Location**: `crates/sframe-query/src/algorithms/join.rs:68-69`

Both left and right streams are fully materialized into `SFrameRows` before
any size check occurs. Even the GRACE hash join path materializes both sides
first, then partitions. The `join_buffer_num_cells` config only controls
whether to use in-memory vs GRACE join after everything is already in memory.

**Peak memory**: O(left_rows * left_cols + right_rows * right_cols) —
unbounded.

**Additionally**: GRACE `partition_rows` (line 263-268) creates a full copy
of each side via `take()` before dropping the originals, so peak is 2x each
side during partitioning.

**Fix direction**: Stream the probe side (left) instead of materializing it.
Only materialize the build side (right), and only if it fits in budget.
For large builds, use a disk-backed partitioned approach where partitions
are spilled to CacheFs before the probe pass.

## 2. ~~Sort: Full Materialization With No External Sort Fallback~~ ✅ FIXED

**Location**: `crates/sframe-query/src/algorithms/sort.rs`

**Fixed in**: Sort now uses streaming run-generation + k-way merge. Input
batches are accumulated up to `SFRAME_SORT_BUFFER_SIZE` (default 256 MiB),
then sorted and spilled to CacheFs as segment files. After all input is
consumed, sorted runs are merged via a BinaryHeap-based min-heap, emitting
sorted batches through a BatchIterator. Data that fits in budget uses a fast
in-memory path with chunked emission (no full sorted copy). Peak memory is
O(sort_buffer_size + merge_batch_size * num_runs * num_cols).

## 3. ~~Parallel Execution: All Workers Materialize Simultaneously~~ ✅ FIXED

**Location**: `crates/sframe-query/src/execute/parallel.rs:107-176`

**Fixed in**: Each rayon worker now streams its output via `consume_to_segment`
directly to a segment file on CacheFS. Workers never hold full `SFrameRows`
in memory — they process batch-by-batch. After all workers finish, segments
are assembled into an SFrame via `assemble_sframe_from_segments`.

## 4. ~~CSV Read: Entire File as In-Memory String~~ ✅ FIXED

**Location**: `crates/sframe-query/src/algorithms/csv_parser.rs:440`

**Fixed in**: commit 5569939 — `read_csv` now delegates to `CsvStreamingParse`
which reads the file in 50MB chunks. Only one chunk + its parsed output exist
at a time, eliminating the 3x memory amplification.

## 5. ~~Groupby: Unbounded Output Collection~~ ✅ FIXED

**Location**: `crates/sframe-query/src/algorithms/groupby.rs`

**Fixed in**: Groupby now accepts a `PlannerNode` and writes output directly
to CacheFS via `SegmentWriter`, returning a `cache://` path. Output rows are
buffered in 64K-row batches before writing, so peak output memory is
O(batch_size * num_columns). Input is also read in parallel via rayon workers
when the plan is parallel-sliceable.

## 6. ~~Parallel Reduce: Borderline Per-Worker Column Reads~~ ✅ FIXED

**Location**: `crates/sframe-query/src/execute/reduce.rs`

**Fixed in**: Each rayon worker now processes its row range in 256K-row
sub-chunks. After feeding each chunk to the aggregator, the column data is
dropped before the next chunk is read. Per-worker peak memory is
O(256K * projected_cols) regardless of total data size.
