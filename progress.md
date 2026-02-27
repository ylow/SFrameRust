# Project Build - Activity Log

## Current Status
**Last Updated:** 2026-02-27
**Tasks Completed:** 9
**Current Task:** Phase 9 complete — next is Phase 10 (SArray operations)

---

## Session Log

### 2026-02-27 — Phase 9: Missing Aggregators

**Commit:** `8369ade` — `feat(query): Phase 9 — missing aggregators and AggSpec constructors`

**What was done:**
Phase 9.1: Wired existing aggregators (variance, stddev, concat) to AggSpec convenience constructors. Phase 9.2: Added 4 new aggregators — CountDistinct (HashSet-based), NonNullCount, SelectOne (first non-Undefined), Quantile (collect+sort+pick percentile). Added AggSpec constructors: count_distinct(), quantile(), median(), select_one(). 6 new tests.

**Tests:** All 216 tests pass.

### 2026-02-27 — Phase 8.3: SArray Element-wise Binary Operations

**Commit:** `20cdb06` — `feat(sarray): Phase 8.3 — element-wise binary operations`

**What was done:**
Added element-wise operations to SArray: arithmetic (add, sub, mul, div, rem — array-array and array-scalar), comparison (eq, ne, lt, le, gt, ge → Integer 0/1 arrays), logical (and, or). Array-array ops materialize the RHS and apply element-wise. 9 new tests.

**Tests:** All 210 tests pass.

### 2026-02-27 — Phase 8.1-8.2: FlexType Arithmetic Operators and Ordering

**Commit:** `727deab` — `feat(types): Phase 8.1-8.2 — FlexType arithmetic operators and ordering`

**What was done:**
Implemented std::ops traits on FlexType: Add, Sub, Mul, Div, Rem, Neg matching C++ flexible_type semantics (int+int→int, float+float→float, int+float→float, string+string→concat, vector+vector→element-wise, vector*scalar→broadcast, div always returns Float). Added PartialOrd with cross-type numeric comparison and Undefined sorting last. 20 new tests.

**Tests:** All 201 tests pass.

### 2026-02-27 — Phase 7.2-7.3: CSV Tokenizer + Structured Type Inference

**Commit:** `d39ed9b` — `feat(csv): Phase 7.2-7.3 — custom CSV tokenizer + structured type inference`

**What was done:**
Built bracket-aware CSV tokenizer replacing the csv crate. Handles nested brackets/braces in unquoted fields, comment lines, C-style escape sequences, double-quote escaping, configurable delimiters, NA values, skip_rows, row_limit. Extended type inference to detect Vector/List/Dict using parse_flextype. Removed csv crate dependency from sframe-query.

**Tests:** All 181 tests pass.

### 2026-02-27 — Phase 7.1: Flexible Type Parser

**Commit:** `74b57d4` — `feat(types): Phase 7.1 — flexible type parser (recursive descent)`

**What was done:**
Created parse_flextype(s) recursive descent parser that parses strings into best-matching FlexType. Priority: float (with dot) > integer > vector > list > dict > string fallback. Supports flexible vector separators (comma/space/semicolon), nested structures, quoted string elements with C-style escapes, Unicode surrogate pairs, graceful fallback on mismatched brackets. 26 new tests.

**Tests:** All 159 tests pass.

### 2026-02-27 — Phase 6.4 + 6.7: EC-Sort Pattern and Memory Budget Config

**Commit:** `c5a9980` — `feat(query): Phase 6.4 + 6.7 — EC-Sort pattern and memory budget config`

**What was done:**
Phase 6.4: Refactored sort to use EC-Sort pattern — index-based permutation where only key columns are accessed during comparisons. Added estimate_batch_size() and public compare_flex_type(). Phase 6.7: Created centralized SFrameConfig with memory budget constants. 2 new tests.

**Tests:** All 133 tests pass.

### 2026-02-27 — Phase 6.3: Multi-Segment Read/Write

**Commit:** `d768c50` — `feat(streaming): Phase 6.3 — multi-segment read/write`

**What was done:**
Added multi-segment support for both writing and reading (+315/-67 lines across 4 files):

1. **`sframe_writer.rs`** — Refactored `SFrameWriter` with segment auto-splitting. New fields: `rows_per_segment`, `current_segment_idx`, `rows_in_current_segment`, `segment_files`, `all_segment_sizes`. When current segment fills, it finishes and creates a new one. Added `with_segment_size()` constructor. Updated `write_sidx` and `write_frame_idx` for multi-segment metadata.

2. **`sframe_reader.rs`** — Added `num_segments()` and `read_segment_columns(segment_idx)` for segment-at-a-time reading.

3. **`execute.rs`** — Rewrote `compile_sframe_source` to read one segment at a time. Only one segment's data is in memory at a time, reducing peak memory by a factor of num_segments.

4. **`roundtrip_write_read.rs`** — 2 new tests: multi-segment roundtrip (2500 rows at 1000/segment = 3 segments), exact boundary test (2000 rows at 1000/segment = 2 segments).

**Tests:** All 132 tests pass (2 new tests).

### 2026-02-27 — Phase 6.2: Streaming Writer with Cross-Batch Buffering

**Commit:** `1311e2d` — `feat(streaming): Phase 6.2 — streaming writer with cross-batch buffering`

**What was done:**
Enhanced SFrameWriter with internal column buffering and added SFrameStreamWriter wrapper (+275/-15 lines across 4 files):

1. **`sframe_writer.rs`** — Added `column_buffers` and `buffered_rows` fields to `SFrameWriter`. `write_columns()` now appends to internal buffers and flushes complete blocks via `flush_full_blocks()`. `finish()` calls `flush_remaining()` before finalizing. Small batches are coalesced into proper-sized blocks; large batches are split.

2. **`sframe.rs`** — Added `SFrameStreamWriter` wrapper that bridges `SFrameRows` (from sframe-query) to `SFrameWriter` (in sframe-storage). Provides `write_batch(&SFrameRows)` API. Updated `SFrame::save()` to use it.

3. **`lib.rs`** — Exported `SFrameStreamWriter` from the sframe crate.

4. **`roundtrip_write_read.rs`** — 4 new tests: small-batch coalescing (100×10 rows), large single batch (50K rows), empty writer, variable-size batches. Plus 1 new `SFrameStreamWriter` test in sframe crate.

**Tests:** All 130 tests pass (5 new tests).

### 2026-02-27 — Phase 6.1: Streaming Execution Engine

**Commit:** `9d3cfe4` — `feat(streaming): Phase 6.1 — streaming execution engine`

**What was done:**
Replaced the "compile → materialize everything" model with true streaming batch execution across 4 files (+377/-34 lines):

1. **`execute.rs`** — Added `materialize_head(stream, limit)` that pulls only enough batches to fill the requested row count, and `for_each_batch_sync(stream, callback)` that consumes a stream batch-by-batch without collecting into memory.

2. **`sframe_writer.rs`** — Added `SFrameWriter` struct for incremental batch-by-batch writing. Accepts column data via `write_columns()`, writes blocks to segment, then finalizes metadata with `finish()`.

3. **`sframe.rs`** — Made `filter()` lazy (builds Filter plan node instead of materializing), `head(n)` streaming (pulls only n rows), `save()` streaming (writes batch-by-batch via SFrameWriter), `append()` lazy (builds Append plan node). Also eliminated double materialize→re-wrap→re-compile cycle in `sort/join/groupby`. Added `shared_plan()` and `compile_stream()` helpers.

4. **`sarray.rs`** — Made `head(n)` use `materialize_head_sync` to pull only n rows.

**Tests:** All 125 tests pass (including 4 new tests for materialize_head and for_each_batch_sync).
