# SFrameRust vs C++ SFrame: Gap Analysis and Roadmap

## What's Already Done (Rust ~ C++ Feature Parity)

| Area | Status |
|---|---|
| **Core data type** (`FlexType`) | All 8 variants (Int, Float, String, Vector, List, Dict, DateTime, Undefined). Arithmetic, comparison, hashing, ordering. |
| **V2 storage format** (read + write) | Block encoding/decoding, segment reader/writer, index files, dir_archive. Full round-trip verified. |
| **Integer codec** | FoR, FoR-delta, FoR-delta-negative with 1/2/4/8/16/32/64-bit packing. |
| **Lazy query engine** | Logical plan DAG, optimizer (project fusion, identity elimination, pushdown), async batch stream execution. |
| **CSV read/write** | Custom bracket-aware tokenizer, type inference, configurable options. 65 C++ compat tests. |
| **JSON Lines read/write** | Full FlexType <-> JSON roundtrip. |
| **Groupby** | Hash-partitioned with disk spill + k-way merge. 13+ aggregators. |
| **Join** | Inner/Left/Right/Full. In-memory fast path + GRACE hash join for large inputs. |
| **External sort** | Greenwald-Khanna quantile sketch -> partition -> per-partition sort -> concatenate. |
| **Unique/dedup** | Hash-set for small, sort-based for large. |
| **CacheFs** | In-memory + disk spill with configurable budget. |
| **Configuration** | `SFRAME_*` env var overrides for memory budgets. |
| **Lazy source reading** | Segment-by-segment streaming (O(1 segment) memory). |
| **SFrame operations** | filter, sort, join, groupby, append, head/tail, sample, random_split, dropna, fillna, pack/unpack/stack, topk, unique, metadata. |
| **SArray operations** | Arithmetic, comparison, logical, reductions, string/dict ops, rolling aggregations, type casting, clip, sort, unique. |

---

## What's Missing or Incomplete

### Tier 1 -- Significant C++ Features Not Yet Ported

#### 1. Query Engine -- Missing Operators

The C++ query engine has operators the Rust version lacks:

| C++ Operator | What it does | Rust status |
|---|---|---|
| `TERNARY_OPERATOR` | `if/else` on three columns | Missing |
| `REDUCE_NODE` (full) | Aggregate reductions inside the plan DAG | Partial -- reductions go through materialization rather than being plan nodes |
| `BINARY_TRANSFORM_NODE` | Two-input column transform | Exists but SArray binary ops materialize the RHS first -- not truly streaming |

Note: `LAMBDA_TRANSFORM_NODE` (serialized Python closure execution) is N/A
for the Rust port. User-defined transforms exist via `apply()`.

#### 2. Query Optimizer -- Missing Passes

The C++ optimizer has ~6 transform families; Rust has 3:

| C++ Optimization | Rust status |
|---|---|
| Project fusion | Done |
| Identity elimination | Done |
| Projection pushdown through Filter | Done |
| **Source transforms** (replace chain with direct source) | Missing |
| **Logical filter transforms** (combine consecutive filters; push filters past transforms) | Missing |
| **Append transforms** (merge appends, eliminate empty sources) | Missing |
| **Union transforms** (merge unions, eliminate redundant) | Missing |

#### 3. Parallel Execution

The C++ SFrame parallelizes heavily via `parallel_for` over segments. The
Rust port is mostly single-threaded in its hot paths:

- CSV parsing: designed for rayon but called sequentially (noted in `csv_improvements.md`)
- External sort partition phase: single-threaded
- Source reading: sequential segment-by-segment (C++ reads segments in parallel with bounded concurrency)
- Transform/filter execution: sequential batch processing

#### 4. Rolling Aggregations (Query Engine Level)

The C++ has `rolling_aggregate.hpp` -- a general sliding-window aggregation
that works with all groupby aggregators. The Rust `SArray` has
`rolling_sum/mean/min/max` but they're simple hardcoded implementations,
not a general rolling framework.

#### 5. Sketches / Summary Statistics

| C++ Feature | Rust status |
|---|---|
| HyperLogLog (approximate cardinality) | Missing |
| Count-Min Sketch (frequency estimation) | Missing |
| Space-Saving (heavy hitters / frequent items) | Missing |
| `unity_sketch` (summary stats: mean, var, min, max, num_unique, quantiles, frequent items) | Missing -- SArray has individual reductions but no unified sketch |
| Streaming quantile sketch | Done (Greenwald-Khanna, for sort) |

#### 6. String Dictionary Encoding

The C++ string block codec supports optional dictionary encoding (dedup
common strings, store indices). The Rust string codec doesn't -- every
string is stored inline with varint-length prefix.

#### 7. Float Encoding

The C++ float codec has two modes: legacy FoR on bit-rotated doubles, and a
newer integer-encoded-float approach. The Rust float codec uses direct
8-byte encoding with a sentinel for UNDEFINED -- no compression at all.

### Tier 2 -- I/O and Infrastructure Gaps

#### 8. S3 and HDFS Backends

S3 backend implemented behind `s3` feature flag using `aws-sdk-s3` (requires
Rust 1.91+). Full-fetch read, buffered write, list/exists/remove.
HDFS remains a config stub.

#### 9. ODBC / Database Connector

The C++ has a full bidirectional ODBC connector (SQL query -> SFrame,
SFrame -> SQL table). Nothing exists in Rust.

#### 10. Avro Reader

The C++ has an Apache Avro reader. Nothing in Rust.

#### 11. CSV Parser -- Streaming/Parallel (DONE)

Two-pass streaming CSV with parallel parse via rayon. O(50MB) memory.
Remaining: no `continue_on_failure` error recovery; tokenizer operates
on `&str` not `&[u8]` (P3 optimization).

#### 12. Streaming CSV Writer (DONE)

`to_csv` now streams batches to disk via `for_each_batch_sync`.

### Tier 3 -- Correctness and Quality Gaps

#### 13. `SFrame::unique()` Hashing (DONE)

Fixed: uses `HashSet<Vec<FlexType>>` with proper `Hash+Eq`.

#### 14. SArray Binary Ops Materialization (DONE)

Fixed: same-plan case uses `GeneralizedTransform` (lazy streaming);
different-plan case materializes both sides eagerly (correct on re-execution).

#### 15. Block Statistics for Predicate Pushdown

The Rust writer doesn't store per-block min/max statistics. Block-skipping
during filtered reads isn't possible. (The C++ doesn't do this either --
it's a shared TODO.)

#### 16. LZ4 Compression on Write

The C++ writer applies LZ4 compression to blocks when the compression ratio
exceeds a threshold. The Rust reader supports LZ4 decompression, but the
writer may not compress. Needs verification.

#### 17. FlexImage in Storage

`FlexImage` is defined in `sframe-types` but not wired into the storage
codecs. Image columns can't be read or written.

#### 18. File Handle Pool / Block Cache

The C++ has a singleton `sarray_v2_block_manager` with:
- File handle pool (bounded number of open file handles)
- LRU decoded block cache (`SFRAME_MAX_BLOCKS_IN_CACHE`)

The Rust opens/closes files per segment read -- no handle pooling or
decoded block caching.

---

## Recommended Priority

| Priority | Item | Rationale |
|---|---|---|
| **P0** | Parallel CSV parsing | CSV is the primary ingestion path; currently single-threaded and memory-heavy |
| **P0** | Parallel CSV parsing | **DONE** — two-pass streaming + rayon parallel parse |
| **P0** | Parallel execution (rayon for transforms, source reading) | **DONE** — segment prefetch + intra-batch rayon |
| **P1** | Float compression (FoR encoding) | **DONE** — was already implemented |
| **P1** | String dictionary encoding | **DONE** — was already implemented |
| **P1** | LZ4 compression on write | **DONE** — was already implemented |
| **P1** | Missing optimizer passes | **DONE** — trivial unions, empty appends, filter-through-transform |
| **P1** | Fix `SFrame::unique()` hashing | **DONE** — proper `HashSet<Vec<FlexType>>` |
| **P1** | Fix SArray binary op materialization | **DONE** — GeneralizedTransform for same-plan |
| **P1** | Fix SArray::apply column index bug | **DONE** — Transform now outputs single column |
| **P2** | Streaming CSV writer | **DONE** — streams via `for_each_batch_sync` |
| **P2** | Sketch data structures (HyperLogLog, Space-Saving) | **DONE** — `approx_count_distinct`, `frequent_items` on SArray |
| **P2** | S3 backend | **DONE** — behind `s3` feature flag (requires Rust 1.91+) |
| **P2** | General rolling aggregation framework | Skipped — hardcoded rolling ops sufficient for now |
| **P3** | ODBC connector | Not started |
| **P3** | Avro reader | Not started |
| **P3** | File handle pool / block cache | Not started |

---

## Previously Completed Improvements

The following out-of-core / bounded-memory improvements have been completed.
Detailed design and implementation notes are preserved here for reference.

### Improvement 1: Lazy Segment-Level Source Reading (DONE)

**Problem**: `compile_sframe_source` read every segment file eagerly into
memory, collecting all results before emitting the first batch.

**Solution**: Replaced with `futures::stream::unfold`-based lazy stream that
reads one segment at a time. Memory is O(one segment) instead of O(all
segments). Added `read_block()`, `num_blocks()`, `block_num_elem()` to
`SegmentReader` for block-level random access.

### Improvement 2: External Sort / EC-Sort (DONE)

**Problem**: Sort materialized the entire input stream in memory.

**Solution**: 5-phase external sort in `sframe/src/external_sort.rs` using
Greenwald-Khanna quantile sketch (`sframe-query/src/algorithms/quantile_sketch.rs`)
for partition boundaries. `SFrame::sort()` dispatches to `sort_in_memory()`
or `external_sort()` based on `estimate_size()` vs `sort_memory_budget`.

### Improvement 3: GRACE Hash Join (DONE)

**Problem**: Join materialized both sides entirely and built a single
in-memory hash table.

**Solution**: `join()` dispatches to `in_memory_join()` when smaller side
fits in budget, otherwise `grace_hash_join()` which hash-partitions both
sides into N buckets and joins each partition independently. All 4 join
types supported.

### Improvement 4: Streaming Unique / Dedup (DONE)

**Solution**: `unique()` uses `unique_in_memory()` (HashSet) for small
arrays and `unique_via_sort()` (sort + streaming dedup) for large arrays.

### Improvement 5: Configuration System (DONE)

**Solution**: `SFrameConfig` with `LazyLock` + `AtomicUsize` in
`sframe-config` crate. Environment variable overrides via `SFRAME_*` prefix.
Keys: `source_batch_size`, `rows_per_segment`, `sort_memory_budget`,
`groupby_buffer_num_rows`, `join_buffer_num_cells`, `source_prefetch_segments`.

### Improvement 6: Block-Level Random Access (DONE)

**Solution**: Added `num_blocks()`, `block_num_elem()`, `read_block()` to
`SegmentReader`. `read_column()` refactored to use `read_block()` internally.

### Improvement 7: Buffer Pool (DROPPED)

Dropped as premature optimization -- allocation pressure not yet a
bottleneck.
