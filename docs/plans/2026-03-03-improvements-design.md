# SFrameRust Improvements Design

Date: 2026-03-03

## Overview

Five improvements to SFrameRust: foundational refactoring (FlexType Hash+Eq,
batch.rs macros, config unification) followed by two features (GRACE hash join,
streaming unique). Ordered so that refactoring creates a cleaner foundation for
the features.

## Design Philosophy

- Simple interfaces over fewer lines
- All configuration in one place — the user should not have to tune anything
- The engine should work under almost all circumstances, with performance as
  secondary

---

## 1. FlexType Hash + Eq

### Problem

`FlexType` derives `PartialEq` which uses standard float comparison (NaN != NaN).
This is wrong for hash table keys in groupby, join, and unique operations. The
codebase has two separate workarounds:

- `join.rs`: `CompositeKey` wrapper with `flex_eq()` + `hash_flex()`
- `groupby.rs`: Its own hashing via `FlexTypeKey`
- `SArray::unique()`: Uses `format!("{:?}")` as hash keys (!)

### Solution

Implement `Hash` and `Eq` directly on `FlexType` and `FlexDateTime`:

- `Float(f)`: Hash/compare via `f64::to_bits()` — NaN == NaN, -0.0 != 0.0
- `Vector(v)`: Hash each f64 element via `to_bits()`
- `List(l)`: Recurse into each element
- `Dict(d)`: Recurse into each (key, value) pair
- `DateTime(dt)`: Hash all three fields (timestamp, tz_offset, microsecond)
- `Undefined`: Equal to itself

Replace the derived `PartialEq` with a manual impl using the same `to_bits()`
logic for floats.

### Cleanup

- Remove `CompositeKey` from `join.rs` — use `Vec<FlexType>` directly
- Remove duplicate `FlexTypeKey` hashing from `groupby.rs`
- Fix `SArray::unique()` to use `HashSet<FlexType>`

### Trade-off

Changing `PartialEq` globally means NaN == NaN everywhere. This is the correct
semantic for a dataframe library (matching C++ SFrame, pandas, and SQL behavior).

---

## 2. batch.rs ColumnData Macro Cleanup

### Problem

`ColumnData` has 7 variants. Every operation requires a 7-arm match. Several
operations have identical logic per variant — the only difference is the enum
arm destructuring. This creates ~200 lines of boilerplate.

### Solution

A `with_column_data!` macro for operations where the logic is uniform across
all variants:

```rust
macro_rules! with_column_data {
    ($col:expr, $vec:ident => $body:expr) => {
        match $col {
            ColumnData::Integer($vec) => $body,
            ColumnData::Float($vec) => $body,
            ColumnData::String($vec) => $body,
            ColumnData::Vector($vec) => $body,
            ColumnData::List($vec) => $body,
            ColumnData::Dict($vec) => $body,
            ColumnData::DateTime($vec) => $body,
        }
    };
}
```

### Scope

Applies to: `len()`, `is_empty()`, `extend()`, `dtype()` (with a variant for
returning values).

Does NOT apply to: `push()`, `get()` — these have genuinely different logic per
variant (mapping between FlexType variants and ColumnData variants). These stay
as explicit matches.

### Principle

The macro eliminates boilerplate where logic is truly identical. Operations with
per-variant behavior stay explicit. You can see at a glance which operations are
uniform and which aren't.

---

## 3. Configuration Unification

### Problem

Two separate config systems with identical patterns:

- `sframe-config` crate: `AtomicUsize` globals + `Once` init (cache capacity)
- `sframe-query::config::SFrameConfig`: `LazyLock<struct>` (query tuning)

These are all system constraints at the same conceptual level. `sort_memory_budget`
and `cache_capacity` are both memory budgets. The user should not need to know
which subsystem a setting belongs to.

### Solution

Move all configuration into `sframe-config` using a single `SFrameConfig` struct:

```rust
pub struct SFrameConfig {
    // Cache settings
    pub cache_capacity: AtomicUsize,
    pub cache_capacity_per_file: AtomicUsize,

    // Query engine settings (immutable after init)
    pub source_batch_size: usize,
    pub rows_per_segment: u64,
    pub sort_memory_budget: usize,
    pub groupby_buffer_num_rows: usize,
    pub join_buffer_num_cells: usize,
    pub source_prefetch_segments: usize,
}
```

- `LazyLock<SFrameConfig>` for initialization from env vars
- Cache capacity fields use `AtomicUsize` for runtime mutation
- `get_cache_capacity()` / `set_cache_capacity()` API preserved
- `SFrameConfig::global()` for read access to all settings
- Remove `sframe-query::config` module entirely

### Environment Variables

All prefixed with `SFRAME_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SFRAME_CACHE_CAPACITY` | 2G | CacheFs in-memory store limit |
| `SFRAME_CACHE_CAPACITY_PER_FILE` | 128M | Max single file in cache |
| `SFRAME_SOURCE_BATCH_SIZE` | 4096 | Rows per batch |
| `SFRAME_ROWS_PER_SEGMENT` | 1000000 | Max rows per segment |
| `SFRAME_SORT_BUFFER_SIZE` | 256M | Sort memory budget |
| `SFRAME_GROUPBY_BUFFER_NUM_ROWS` | 1048576 | Groupby hash table limit |
| `SFRAME_JOIN_BUFFER_NUM_CELLS` | 50000000 | Join hash table limit |
| `SFRAME_SOURCE_PREFETCH_SEGMENTS` | 2 | Lazy source prefetch |

---

## 4. GRACE Hash Join

### Problem

`join.rs` materializes both sides entirely and builds a single in-memory hash
table. Datasets larger than RAM cannot be joined.

### Algorithm

**Phase 1 — Partition both sides by hash(key)**:

1. Estimate the smaller side's size from the first few batches using
   `estimate_batch_size()` (already in sort.rs)
2. Choose P = max(2, estimated_size / join_buffer_budget)
3. Stream left input: hash each row's join key, write to partition
   `hash(key) % P`. Each partition is a temp SFrame via `cache://` using
   `SFrameBuilder::anonymous()`
4. Stream right input: same partitioning

**Phase 2 — Per-partition hash join**:

5. For each partition p = 0..P:
   - Load the smaller side's partition into `HashMap<Vec<FlexType>, Vec<usize>>`
   - Stream the larger side's partition and probe the hash table
   - Emit matched rows as batches (and unmatched for LEFT/RIGHT/FULL)
6. Stream partition results sequentially

### Interface Change

```rust
// Before
pub async fn join(...) -> Result<SFrameRows>

// After
pub async fn join(...) -> Result<BatchStream>
```

The in-memory fast path wraps its result in `stream::once()`. The GRACE path
returns a stream that processes partitions one at a time via `stream::unfold()`.

`SFrame::join()` consumes the returned stream via `SFrameBuilder`, writing
batches incrementally. The join output never needs to be fully in memory.

### Fast Path

If estimated cells (rows x columns) for both sides fit within
`join_buffer_num_cells`, use the current in-memory hash join directly.

### Memory Bound

O(join_buffer_size) — only one partition's hash table is in memory at a time.

### Unmatched Row Tracking for Outer Joins

For LEFT/FULL joins: unmatched left rows are emitted immediately during the
probe phase (no match found → emit with NULL-padded right).

For RIGHT/FULL joins: track a `BitVec` per partition for right-side matches.
After probing, sweep unmatched right rows.

---

## 5. Streaming Unique

### Problem

`SArray::unique()` materializes the entire array via `to_vec()` and uses
`HashSet<String>` with debug-format keys — both slow and semantically incorrect.

### Solution: Sort-Based Unique (Large Data)

1. Sort the array using existing external sort infrastructure
2. Single streaming pass removing consecutive duplicates

This reuses the external sort with zero new spill infrastructure. The dedup
pass is ~10 lines of code.

Note: sort-based unique does NOT preserve first-occurrence order. For out-of-core
data this is impractical anyway.

### Solution: HashSet (Small Data, Fast Path)

If estimated data fits in memory, use `HashSet<FlexType>` directly. This
preserves first-occurrence order and is faster for small inputs.

### Size Threshold

Use the same `estimate_batch_size()` logic as sort to decide between in-memory
and sort-based paths.

---

## Implementation Order

1. **FlexType Hash + Eq** — prerequisite for items 4 and 5
2. **batch.rs macros** — reduces noise when implementing GRACE join
3. **Configuration unification** — GRACE join config goes in the right place
4. **GRACE hash join** — largest feature, benefits from all three refactors
5. **Streaming unique** — smallest, reuses external sort + new FlexType Hash

---

## What This Design Does NOT Cover

- Buffer pool / arena allocation (dropped — modern allocators handle this)
- Predicate pushdown with block statistics (orthogonal)
- Parallel GRACE join (can be added later — start single-threaded for correctness)
- Making join a planner-level operator (currently called directly from SFrame API)
