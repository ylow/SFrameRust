# SFrame Rust — Gap Closure Plan

## Status

The core 5-phase port is complete: types, IO, storage (read/write), query engine,
and top-level SFrame/SArray API. All 120 tests pass. The Rust port can read and
write the C++ V2 format, load CSVs, and perform filter/sort/join/groupby.

This document lists everything the C++ SFrame supports that the Rust port is
missing or has only partially implemented, organized by priority.

---

## Phase 6: Out-of-Core Execution (Fundamental)

The defining property of SFrame is that it processes datasets larger than memory.
The C++ achieves this through a "never fully materialize" principle: at no point
in the pipeline is the full dataset held in memory. The Rust port currently
materializes everything into memory for sort, join, groupby, and even simple
filter/head operations. **This is the most architecturally significant gap.**

The C++ implements out-of-core through five interlocking patterns:

1. **Batch pipeline streaming** — operators pull fixed-size batches (256 rows)
   through coroutines; memory is O(batch_size × pipeline_depth), not O(dataset)
2. **Segment parallelism** — data is physically partitioned into independent
   segments that can be read, processed, and written without loading neighbors
3. **Incremental buffered I/O** — readers use 1024-row buffers; writers flush
   blocks to disk when soft/hard memory limits are hit
4. **Spillover with memory budgets** — groupby/join maintain explicit memory
   budgets and spill excess state to temp SArrays on disk
5. **Indirect permutation** — sort builds a forward-map of integers (tiny) then
   applies it to value columns in a streaming pass, never shuffling the full data

### 6.1 Streaming Execution Engine

Replace the current "compile → materialize everything" model with true
streaming batch execution.

**Current problem:** `materialize_sync()` collects all batches into one giant
`SFrameRows` in memory. Every SFrame operation (filter, head, sort, etc.) calls
`materialize_batch()` which does this.

**Target architecture:**
- `BatchStream` already exists as `Pin<Box<dyn Stream<Item=Result<SFrameRows>>>>`.
  The stream infrastructure is in place — the issue is that consumers always
  drain the entire stream into memory immediately.
- `SFrame::head(n)` should pull only enough batches to get n rows, then stop.
- `SFrame::filter()` should return a lazy SFrame whose plan node wraps a
  filter operator, not materialize-filter-rematerialize.
- `SFrame::save()` should stream batches directly to the segment writer
  without collecting them first.
- Operators like Project, Filter, Transform should remain streaming (they
  already are in the async compile step — the problem is the final
  materialization).

**Configurable constants:**
- `SOURCE_BATCH_SIZE` (currently 4096, C++ uses 256 — tune to taste)
- `WRITER_BUFFER_SOFT_LIMIT` — start flushing writer blocks
- `WRITER_BUFFER_HARD_LIMIT` — force flush

**Files:** `crates/sframe-query/src/execute.rs`, `crates/sframe/src/sframe.rs`,
`crates/sframe/src/sarray.rs`

### 6.2 Streaming Writer

The current `write_sframe()` takes `&[Vec<FlexType>]` (all rows in memory).
Replace with a streaming writer that accepts a `BatchStream` and writes blocks
incrementally.

- `SFrameStreamWriter::new(path, column_names, column_types, num_segments)`
- `write_batch(batch: &SFrameRows)` — encode and flush blocks
- `finish()` — write segment footers and index files
- Memory budget: flush blocks when buffered data exceeds soft limit

**Files:** `crates/sframe-storage/src/sframe_writer.rs`,
`crates/sframe-storage/src/segment_writer.rs`

### 6.3 Multi-Segment Read/Write

The current writer always produces a single segment. The current reader loads
an entire column at once.

- Writer: configurable `num_segments`, auto-split based on row count
  (e.g., 1M rows per segment), round-robin or hash-based assignment
- Reader: segment-at-a-time iteration; `SFrameSource` operator should emit
  batches from one segment at a time, not load all segments
- This enables future parallel segment processing (Phase 14)

**Files:** `crates/sframe-storage/src/sframe_writer.rs`,
`crates/sframe-query/src/execute.rs`

### 6.4 External Columnar Sort (EC-Sort)

The C++ EC-Sort avoids shuffling large data by:
1. Sorting only the key columns + row numbers
2. Double-sorting to produce a forward-map (permutation of integers)
3. Applying the forward-map to value columns in a single streaming pass

**Algorithm:**
```
A = key_columns.add_row_number("orig_row")
B = A.sort(key_columns)              // sort small data
C = B.add_row_number("dest_row")
C = C.sort("orig_row")               // restore original order
forward_map = C["dest_row"]          // forward_map[i] = destination of row i
result = apply_forward_map(all_columns, forward_map)
```

For data that fits in memory, fall back to the current in-memory sort.
Threshold: if `num_rows * num_value_columns * avg_cell_size > memory_budget`.

**Files:** `crates/sframe-query/src/algorithms/sort.rs`

### 6.5 GRACE Hash Join

The C++ uses GRACE (Generalized Resolution and Compute Environment) partitioned
hash join:

1. **Partition phase:** Hash-partition both left and right tables on join key
   into N partitions (N = total_cells / budget). Rows with the same join key
   land in the same partition on both sides.
2. **Join phase:** For each partition i, load left[i] into a hash table,
   probe with right[i]. Only one partition pair is in memory at a time.

Memory budget: `JOIN_BUFFER_NUM_CELLS` (C++ default: 50M cells).
If the dataset fits in the budget, skip partitioning and do a single-pass
hash join (current behavior).

**Files:** `crates/sframe-query/src/algorithms/join.rs`

### 6.6 Spillable Groupby

The C++ groupby uses per-segment hash tables with a row budget
(`GROUPBY_BUFFER_NUM_ROWS`, default 1M). When a segment's hash table exceeds
the budget, its contents are serialized and spilled to a temp SArray on disk.
After the input is exhausted, spilled chunks are re-read and merged.

**Strategy:**
1. Hash each row's group key to assign it to a segment (not the input segment —
   an output segment chosen by `hash(key) % num_output_segments`)
2. Each output segment has its own hash table of aggregator states
3. When a segment's hash table exceeds the row budget, serialize and flush
4. After input is exhausted, re-read spilled chunks and merge aggregator states

For data that fits in memory, fall back to current in-memory groupby.

**Files:** `crates/sframe-query/src/algorithms/groupby.rs`

### 6.7 Memory Budget System

Centralize memory budget constants (like C++'s `sframe_constants.hpp`):

```rust
pub struct SFrameConfig {
    pub source_batch_size: usize,           // default 4096
    pub writer_buffer_soft_limit: usize,    // default 8MB
    pub writer_buffer_hard_limit: usize,    // default 32MB
    pub groupby_buffer_num_rows: usize,     // default 1_048_576
    pub join_buffer_num_cells: usize,       // default 50_000_000
    pub sort_memory_budget: usize,          // default 256MB
    pub reader_buffer_size: usize,          // default 1024
}
```

Thread-local or passed through execution context.

**Files:** new `crates/sframe-query/src/config.rs`

---

## Phase 7: CSV Parser (Structured Data Parser)

The SFrame CSV parser is not a typical CSV library. It is a **structured data
parser** that happens to use CSV as a transport format. It can parse vectors,
dictionaries, and lists directly from CSV cells — no post-processing needed.
Our current parser delegates to the `csv` crate and only infers int/float/string.

### 7.1 Flexible Type Parser (Recursive Descent)

The core of the C++ CSV parser is `flexible_type_spirit_parser` which parses
a string into any FlexType. The Rust equivalent should be a standalone function:

```rust
pub fn parse_flextype(s: &str) -> FlexType
```

**Supported syntax:**
- Integers: `123`, `-42`
- Floats: `1.5`, `-3.14e10`, `inf`, `nan`
- Vectors: `[1, 2, 3]` or `[1 2 3]` or `[1;2;3]` (flexible separators:
  comma, space, semicolon)
- Lists: `[a, "b", 3, 4.5]` (heterogeneous — mixed types)
- Dicts: `{"key": "value", 1: 2}` (JSON-like with mixed-type keys/values)
- Strings: anything that doesn't parse as a structured type

**Lookahead logic:**
When the parser sees `[` or `{`, it attempts to parse the entire bracketed
structure. If parsing fails (mismatched brackets, invalid content), it falls
back to treating the entire field as a string. This is what lets
`[abc` parse as the string `"[abc"` instead of erroring.

**Files:** new `crates/sframe-types/src/flex_type_parser.rs`

### 7.2 CSV Tokenizer

Replace the `csv` crate dependency with a custom tokenizer that understands
SFrame's rules:

**Features the `csv` crate cannot do:**
- Nested bracket/brace parsing inside unquoted fields (the `csv` crate treats
  `[1,2,3]` as three fields because it splits on commas)
- Recursive type inference after unquoting (`"123"` → integer 123, not string)
- Configurable NA value sets (`{"NA", "Pokemon", ""}` → Undefined)
- Comment lines (`# this is a comment`)
- Arbitrary line terminators (`zzz` instead of `\n`)
- C-style escape sequences in quoted fields (`\n` → newline, `\t` → tab)
- Double-quote escaping (`""` → `"` per RFC 4180)
- Space as delimiter with special excess-whitespace handling
- Column subsetting (skip unwanted columns during parse, not after)
- Multi-character delimiters

**Tokenizer configuration:**
```rust
pub struct CsvOptions {
    pub delimiter: String,          // default ","  (can be multi-char)
    pub line_terminator: String,    // default "\n" (matches \r, \r\n too)
    pub escape_char: char,          // default '\\'
    pub quote_char: char,           // default '"'
    pub double_quote: bool,         // default false
    pub skip_initial_space: bool,   // default true
    pub comment_char: Option<char>, // default Some('#')
    pub has_header: bool,           // default true
    pub na_values: Vec<String>,     // default empty
    pub type_hints: HashMap<String, FlexTypeEnum>,
    pub skip_rows: usize,           // default 0
    pub row_limit: Option<usize>,   // default None
    pub output_columns: Option<Vec<String>>,  // column subsetting
}
```

**Files:** new `crates/sframe-query/src/algorithms/csv_tokenizer.rs`,
modify `crates/sframe-query/src/algorithms/csv_parser.rs`

### 7.3 Type Inference with Complex Types

The current type inference only considers integer, float, and string.
Extend to consider all FlexType variants:

**Inference priority** (per column, first type that parses all non-NA values):
1. Integer
2. Float
3. Vector (if all values are `[number, ...]`)
4. List (if all values are `[mixed, ...]`)
5. Dict (if all values are `{key: value, ...}`)
6. DateTime (if a date format is detected)
7. String (fallback)

When `type_hints` are provided for a column, skip inference and parse
directly as the hinted type. Values that fail to parse as the hinted type
become Undefined (if `undefined_on_failure` is true) or error.

**Files:** `crates/sframe-query/src/algorithms/csv_parser.rs`

### 7.4 Test Suite (Port from C++)

Port the 24 test cases from `SFrame/oss_test/sframe/sframe_csv_test.cxx`:

| Test | What it covers |
|------|---------------|
| `basic` | One row with float, int, string, vector, dict, list |
| `basic_comments_and_skips` | `#` comment lines, `skip_rows` |
| `quoted_basic` | All fields double-quoted with escapes |
| `test_type_inference` | Auto-detect all types (UNDEFINED hint) |
| `test_quoted_type_inference` | Quoted values still type-infer |
| `test_embedded_strings` | `[abc`, `cde]`, `a[a]b` are strings not structures |
| `test_quoted_embedded_strings` | Same with escaped quoting |
| `interesting` | Semicolon delimiter, inline comments, double-quote escaping, embedded delimiters, NA values, missing trailing fields |
| `excess_white_space` | Space delimiter with dicts containing spaces |
| `wierd_bracketing_thing` | Mismatched brackets → string fallback |
| `another_wierd_bracketing_thing_issue_1514` | Tab-delimited bracket chars |
| `string_integers` | `"""1"""` stays string, not integer |
| `string_integers2` | `"1"` becomes integer (recursive parse) |
| `alternate_endline_test` | `zzz` as line terminator |
| `escape_parsing` | `\n` → newline, `\t` → tab, `\"` → quote |
| `escape_parsing_string_hint` | Same with STRING type hint |
| `non_escaped_parsing` | Unquoted backslashes are literal |
| `single_string_column` | Newline as delimiter |
| `unicode_surrogate_pairs` | `\uD834\uDD1E` → 𝄞, invalid surrogates preserved |
| `multiline_json` | Empty delimiter/terminator, multi-line dict |
| `tab_delimited_csv_with_list` | Tab-delimited with list columns |
| `test_na_values` | Custom NA strings (`NA`, `PIKA`, `CHU`) |
| `test_na_values2` | Numeric NA values (`-8` → Undefined) |
| `test_missing_tab_values` | Empty tab-delimited fields → Undefined |

**Files:** `crates/sframe-query/tests/csv_parser_tests.rs`

### 7.5 CSV Writer

- `SFrame::to_csv(path, options) -> Result<()>`
- `CsvWriterOptions { delimiter, quote_char, escape_char, line_terminator, na_rep, header, quoting: QuoteStyle }`
- `QuoteStyle` enum: `Minimal`, `All`, `NonNumeric`, `None`
- Vectors/lists/dicts serialize back to bracket notation

**Files:** new `crates/sframe-query/src/algorithms/csv_writer.rs`,
`crates/sframe/src/sframe.rs`

---

## Phase 8: FlexType Operators & SArray Element-wise Operations

The C++ `flexible_type` has full arithmetic and comparison operator overloads.
The C++ `unity_sarray` exposes element-wise binary operations between arrays
and between arrays and scalars. The Rust `FlexType` has none of this — users
must use `apply()` with closures for everything.

### 8.1 FlexType Arithmetic Operators

Implement `std::ops` traits on `FlexType`:

- `Add` (`+`): int+int→int, float+float→float, int+float→float, string+string→concat, vector+vector→element-wise
- `Sub` (`-`): numeric types, vector element-wise
- `Mul` (`*`): numeric types, vector-scalar
- `Div` (`/`): always returns Float (like C++)
- `Rem` (`%`): integer only
- `Neg` (unary `-`): numeric and vector

**Files:** `crates/sframe-types/src/flex_type.rs`

### 8.2 FlexType Comparison & Ordering

- Implement `PartialOrd` on `FlexType` with cross-type numeric comparison
- Undefined sorts last (consistent with C++)

**Files:** `crates/sframe-types/src/flex_type.rs`

### 8.3 SArray Element-wise Binary Operations

Add methods to `SArray`:

- `add(&self, other: &SArray) -> SArray` (and `add_scalar`)
- `sub`, `mul`, `div`, `rem` (array-array and array-scalar variants)
- `eq`, `ne`, `lt`, `le`, `gt`, `ge` → returns `SArray` of Integer (0/1)
- `and`, `or` → logical operations on integer arrays
- `abs` → element-wise absolute value
- `power` → exponentiation
- `floor_div` → integer division

These should build lazy plan nodes (BinaryTransform / Transform), not
materialize eagerly.

**Files:** `crates/sframe/src/sarray.rs`, `crates/sframe-query/src/planner.rs`

### 8.4 SArray Type Casting

- `astype(dtype, undefined_on_failure: bool) -> SArray`
  - int↔float, numeric→string, string→numeric (parse), datetime→string, string→datetime

**Files:** `crates/sframe/src/sarray.rs`

---

## Phase 9: Missing Aggregators

The C++ has 17+ aggregators. We have 8 implemented but only 5 wired to
`AggSpec` convenience constructors.

### 9.1 Wire Existing Aggregators to AggSpec

Add `AggSpec` convenience constructors for the three that already exist:

- `AggSpec::variance(column, output_name)`
- `AggSpec::stddev(column, output_name)`
- `AggSpec::concat(column, output_name)`

**Files:** `crates/sframe-query/src/algorithms/aggregators.rs`

### 9.2 New Aggregators

Implement and wire to `AggSpec`:

- `QuantileAggregator(quantile: f64)` — collect values, sort, pick percentile
- `CountDistinctAggregator` — count unique values (HashSet-based)
- `NonNullCountAggregator` — count non-Undefined values
- `FrequencyCountAggregator` — return Dict of value→count
- `SelectOneAggregator` — return first non-Undefined value in group
- `ArgMinAggregator(value_col)` — value of another column at the row with min
- `ArgMaxAggregator(value_col)` — value of another column at the row with max
- `ZipListAggregator` — collect group values into a List
- `ZipDictAggregator(key_col, value_col)` — collect into Dict
- `VectorSumAggregator` — element-wise sum of Vector columns
- `VectorAvgAggregator` — element-wise mean of Vector columns

**Files:** `crates/sframe-query/src/algorithms/aggregators.rs`

---

## Phase 10: Missing SArray Operations

### 10.1 Core Missing Operations

- `tail(n)` — last n elements
- `sort(ascending: bool)` — sort the array itself
- `unique()` — deduplicated values
- `append(other: &SArray)` — concatenate two arrays
- `sample(fraction: f64, seed: Option<u64>)` — random sample
- `hash(seed: u64)` — hash each element

**Files:** `crates/sframe/src/sarray.rs`

### 10.2 Missing Value Handling

- `countna() -> u64` — count Undefined values
- `dropna() -> SArray` — remove Undefined values
- `fillna(value: FlexType) -> SArray` — replace Undefined with value
- `is_na() -> SArray` — returns Integer array (1 where Undefined)

**Files:** `crates/sframe/src/sarray.rs`

### 10.3 Numeric Operations

- `clip(lower: FlexType, upper: FlexType) -> SArray` — clamp values
- `cumulative_sum() -> SArray`
- `cumulative_min() -> SArray`
- `cumulative_max() -> SArray`
- `cumulative_avg() -> SArray`

**Files:** `crates/sframe/src/sarray.rs`

### 10.4 Reduction Operations

These return a single value, not an SArray:

- `sum() -> FlexType`
- `min() -> FlexType`
- `max() -> FlexType`
- `mean() -> FlexType`
- `std(ddof: u8) -> FlexType`
- `var(ddof: u8) -> FlexType`
- `any() -> bool`
- `all() -> bool`
- `nnz() -> u64` — count non-zero elements
- `num_missing() -> u64`

**Files:** `crates/sframe/src/sarray.rs`

### 10.5 String Operations (for String-typed SArrays)

- `count_bag_of_words(options)` — word frequency per element
- `count_ngrams(n, options)` — n-gram frequency per element
- `count_character_ngrams(n, options)` — character n-gram frequency
- `contains(substring) -> SArray<Integer>` — substring search

**Files:** `crates/sframe/src/sarray.rs` (or a new `sarray_string_ops.rs`)

### 10.6 Dict Operations (for Dict-typed SArrays)

- `dict_keys() -> SArray<List>`
- `dict_values() -> SArray<List>`
- `dict_trim_by_keys(keys, exclude) -> SArray`
- `dict_trim_by_values(lower, upper) -> SArray`
- `dict_has_any_keys(keys) -> SArray<Integer>`
- `dict_has_all_keys(keys) -> SArray<Integer>`

**Files:** `crates/sframe/src/sarray.rs` (or a new `sarray_dict_ops.rs`)

### 10.7 Structured Data Operations

- `item_length() -> SArray<Integer>` — length of vector/list/dict/string elements
- `vector_slice(start, end) -> SArray` — slice vector subarrays
- `unpack(prefix, keys, types) -> SFrame` — unpack dict/list to columns

**Files:** `crates/sframe/src/sarray.rs`

### 10.8 Rolling / Windowed Aggregations

- `rolling_sum(before, after, min_observations)`
- `rolling_mean(before, after, min_observations)`
- `rolling_min(before, after, min_observations)`
- `rolling_max(before, after, min_observations)`

**Files:** `crates/sframe/src/sarray.rs`

---

## Phase 11: Missing SFrame Operations

### 11.1 Column Mutation

- `replace_column(name, col) -> SFrame`
- `rename(mapping: HashMap<&str, &str>) -> SFrame`
- `swap_columns(name1, name2) -> SFrame`

**Files:** `crates/sframe/src/sframe.rs`

### 11.2 Missing Value Handling

- `dropna(column_name_or_all, how: Any|All) -> SFrame`
- `fillna(column_name, value) -> SFrame`

**Files:** `crates/sframe/src/sframe.rs`

### 11.3 Sampling & Splitting

- `sample(fraction, seed) -> SFrame`
- `random_split(fraction, seed) -> (SFrame, SFrame)`
- `topk(column, k, reverse) -> SFrame`

**Files:** `crates/sframe/src/sframe.rs`

### 11.4 Reshaping

- `pack_columns(columns, new_column_name, dtype) -> SFrame` — pack multiple columns into one dict/list/vector column
- `unpack_column(column_name, prefix) -> SFrame` — unpack dict/list/vector column into multiple columns
- `stack(column_name, new_column_name) -> SFrame` — unnest list/dict column (one row per element)
- `unstack(column, new_column_name) -> SFrame` — reverse of stack

**Files:** `crates/sframe/src/sframe.rs`

### 11.5 Deduplication

- `unique() -> SFrame` — deduplicate rows

**Files:** `crates/sframe/src/sframe.rs`

### 11.6 Multi-Column Join Keys

Currently join only supports a single column key. The C++ supports joining on
multiple columns.

- `join(other, on: &[(&str, &str)], how) -> SFrame`

**Files:** `crates/sframe/src/sframe.rs`, `crates/sframe-query/src/algorithms/join.rs`

### 11.7 Tail

- `tail(n) -> SFrame` — last n rows

**Files:** `crates/sframe/src/sframe.rs`

---

## Phase 12: Query Optimizer

The C++ has optimizer passes between the logical plan and physical execution.
The Rust port compiles the DAG directly.

- Predicate pushdown — push filters closer to sources
- Projection pushdown — only read needed columns from SFrame source
- Common subexpression elimination — share materialized results
- Constant folding
- Operator fusion — merge adjacent project/filter nodes

**Files:** new `crates/sframe-query/src/optimizer.rs`

---

## Phase 13: Parallel Execution

The C++ uses OpenMP for parallel segment processing. The Rust port is
single-threaded.

### 13.1 Parallel Segment Reads

- Read multiple segments concurrently using tokio tasks or rayon
- Configurable parallelism level

**Files:** `crates/sframe-query/src/execute.rs`

### 13.2 Parallel Segment Writes

- Write segments in parallel using thread pool
- Shard rows across segments before writing

**Files:** `crates/sframe-storage/src/sframe_writer.rs`

### 13.3 Parallel Transforms

- Partition batch streams across worker tasks
- Fan-out / fan-in for CPU-bound transforms

**Files:** `crates/sframe-query/src/execute.rs`

---

## Phase 14: JSON Support

- `SFrame::to_json(path) -> Result<()>` — write rows as JSON lines
- `SFrame::from_json(path) -> Result<SFrame>` — read JSON lines
- `FlexType` ↔ JSON serialization (serde_json)

**Files:** new `crates/sframe-query/src/algorithms/json.rs`,
`crates/sframe-types/src/flex_type.rs`

---

## Phase 15: Storage Backends

The VFS trait is already designed for pluggable backends. The C++ supports
S3, HDFS, and HTTP.

### 15.1 S3 Backend

- `S3FileSystem` implementing `VirtualFileSystem`
- Seekable reads via range requests
- Multipart upload for writes
- AWS credential chain (env vars, config file, instance profile)

**Files:** new `crates/sframe-io/src/s3_fs.rs`
**Dependencies:** `aws-sdk-s3`

### 15.2 HTTP/HTTPS Backend (Read-Only)

- `HttpFileSystem` implementing read portion of `VirtualFileSystem`
- Range request support for seeking
- Response caching

**Files:** new `crates/sframe-io/src/http_fs.rs`
**Dependencies:** `reqwest`

### 15.3 HDFS Backend

- `HdfsFileSystem` implementing `VirtualFileSystem`
- Via WebHDFS REST API

**Files:** new `crates/sframe-io/src/hdfs_fs.rs`

---

## Phase 16: Miscellaneous

### 16.1 SFrame/SArray Metadata

- `get_metadata(key) -> Option<String>`
- `set_metadata(key, value)`
- Persist metadata through save/load cycle
- Read existing metadata from C++ SFrame files

**Files:** `crates/sframe/src/sframe.rs`, `crates/sframe-storage/src/sframe_writer.rs`

### 16.2 Lazy Evaluation Caching (Memoization)

Currently, every materialization re-executes the full DAG. The C++ caches
computed results.

- Cache computed `SFrameRows` on plan nodes
- Invalidation when inputs change
- Memory-bounded LRU eviction

**Files:** `crates/sframe-query/src/execute.rs`, `crates/sframe-query/src/planner.rs`

### 16.3 Block Skipping in Reader

The C++ reader can skip blocks that don't match filter predicates (min/max
statistics per block).

- Store per-block min/max statistics in segment footer
- Skip blocks during filtered reads

**Files:** `crates/sframe-storage/src/segment_reader.rs`,
`crates/sframe-storage/src/segment_writer.rs`

### 16.4 Read Caching & File Handle Pooling

- Block cache (LRU) for frequently accessed blocks
- File handle pool to avoid repeated open/close
- Read-ahead for sequential access patterns

**Files:** `crates/sframe-io/src/cache.rs`

### 16.5 Image Type

The C++ has `flex_image` as a first-class type. Low-priority — specialized
for ML workflows.

- `FlexType::Image(ImageData)` variant
- Pixel array encoding/decoding

**Files:** `crates/sframe-types/src/flex_type.rs`

---

## Priority Order

1. **Phase 6** — Out-of-core execution (fundamental architectural gap; everything
   else builds on this)
2. **Phase 7** — CSV parser with structured type parsing (unique capability;
   no substitute)
3. **Phase 8** — FlexType operators & SArray element-wise ops (most visible
   user-facing gap)
4. **Phase 9** — Missing aggregators
5. **Phase 10.1–10.4** — Core SArray operations (tail, sort, unique, dropna,
   reductions)
6. **Phase 11.1–11.3** — SFrame column mutations, missing values, sampling
7. **Phase 11.6** — Multi-column join keys
8. **Phase 12** — Query optimizer
9. **Phase 10.5–10.8** — String/dict/rolling ops (specialized)
10. **Phase 11.4–11.5** — Reshaping, deduplication
11. **Phase 7.5** — CSV writer
12. **Phase 13** — Parallel execution
13. **Phase 14** — JSON support
14. **Phase 15** — Storage backends (S3, HTTP, HDFS)
15. **Phase 16** — Miscellaneous (metadata, caching, block skipping, images)
