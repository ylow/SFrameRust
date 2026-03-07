# EC Sort (External Columnar Sort) Design

## Problem

The standard bucket sort reads and writes entire rows. For SFrames with many or
large value columns, this causes poor compression in row-wise buckets and high
memory usage. EC Sort separates sorting from data permutation, processing
columns independently.

## Algorithm Overview

1. **Sort key columns** with row numbers attached using existing external sort.
   Produces `inverse_map` (SArray) and `sorted_key_columns`.
2. **Compute forward_map** by permuting `[0..N)` using `inverse_map` via
   `permute_sframe`. Produces `forward_map` (SArray on CacheFs).
3. **Permute value columns** using the forward_map via `permute_sframe`.
   Produces `sorted_value_columns`.
4. **Assemble** sorted key columns + sorted value columns in original column order.

Steps 2 and 3 both use `permute_sframe`, the core new function.

## permute_sframe

`permute_sframe(input: SFrame, forward_map: SArray<u64>) -> SFrame`

`forward_map[i] = j` means input row `i` is written to output row `j`.

Two phases: scatter and permute.

### Scatter Phase

Distributes each column's values into M buckets (segments) based on the
forward_map. Each value at input row `r` goes to bucket
`forward_map[r] / rows_per_bucket`.

The forward_map itself is also scattered as the last column, with values
adjusted to per-bucket offsets (i.e., `forward_map[r] % rows_per_bucket`...
actually the raw value is stored; the permute phase subtracts the bucket start).

**Column-sequential processing:** Columns are processed one at a time during
scatter. This ensures blocks in each segment file are naturally in column order
(all column 0 blocks, then column 1, etc.), matching the existing segment
format. No format changes needed.

**Forward map re-reads:** The forward_map is re-read once per column. Since it's
a compressed SArray of integers (1-2 bytes/value with integer packing) on
CacheFs, subsequent reads are fast due to caching.

**Forward map chunking:** The forward_map is read in chunks that fit in memory
(controlled by the memory budget). For each chunk, the corresponding row range
of the current column is read and scattered.

### Permute Phase

For each bucket (in parallel across buckets):

1. Read the per-bucket forward_map (last column of scatter output).
2. Group remaining columns by estimated memory usage.
3. For each group of columns:
   - Read all values from this segment using existing SegmentReader.
   - Allocate permutation buffer: `rows_in_bucket x num_cols_in_group`.
   - Apply: `permute_buf[forward_map[r] - bucket_start][col] = values[r][col]`.
   - Write permuted columns to output segment via BufferedSegmentWriter.
   - Flush.
4. Output segment files assembled into final SFrame via
   `assemble_sframe_from_segments`.

### Bucket Count Determination

```
max_col_bytes = max(bytes_per_value[col] * num_rows for all cols)
num_buckets = ceil(max_col_bytes / (sort_max_memory / 2))
num_buckets *= num_cpus
num_buckets = max(1, num_buckets)
```

Ensures each bucket's largest column fits in `sort_max_memory / (2 * num_cpus)`.

### Column Size Estimation

Per-column byte estimates derived from block metadata (uncompressed block sizes
divided by element count), adjusted by type:
- Integer/Float/Datetime: fixed size (size_of FlexType)
- String/Vector: disk estimate + overhead
- Other (dict, list): disk estimate * 2 + overhead

## ScatterWriter

New type in `sframe-storage`. Manages M segment files and supports writing
values to specific (column, segment) pairs.

```
ScatterWriter {
    segment_writers: Vec<SegmentWriter<CacheFsFile>>
    column_types: Vec<FlexType>
    num_segments: usize
    per_segment_row_counts: Vec<Vec<u64>>  // [col][seg] -> count
}
```

**API:**
- `new(vfs, path, column_types, num_segments)` - creates M segment files
- `write_value(col_id, segment_id, value)` - append value to per-segment buffer;
  flush block when buffer reaches block_size
- `write_values(col_id, segment_id, &[FlexType])` - batch write
- `finish_column(col_id)` - flush remaining buffers for this column across all
  segments
- `finish() -> SFrame` - write segment footers, assemble SFrame metadata

**Memory:** M buffers x block_size per active column. With M=4000 and 8KB
buffers = 32MB. Segment files are on CacheFs with incremental disk spill.

## Forward Map Generation

1. Create SFrame: `[row_number (0..N), key_col_0, key_col_1, ...]`
2. Sort by key columns using existing external sort
3. Extract `inverse_map` = sorted row_number column (SArray)
4. Extract `sorted_key_columns` (kept for final assembly)
5. Create SArray `[0, 1, 2, ..., N-1]`
6. `forward_map = permute_sframe(range_sarray, inverse_map)`

`inverse_map[i] = j` means output position `i` came from input row `j`.
`forward_map[j] = i` means input row `j` goes to output position `i`.

## ec_sort Top-Level

```
fn ec_sort(input: &SFrame, key_indices: &[usize], sort_orders: &[bool]) -> SFrame
```

1. Sort key columns + row numbers -> sorted_keys, inverse_map
2. forward_map = permute_sframe([0..N), inverse_map)
3. sorted_values = permute_sframe(value_columns, forward_map)
4. Assemble sorted_keys + sorted_values in original column order

## Scope Decisions

- **No conditional dispatch**: ec_sort is a standalone function, caller chooses
- **No indirect columns**: all columns written directly (no row-number
  indirection for large-value columns)
- **Forward map on disk**: stored as SArray on CacheFs, read in chunks (supports
  10B+ rows)

## Code Organization

| File | Contents |
|------|----------|
| `crates/sframe-storage/src/scatter_writer.rs` | ScatterWriter |
| `crates/sframe/src/ec_sort.rs` | `permute_sframe`, `ec_sort` |

No changes to existing SFrameWriter, SegmentWriter, SegmentReader, or segment
format.

## Testing

- **ScatterWriter unit tests**: write known values, verify SFrame contents
- **permute_sframe tests**: known permutations, verify output row order
- **Forward map tests**: sort small dataset, verify forward_map correctness
- **ec_sort integration**: compare output against existing sort for same input
- **Scale test**: 100K+ rows, multi-column, mixed types
- **Edge cases**: single row, identical keys, already sorted, reverse sorted
