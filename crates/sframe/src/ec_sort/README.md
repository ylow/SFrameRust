# EC Sort (External Columnar Sort)

## The Problem

Consider sorting a 100-million-row SFrame with 50 columns by one key column.
A standard sort must load and rearrange *all 50 columns at once*. If each
row is ~400 bytes, that's 40GB of data that must fit in memory simultaneously.

EC Sort avoids this by observing that sorting really has two separable concerns:

1. **Deciding the order** — only the key column(s) matter.
2. **Moving the data** — every column must be rearranged, but each column can
   be moved independently.

EC Sort exploits this separation. It sorts only the key columns (cheap),
computes a *permutation map*, then applies that map to each value column
independently (bounded memory per column).

## High-Level Algorithm (4 Steps)

```
Input: SFrame with columns [K1, K2, V1, V2, ..., Vn]
       Sort by K1 ascending, K2 descending

Step 1: Sort keys + row numbers
        Build a mini SFrame: [K1, K2, row_number]
        where row_number = [0, 1, 2, ..., N-1]
        Sort this mini SFrame by (K1 asc, K2 desc).
        Result: sorted key columns + inverse_map (the sorted row_number column).

Step 2: Invert the permutation
        Compute forward_map from inverse_map.
        forward_map[i] tells us where input row i should go in the output.

Step 3: Permute value columns
        Use the forward_map to rearrange [V1, V2, ..., Vn]
        into their sorted positions. (Processes one column group at a time.)

Step 4: Reassemble
        Interleave sorted key columns and permuted value columns
        back into original column order.
        Result: fully sorted SFrame.
```

### Concrete Example

Start with 6 rows, sort by column `name` ascending:

```
Input:
  row  | name    | score | city
  -----+---------+-------+---------
  0    | Charlie | 80    | Denver
  1    | Alice   | 95    | Boston
  2    | Eve     | 70    | Seattle
  3    | Bob     | 88    | Austin
  4    | Alice   | 92    | Chicago
  5    | Dave    | 76    | Miami
```

**Step 1** — Sort `[name, row_number]` by name:

```
Sorted keys:
  output_pos | name    | row_number (= inverse_map)
  -----------+---------+---
  0          | Alice   | 1
  1          | Alice   | 4
  2          | Bob     | 3
  3          | Charlie | 0
  4          | Dave    | 5
  5          | Eve     | 2
```

The `inverse_map` is `[1, 4, 3, 0, 5, 2]` — it answers "which input row
goes to output position p?" For output position 0, input row 1 (Alice).

**Step 2** — Invert to get `forward_map`:

```
inverse_map[0] = 1  →  forward_map[1] = 0
inverse_map[1] = 4  →  forward_map[4] = 1
inverse_map[2] = 3  →  forward_map[3] = 2
inverse_map[3] = 0  →  forward_map[0] = 3
inverse_map[4] = 5  →  forward_map[5] = 4
inverse_map[5] = 2  →  forward_map[2] = 5

forward_map = [3, 0, 5, 2, 1, 4]
```

The `forward_map` answers "where does input row i go?" Row 0 (Charlie)
goes to output position 3. Row 1 (Alice) goes to output position 0.

**Step 3** — Permute the value columns using the forward map:

```
score column:  input = [80, 95, 70, 88, 92, 76]
  forward_map[0]=3 → output[3] = 80
  forward_map[1]=0 → output[0] = 95
  forward_map[2]=5 → output[2] = 70  (wait, this is wrong... let me re-derive)
```

Actually: `output[forward_map[i]] = input[i]`

```
  output[3] = input[0] = 80
  output[0] = input[1] = 95
  output[5] = input[2] = 70
  output[2] = input[3] = 88
  output[1] = input[4] = 92
  output[4] = input[5] = 76
  → output = [95, 92, 88, 80, 76, 70]
```

**Step 4** — Reassemble:

```
Output:
  row  | name    | score | city
  -----+---------+-------+---------
  0    | Alice   | 95    | Boston
  1    | Alice   | 92    | Chicago
  2    | Bob     | 88    | Austin
  3    | Charlie | 80    | Denver
  4    | Dave    | 76    | Miami
  5    | Eve     | 70    | Seattle
```

### Why the Forward Map?

We need the **forward** map (not the inverse map) for the value column
permutation because we're *reading input sequentially* and need to know
where each input row should be *written*. Reading sequentially is critical
for streaming — we never need all rows in memory at once.

If we used the inverse map instead, we'd need to read random input
positions (since `inverse_map[output_pos]` tells us *which input row* goes
to each output position), which requires random access to the full input.


## The Scatter-Permute Algorithm (`permute_sframe`)

The forward map gives us a complete permutation, but applying it naively
(`output[forward_map[i]] = input[i]`) requires the entire output to be in
memory. For a 100M-row column, that could be gigabytes.

The solution: **partition the output space into buckets**, so we only need
one bucket's worth of data in memory at a time.

### Bucketing

Divide the output row space `[0, N)` into M contiguous buckets:

```
Bucket 0: output rows [0, rows_per_bucket)
Bucket 1: output rows [rows_per_bucket, 2*rows_per_bucket)
...
Bucket M-1: output rows [(M-1)*rows_per_bucket, N)
```

Each input row's bucket is determined by its forward_map value:

```
bucket(i) = forward_map[i] / rows_per_bucket
```

### Phase 1: Scatter

Stream through the input and forward_map together. For each input row `i`,
route its values to `bucket(i)`:

```
For each input row i (streamed block-by-block):
    b = forward_map[i] / rows_per_bucket
    Write (all column values of row i) to scatter_segment[b]
    Write forward_map[i] to scatter_segment[b]   (as an extra column)
```

After scatter, each scatter segment contains all the values that belong to
its bucket, but they're in **arrival order** (the order they appeared in
the input), not their final sorted order within the bucket.

```
Scatter segment for bucket b contains:
  - All rows whose forward_map value falls in [b*rpb, (b+1)*rpb)
  - The forward_map values themselves (as the last column)
  - Rows are in input order, NOT output order
```

**Why is scatter correct?** Every input row goes to exactly one bucket
(because forward_map is a permutation of `[0,N)`, each value falls in
exactly one bucket range). No row is lost, no row is duplicated.

### Phase 2: Permute (per bucket, in parallel)

For each bucket independently (all buckets processed in parallel):

```
For bucket b:
    1. Read the forward_map column from scatter_segment[b].
    2. Build local permutation:
         local_target[i] = forward_map_value[i] - (b * rows_per_bucket)
       This converts global output positions to positions within the bucket.
    3. For each value column (in groups that fit in memory):
         Read the column values from scatter_segment[b]
         Apply the local permutation:
           permuted[local_target[i]] = values[i]
         Write permuted values to output segment[b]
```

**Why is permute correct?** The scatter phase guarantees that every row in
scatter_segment[b] has a forward_map value in `[b*rpb, (b+1)*rpb)`.
Subtracting `b*rpb` gives a local index in `[0, bucket_size)`. Since
forward_map is a permutation, no two rows in the same bucket map to the
same local target. Therefore the local permutation is a valid permutation
within the bucket.

**Why is the assembled output correct?** Bucket 0's output segment
contains output rows 0..rpb in order. Bucket 1's contains rows rpb..2*rpb.
Concatenating all bucket segments in order gives the complete sorted output.

### Visual Example

```
Input (4 rows):
  row 0: value=A    forward_map[0] = 3
  row 1: value=B    forward_map[1] = 0
  row 2: value=C    forward_map[2] = 2
  row 3: value=D    forward_map[3] = 1

2 buckets, rows_per_bucket = 2:
  Bucket 0: output rows [0, 2)
  Bucket 1: output rows [2, 4)

Scatter phase:
  Row 0: fmap=3, bucket=3/2=1 → scatter_seg[1] gets (A, fmap=3)
  Row 1: fmap=0, bucket=0/2=0 → scatter_seg[0] gets (B, fmap=0)
  Row 2: fmap=2, bucket=2/2=1 → scatter_seg[1] gets (C, fmap=2)
  Row 3: fmap=1, bucket=1/2=0 → scatter_seg[0] gets (D, fmap=1)

  scatter_seg[0]: [(B, fmap=0), (D, fmap=1)]   ← arrival order
  scatter_seg[1]: [(A, fmap=3), (C, fmap=2)]   ← arrival order

Permute phase:
  Bucket 0 (bucket_start=0):
    local_target[0] = 0 - 0 = 0  → permuted[0] = B
    local_target[1] = 1 - 0 = 1  → permuted[1] = D
    output_seg[0] = [B, D]

  Bucket 1 (bucket_start=2):
    local_target[0] = 3 - 2 = 1  → permuted[1] = A
    local_target[1] = 2 - 2 = 0  → permuted[0] = C
    output_seg[1] = [C, A]

Final output (concatenate segments): [B, D, C, A]

Verify: output[forward_map[i]] should equal input[i]:
  output[3] = A = input[0] ✓
  output[0] = B = input[1] ✓
  output[2] = C = input[2] ✓
  output[1] = D = input[3] ✓
```


## Permutation Inversion (`invert_permutation`)

Step 2 of EC Sort computes `forward_map` from `inverse_map`. This is itself
a permutation problem: we need to compute `forward_map[inverse_map[i]] = i`
for all `i`.

This is equivalent to `permute_sframe([0, 1, 2, ..., N-1], inverse_map)`.
The input "data" is just the sequence of row indices, which can be
synthesized during scatter (no actual input data to read).

The invert module uses the same scatter-permute two-phase algorithm:

```
Scatter phase:
  For each row i in [0, N):
    Read inverse_map[i]
    bucket = inverse_map[i] / rows_per_bucket
    Write to scatter_seg[bucket]:
      Column 0 (value): i            ← synthesized, not read from input
      Column 1 (fmap):  inverse_map[i]

Permute phase:
  For each bucket b:
    Read values (column 0) and fmap (column 1)
    local_target[j] = fmap[j] - bucket_start
    permuted[local_target[j]] = values[j]
    Write permuted to output
```

The result is the forward_map stored as an SArray on CacheFs.


## Memory Management

### Budget Structure

```
Global budget = sort_max_memory (from config)
Per-thread budget = global / num_rayon_threads
```

### Bucket Count Calculation

The number of buckets is chosen so the largest column in one bucket fits in
half the per-thread budget:

```
max_col_bytes = max(bytes_per_value[col] * num_rows) over all columns
buckets_per_cpu = ceil(max_col_bytes / (per_thread_budget / 2))
num_buckets = buckets_per_cpu * num_cpus
num_buckets = min(num_buckets, num_rows)
```

The factor of 2 is conservative: it accounts for both the read buffer and
the permutation array during the permute phase.

### Scatter Phase Memory

During scatter, memory is bounded per thread to:
- One decoded block from the input (~64KB of values)
- Per-bucket flush buffers (one per column, ~64KB each, flushed at threshold)
- One forward_map chunk (shared across work items in the batch)

The forward_map is read in chunks sized by `budget / (2 * 8 * concurrent_chunks)`,
where `8` is bytes per integer.

### Permute Phase Memory

During permute, each bucket is processed independently. Per bucket:
- The forward_map column (one integer per row in the bucket)
- The local permutation array (one usize per row)
- A group of value columns (sized to fit in the per-thread budget)
- The permuted output for those columns

Columns are processed in groups. If a single column exceeds the budget,
it is still processed (at least one column per group), but no others are
batched with it.


## Parallelism

### Scatter Phase (Direct Path)

Work items are `(fmap_chunk, column, input_segment)` triples. Multiple
fmap chunks are batched so that fmap data can be shared across work items
for different columns and segments. All work items in a batch are processed
by rayon in parallel.

Each thread opens its own `SegmentReader` (no sharing). The output
`SegmentWriter`s are `Mutex`-wrapped, but encoding and compression happen
*outside* the lock — only the file write is serialized.

### Permute Phase

All M buckets are processed in parallel via `rayon::into_par_iter()`.
Each bucket is fully independent: it reads from its own scatter segment
and writes to its own output segment. No synchronization needed.


## Direct vs Lazy Scatter Paths

The scatter phase has two implementations:

**Direct path** (`scatter_columns_direct`): Used when both the input SFrame
and forward_map are backed by physical segment files (on disk or CacheFs).
Reads column blocks directly via `SegmentReader`, classifying values into
buckets as they're decoded. This is the fast path — no plan compilation,
no intermediate materialization.

**Lazy path** (`scatter_columns_lazy`): Used when the input involves lazy
plans (transforms, filters, etc.). Uses `try_slice()` to project a row
range through the plan, then `materialize_sync()` to get the values.
Processes columns in parallel within each chunk via rayon.


## Code Map

| File | Contents |
|------|----------|
| `mod.rs` | Public API (`ec_sort`, `permute_sframe`), helpers, tests |
| `scatter.rs` | Scatter phase (direct + lazy paths) |
| `permute.rs` | Permute phase, output SFrame assembly |
| `invert.rs` | Permutation inversion via scatter-permute |


## Correctness Summary

1. **Step 1 (sort keys)** is correct by delegation to `standard_sort`,
   which is a well-tested in-memory or external merge sort.

2. **Step 2 (invert permutation)** produces `forward_map` such that
   `forward_map[inverse_map[i]] = i`. This is verified by the
   scatter-permute algorithm (see below) applied to the identity sequence.

3. **Scatter** routes every input row to exactly one bucket (since
   `forward_map` is a permutation). No row is lost or duplicated. The
   forward_map value is stored alongside each row's data, preserving
   the information needed for the permute phase.

4. **Permute** reorders each bucket's rows into their final positions.
   Within a bucket, the forward_map values (minus bucket_start) form a
   valid local permutation (since they are distinct values in
   `[0, bucket_size)`). Applying this permutation places every value at
   its correct output position.

5. **Assembly** concatenates bucket segments in order. Since bucket k
   covers output rows `[k*rpb, (k+1)*rpb)` and each bucket's rows are
   now in sorted order within that range, the concatenation is the
   complete sorted output.

6. **Step 4 (reassemble)** interleaves key and value columns by original
   column index. Key columns come from the sorted key SFrame, value
   columns from the permuted value SFrame. Both have been arranged into
   the same output row order, so column alignment is preserved.
