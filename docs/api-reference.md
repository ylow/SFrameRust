# SFrame & SArray API Reference

## Overview

SFrame is a lazy, out-of-core columnar dataframe. SArray is a lazy columnar
array. Operations build a planner DAG; execution is deferred until
materialization (`.head()`, `.num_rows()`, `.save()`, `.materialize()`, or
`Display`).

Data types (`FlexTypeEnum`): `Integer`, `Float`, `String`, `Vector`, `List`,
`Dict`, `DateTime`, `Undefined`.

---

## SFrame

### Construction

| Method | Description |
|--------|-------------|
| `SFrame::read(path)` | Open an SFrame directory from disk (metadata-only, lazy). |
| `SFrame::from_csv(path, options)` | Parse a CSV file. Options: `CsvOptions { delimiter, has_header, column_type_hints, na_values, true_values, false_values, skip_rows, row_limit }`. |
| `SFrame::from_json(path)` | Parse a JSON Lines file (one JSON object per line). |
| `SFrame::from_columns(cols)` | Build from `Vec<(&str, SArray)>` column pairs. |

### Schema Inspection

| Method | Returns | Description |
|--------|---------|-------------|
| `num_rows()` | `Result<u64>` | Row count (may trigger materialization). |
| `num_columns()` | `usize` | Number of columns. |
| `column_names()` | `&[String]` | Column name list. |
| `column_types()` | `Vec<FlexTypeEnum>` | Column type list. |
| `schema()` | `Vec<(String, FlexTypeEnum)>` | Name-type pairs. |
| `explain()` | `String` | Pretty-printed query plan DAG. |

### Column Access & Mutation

| Method | Description |
|--------|-------------|
| `column(name)` | Get a reference to a column as `&SArray`. |
| `select(names)` | Project to a subset of columns. |
| `add_column(name, col)` | Add a new column (returns new SFrame). |
| `remove_column(name)` | Drop a column (returns new SFrame). |
| `replace_column(name, col)` | Replace a column with a new SArray. |
| `rename(mapping)` | Rename columns via `HashMap<&str, &str>`. |
| `swap_columns(name1, name2)` | Swap two columns' positions. |

### Row Operations

| Method | Description |
|--------|-------------|
| `head(n)` | First `n` rows (lazy, streaming). |
| `tail(n)` | Last `n` rows (materializes). |
| `filter(column_name, pred)` | Keep rows where `pred(&FlexType) -> bool` on the named column. Lazy. |
| `sample(fraction, seed)` | Random sample of rows (deterministic with seed). |
| `random_split(fraction, seed)` | Split into `(train, test)` SFrames. |
| `unique()` | Remove duplicate rows. |
| `topk(column_name, k, reverse)` | Top-k rows by a column (sort + head). |
| `append(other)` | Vertical concatenation (lazy). |
| `iter_rows()` | Materialize all rows as `Vec<Vec<FlexType>>`. |

### Sorting

```rust
sf.sort(&[("amount", SortOrder::Descending), ("name", SortOrder::Ascending)])
```

Supports multi-key sort. Automatically uses **external memory sort** when
estimated data size exceeds the sort memory budget (default 256 MB, configurable
via `SFRAME_SORT_BUFFER_SIZE`). External sort uses quantile-based range
partitioning with CacheFs-backed spill-to-disk.

### Groupby & Aggregation

```rust
sf.groupby(
    &["category"],
    vec![
        AggSpec::count(col_idx, "count"),
        AggSpec::sum(col_idx, "total"),
        AggSpec::mean(col_idx, "avg"),
    ],
)
```

Available aggregators: `count`, `sum`, `mean`, `min`, `max`, `variance`,
`std_dev`, `count_distinct`, `concat_list`, `select_one`.

Uses O(groups) memory with spill-to-disk when hash table exceeds
`groupby_buffer_num_rows` (default 1M rows).

### Joins

```rust
// Single-key join
sf.join(&other, "left_key", "right_key", JoinType::Inner)

// Multi-key join
sf.join_on(&other, &[("l1", "r1"), ("l2", "r2")], JoinType::Left)
```

Join types: `Inner`, `Left`, `Right`, `Full`.

Uses hash join with GRACE partitioning for large inputs (spills when hash
table exceeds `join_buffer_num_cells`, default 50M cells).

### Missing Values

| Method | Description |
|--------|-------------|
| `dropna(column_name, how)` | Drop rows with `Undefined`. `how`: `"any"` (default) or `"all"`. Optional column filter. |
| `fillna(column_name, value)` | Replace `Undefined` in a column with a fill value. |

### Reshaping

| Method | Description |
|--------|-------------|
| `pack_columns(columns, new_name)` | Pack multiple columns into a single Dict column. |
| `unpack_column(column, prefix)` | Unpack a Dict/List column into separate columns. |
| `stack(column, new_name)` | Unnest a List/Vector/Dict column (one row per element). |

### Metadata

| Method | Description |
|--------|-------------|
| `get_metadata(key)` | Read a metadata value. |
| `set_metadata(key, value)` | Set a metadata key-value pair. |
| `remove_metadata(key)` | Remove a metadata key. |
| `metadata()` | All metadata as `&HashMap<String, String>`. |

### Serialization

| Method | Description |
|--------|-------------|
| `save(path)` | Write to disk as an SFrame directory. Streaming — does not materialize in memory. |
| `to_csv(path, options)` | Write to CSV. Options: `CsvWriterOptions { delimiter, header, quote_style }`. Streaming. |
| `to_json(path)` | Write to JSON Lines file. |
| `materialize()` | Force evaluation of all lazy operations. |

### Streaming Writer

For writing large datasets without holding everything in memory:

```rust
let mut writer = SFrameStreamWriter::new("output.sf", &["id", "name"], &[Integer, String])?;
writer.write_batch(&batch1)?;
writer.write_batch(&batch2)?;
writer.set_metadata("source", "pipeline_v2");
writer.finish()?;
```

Also available: `SFrameStreamWriter::with_vfs(vfs, path, names, types)` for
custom filesystem backends.

---

## SArray

### Construction

| Method | Description |
|--------|-------------|
| `SArray::from_vec(values, dtype)` | Create from a `Vec<FlexType>`. |

### Inspection

| Method | Returns | Description |
|--------|---------|-------------|
| `dtype()` | `FlexTypeEnum` | Element type. |
| `len()` | `Result<u64>` | Number of elements (may trigger materialization). |
| `head(n)` | `Result<Vec<FlexType>>` | First `n` values. |
| `tail(n)` | `Result<Vec<FlexType>>` | Last `n` values. |
| `to_vec()` | `Result<Vec<FlexType>>` | Materialize all values. |

### Transforms

| Method | Description |
|--------|-------------|
| `apply(func, output_type)` | Element-wise transform `Fn(&FlexType) -> FlexType`. Lazy. |
| `filter(pred)` | Keep elements where `pred(&FlexType) -> bool`. Lazy. |
| `astype(target, undefined_on_failure)` | Cast elements to a different type. |
| `clip(lower, upper)` | Clamp values to `[lower, upper]`. |
| `sort(ascending)` | Sort values. |
| `unique()` | Deduplicate. Uses in-memory HashSet for small arrays, sort-based dedup for large. |
| `append(other)` | Concatenate with another SArray (lazy). |
| `sample(fraction, seed)` | Random sample. |

### Arithmetic (Element-wise)

| Method | Description |
|--------|-------------|
| `add(other)` | `a + b` |
| `sub(other)` | `a - b` |
| `mul(other)` | `a * b` |
| `div(other)` | `a / b` (always Float) |
| `rem(other)` | `a % b` (integers only) |

When both SArrays share the same underlying plan, these are **lazy and
streaming** (GeneralizedTransform). Otherwise, both sides are materialized.

### Scalar Arithmetic

| Method | Description |
|--------|-------------|
| `add_scalar(scalar)` | Add scalar to each element. |
| `sub_scalar(scalar)` | Subtract scalar from each element. |
| `mul_scalar(scalar)` | Multiply each element by scalar. |
| `div_scalar(scalar)` | Divide each element by scalar (always Float). |
| `rem_scalar(scalar)` | Remainder by scalar (integers only). |

### Comparisons

All return `SArray` of `Integer` (0 or 1):

| Method | Description |
|--------|-------------|
| `eq(other)` / `eq_scalar(scalar)` | `a == b` |
| `ne(other)` / `ne_scalar(scalar)` | `a != b` |
| `lt(other)` / `lt_scalar(scalar)` | `a < b` |
| `le(other)` / `le_scalar(scalar)` | `a <= b` |
| `gt(other)` / `gt_scalar(scalar)` | `a > b` |
| `ge(other)` / `ge_scalar(scalar)` | `a >= b` |

### Logical

| Method | Description |
|--------|-------------|
| `and(other)` | Logical AND (non-zero = true). |
| `or(other)` | Logical OR (non-zero = true). |

### Reductions

All reductions use streaming plan-based execution (never materialize the full
column). Parallel reduce is used automatically when the source is an SFrame on
disk.

| Method | Returns | Description |
|--------|---------|-------------|
| `sum()` | `FlexType` | Sum of all values. |
| `min_val()` | `FlexType` | Minimum value. |
| `max_val()` | `FlexType` | Maximum value. |
| `mean()` | `FlexType` | Arithmetic mean. |
| `std_dev(ddof)` | `FlexType` | Standard deviation. `ddof=0` for population, `ddof=1` for sample. |
| `variance(ddof)` | `FlexType` | Variance. |
| `any()` | `bool` | True if any element is non-zero. |
| `all()` | `bool` | True if all elements are non-zero. |
| `nnz()` | `u64` | Count of non-zero elements. |
| `countna()` / `num_missing()` | `u64` | Count of `Undefined` values. |

### Approximate Analytics

| Method | Description |
|--------|-------------|
| `approx_count_distinct()` | Approximate distinct count via HyperLogLog (precision=12, ~1.6% error, 4 KB memory). |
| `frequent_items(k)` | Top-k frequent items via Space-Saving algorithm. O(k) memory. Returns `Vec<(FlexType, u64)>`. |

### Missing Value Handling

| Method | Description |
|--------|-------------|
| `countna()` | Count `Undefined` values. |
| `dropna()` | Remove `Undefined` values. Lazy. |
| `fillna(value)` | Replace `Undefined` with a fill value. Lazy. |
| `is_na()` | Returns Integer SArray: 1 where `Undefined`, 0 otherwise. Lazy. |

### String Operations

| Method | Description |
|--------|-------------|
| `contains(substring)` | Returns Integer SArray (1 if contains, 0 otherwise). |
| `count_bag_of_words(to_lower)` | Word frequencies as Dict per element. |
| `count_ngrams(n, to_lower)` | Word n-gram frequencies as Dict per element. |
| `count_character_ngrams(n, to_lower)` | Character n-gram frequencies as Dict per element. |

### Dict Operations

| Method | Description |
|--------|-------------|
| `dict_keys()` | Extract keys from each Dict as a List. |
| `dict_values()` | Extract values from each Dict as a List. |
| `dict_trim_by_keys(keys, exclude)` | Keep or exclude entries by key set. |
| `dict_trim_by_values(lower, upper)` | Keep entries with values in `[lower, upper]`. |
| `dict_has_any_keys(keys)` | Integer SArray: 1 if dict has any of the keys. |
| `dict_has_all_keys(keys)` | Integer SArray: 1 if dict has all of the keys. |

### Structured Data

| Method | Description |
|--------|-------------|
| `item_length()` | Length of each element (string len, vector/list/dict size). |
| `vector_slice(start, end)` | Slice vector subarrays. |

### Rolling / Windowed Aggregations

Window defined as `[i - before, i + after]` (inclusive). If fewer than
`min_observations` values in the window, the result is `Undefined`.

| Method | Description |
|--------|-------------|
| `rolling_sum(before, after, min_observations)` | Rolling sum. |
| `rolling_mean(before, after, min_observations)` | Rolling mean. |
| `rolling_min(before, after, min_observations)` | Rolling minimum. |
| `rolling_max(before, after, min_observations)` | Rolling maximum. |

---

## Configuration

Environment variables (read at process startup):

| Variable | Default | Description |
|----------|---------|-------------|
| `SFRAME_CACHE_CAPACITY` | 2 GiB | Total in-memory cache for CacheFs. |
| `SFRAME_CACHE_CAPACITY_PER_FILE` | 128 MiB | Max single file size in memory cache. |
| `SFRAME_SORT_BUFFER_SIZE` | 256 MiB | Memory budget before external sort kicks in. |
| `RAYON_NUM_THREADS` | CPU count | Number of worker threads for parallel execution. |

---

## Execution Model

- **Lazy evaluation**: Most operations build a planner DAG without touching data.
- **Data-parallel execution**: Queries are automatically split across rayon
  worker threads by row range. Each worker reads its slice independently with
  block-level skipping.
- **Streaming I/O**: Sources prefetch segments ahead via a background thread.
  Sinks (`save`, `to_csv`) stream batches to disk without full materialization.
- **External memory**: Sort, groupby, and join all support spill-to-disk via
  CacheFs when data exceeds memory budgets.
- **Columnar compression**: SFrame files use frame-of-reference integer packing,
  dictionary encoding for strings, and LZ4 block compression.
