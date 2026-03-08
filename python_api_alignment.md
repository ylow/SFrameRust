# Python API Alignment: SFrameRust vs C++ SFrame

This document compares the Python API exposed by the Rust implementation
(`SFrameRust/crates/sframe-py`) against the C++ reference implementation
(`SFrame/oss_src/unity/python/sframe/`). The goal is to identify gaps and
prioritize alignment work. 100% compatibility is not required — some C++ APIs
are obsolete, infeasible, or out-of-scope.

---

## 1. Naming / Signature Differences (Easy Fixes)

These are cases where both implementations provide equivalent functionality but
the names or signatures differ. Aligning these is low-effort, high-value.

| C++ API | Rust API | Recommendation |
|---------|----------|----------------|
| `SFrame.read_csv(url, ...)` | `SFrame.from_csv(path, ...)` | Add `read_csv` as alias for `from_csv` |
| `SFrame.read_json(url, orient)` | `SFrame.from_json(path)` | Add `read_json` as alias for `from_json` |
| `SFrame.export_csv(filename, ...)` | `SFrame.to_csv(path, ...)` | Add `export_csv` as alias for `to_csv` |
| `SFrame.export_json(filename, orient)` | `SFrame.to_json(path)` | Add `export_json` as alias for `to_json` |
| `SFrame.select_columns(keylist)` | `SFrame.select(names)` | Add `select_columns` as alias for `select` |
| `SFrame.select_column(key)` | `SFrame.column(name)` | Add `select_column` as alias for `column` |
| `SFrame.num_cols()` | `SFrame.num_columns()` | Add `num_cols` as alias |
| `SArray.num_missing()` | `SArray.num_missing()` (**same**) | Already aligned. Rust also has `countna()` as bonus alias. |
| `aggregate.AVG(col)` | *(missing)* | Add `AVG` as alias for `MEAN` |
| `aggregate.STDV(col)` | *(missing)* | Add `STDV` as alias for `STD` |

---

## 2. SFrame Constructor Differences

### C++ constructor: `SFrame(data)`
The C++ `SFrame.__init__` accepts many input types directly:
- `dict` of lists/SArrays
- `list` of dicts (row-oriented)
- `pandas.DataFrame`
- CSV/JSON file path (auto-detected by extension)
- Another `SArray` (single-column SFrame)

### Rust constructor
The Rust SFrame has no `__init__` constructor — users must use explicit static
methods (`from_csv`, `from_json`, `from_parquet`, `from_columns`, `read`).

### Recommendation
Add an `SFrame(data)` constructor that accepts:
- **`dict`** → delegates to `from_columns` (wrapping lists into SArrays)
- **`list` of dicts** → construct column-wise from row-oriented data
- **`pandas.DataFrame`** → convert via column iteration
- **file path string** → auto-detect format by extension

This is the single highest-impact alignment change because it's how most users
create SFrames interactively.

---

## 3. Missing SFrame Methods

### High Priority (commonly used)

| C++ Method | Description | Difficulty |
|------------|-------------|------------|
| `shape` (property) | Returns `(num_rows, num_columns)` tuple | Trivial |
| `dtype` (property) | Returns `dict` of `{name: type}` | Trivial |
| `__delitem__(key)` | `del sf['col']` syntax for column removal | Trivial |
| `add_columns(data, namelist)` | Add multiple columns at once | Easy |
| `remove_columns(column_names)` | Remove multiple columns at once | Easy |
| `filter_by(values, column_name, exclude)` | Filter rows by membership in a set of values | Medium |
| `add_row_number(column_name, start)` | Add sequential row ID column | Medium |
| `apply(fn, dtype)` | Row-wise UDF (receives dict per row) | Medium |
| `to_dataframe()` | Convert to pandas DataFrame | Medium |

### Medium Priority (useful but less common)

| C++ Method | Description | Difficulty |
|------------|-------------|------------|
| `flat_map(column_names, fn, column_types)` | One-to-many row transformation | Medium |
| `unstack(column, new_column_name)` | Pivot rows into columns | Hard |
| `dropna_split(columns, how)` | Split into (clean, dirty) SFrames | Easy |
| `copy()` | Explicit copy | Trivial |
| `print_rows(num_rows, num_columns, max_column_width)` | Formatted table printing | Easy |
| `split_datetime(expand_column, column_name_prefix)` | Split datetime column into components | Medium |
| `read_csv_with_errors(...)` | CSV read returning `(SFrame, SFrame)` with bad rows | Medium |

### Low Priority / Out of Scope

| C++ Method | Description | Why low priority |
|------------|-------------|-----------------|
| `from_odbc`, `to_odbc` | ODBC database connector | Niche, better served by external tools |
| `from_sql`, `to_sql` | SQL database connector | Same — users can use sqlalchemy + pandas bridge |
| `from_rdd`, `to_rdd`, `to_spark_dataframe` | Spark integration | Obsolete (Spark ecosystem has moved on) |
| `show(columns, view, x, y)` | Interactive visualization | Out of scope for core library |
| `to_numpy()` | NumPy array conversion | Low priority; `to_dataframe().to_numpy()` works |
| `_repr_html_()` | Notebook HTML rendering | Nice-to-have, not critical |

---

## 4. Missing SArray Methods

### High Priority

| C++ Method | Description | Difficulty |
|------------|-------------|------------|
| `shape` (property) | Returns `(len,)` tuple | Trivial |
| `size()` | Alias for `len()` | Trivial |
| `__contains__(item)` | `x in sa` membership test | Easy |
| `is_in(other)` | Element-wise membership test (like pandas `isin`) | Medium |
| `argmin()` | Index of minimum value | Medium |
| `argmax()` | Index of maximum value | Medium |
| `__neg__`, `__pos__`, `__abs__` | Unary numeric operators | Easy |
| `__floordiv__` | Floor division (`//`) | Easy |
| `__pow__` | Power (`**`) | Easy |
| `where(cls, condition, istrue, isfalse)` | Ternary conditional (like `np.where`) | Medium |
| `to_numpy()` | Convert to numpy array | Medium |

### Medium Priority

| C++ Method | Description | Difficulty |
|------------|-------------|------------|
| `from_const(cls, value, size, dtype)` | Create constant-valued array | Easy |
| `from_sequence(cls, *args)` | Create range array (Rust supports `range()` in constructor) | Easy |
| `clip_lower(threshold)` | Clip lower bound only | Trivial (sugar over `clip`) |
| `clip_upper(threshold)` | Clip upper bound only | Trivial (sugar over `clip`) |
| `subslice(start, stop, step)` | Slice strings/lists element-wise | Medium |
| `hash(seed)` | Hash each element | Medium |
| `random_split(fraction, seed)` | Split array into two | Easy |
| `cumulative_sum()` | Cumulative sum | Medium |
| `cumulative_mean()` | Cumulative mean | Medium |
| `cumulative_min()` | Cumulative min | Medium |
| `cumulative_max()` | Cumulative max | Medium |
| `cumulative_std()` | Cumulative std dev | Medium |
| `cumulative_var()` | Cumulative variance | Medium |
| `rolling_var(...)` | Rolling variance | Medium |
| `rolling_stdv(...)` | Rolling std dev | Medium |
| `rolling_count(...)` | Rolling count | Medium |
| `save(filename, format)` | Save SArray to file | Easy |

### Low Priority / Out of Scope

| C++ Method | Description | Why low priority |
|------------|-------------|-----------------|
| `from_avro(filename)` | Load from Avro file | Niche format |
| `date_range(start, end, freq)` | Create datetime range | Sugar; can use Python datetime + list |
| `random_integers(size, seed)` | Generate random int array | Sugar; can use Python random + list |
| `pixel_array_to_image(...)` | Image type support | Out of scope (no Image type) |
| `split_datetime(...)` | Split datetime into components | Medium priority if datetime is supported |
| `unpack(...)` | Unpack list/dict into columns | Overlaps with SFrame.unpack_column |
| `topk_index(topk, reverse)` | Indices of top k values | Niche |
| `sketch_summary(...)` | Statistical sketch object | Rust uses direct methods instead |
| `show()` | Visualization | Out of scope |

---

## 5. Missing Aggregation Functions

| C++ Aggregate | Description | Difficulty |
|---------------|-------------|------------|
| `ARGMAX(agg_col, out_col)` | Value of `out_col` at row where `agg_col` is max | Medium |
| `ARGMIN(agg_col, out_col)` | Value of `out_col` at row where `agg_col` is min | Medium |
| `QUANTILE(col, *quantiles)` | Compute quantile(s) | Medium |
| `DISTINCT(col)` | Collect distinct values into list | Easy (alias for unique concat) |
| `FREQ_COUNT(col)` | Frequency count as dict | Medium |

---

## 6. Missing Classes / Modules

| C++ Class | Description | Recommendation |
|-----------|-------------|----------------|
| `SGraph`, `Vertex`, `Edge` | Graph data structure | Out of scope — graph analytics is a separate concern |
| `Sketch` | Approximate statistics object | Out of scope — Rust exposes equivalent methods directly on SArray (`approx_count_distinct`, `frequent_items`) |
| `Image` | Image data type | Out of scope for now |
| `SFrameBuilder` | Row-by-row SFrame construction | Rust has `SFrameStreamWriter` which serves the same purpose |
| `SArrayBuilder` | Row-by-row SArray construction | Low priority — users can build lists then wrap in SArray |
| `GroupedSFrame` | Lazy grouped view with `get_group()`/iteration | Medium priority — useful for exploratory analysis |

---

## 7. Features Rust Has That C++ Doesn't

These are Rust-only features that should be kept:

| Rust Feature | Description |
|--------------|-------------|
| `SFrame.from_parquet()` / `to_parquet()` | Parquet I/O support |
| `SFrame.explain()` / `SArray.explain()` | Query plan inspection |
| `SFrame.replace_column(name, col)` | In-place column replacement |
| `SFrameStreamWriter` | Streaming batch writer with context manager |
| `config` object | Runtime tuning (cache, batch size, sort memory, etc.) |
| `SArray.approx_count_distinct()` | Direct approximate cardinality |
| `SArray.frequent_items(k)` | Direct top-k frequent items |
| `SArray.count_bag_of_words()` | Bag-of-words text feature |
| `SArray.count_ngrams()` / `count_character_ngrams()` | N-gram text features |
| `SArray(range(...))` constructor | Lazy range-based construction |

---

## 8. Prioritized Action Plan

### Phase 1: Trivial Alignment (< 1 day)
1. Add `shape` property to both SFrame and SArray
2. Add `dtype` property to SFrame (dict form)
3. Add `size()` to SArray
4. Add `num_cols()` alias to SFrame
5. Add `read_csv` / `read_json` / `export_csv` / `export_json` aliases to SFrame
6. Add `select_columns` / `select_column` aliases to SFrame
7. Add `__delitem__` to SFrame
8. Add `copy()` to SFrame
9. Add `AVG` and `STDV` aliases to aggregate module
10. Add `clip_lower` / `clip_upper` to SArray
11. Add `__neg__` / `__pos__` / `__abs__` to SArray
12. Add `__floordiv__` / `__pow__` to SArray

### Phase 2: Easy Additions (1-3 days)
1. `SFrame(data)` constructor (dict, list-of-dicts, pandas DataFrame)
2. `add_columns` / `remove_columns` on SFrame
3. `filter_by(values, column_name, exclude)` on SFrame
4. `add_row_number(column_name, start)` on SFrame
5. `SArray.__contains__(item)` membership test
6. `SArray.is_in(other)` element-wise membership
7. `SArray.from_const(value, size, dtype)`
8. `SArray.random_split(fraction, seed)`
9. `dropna_split` on SFrame
10. `print_rows` on SFrame

### Phase 3: Medium Effort (1-2 weeks)
1. `SArray.argmin()` / `argmax()`
2. `SArray.where(condition, istrue, isfalse)` — ternary conditional
3. `SArray.cumulative_sum/mean/min/max` — cumulative operations
4. `SArray.rolling_var/rolling_stdv/rolling_count` — additional rolling ops
5. `SArray.subslice(start, stop, step)` — element-wise string/list slicing
6. `SArray.hash(seed)` — element hashing
7. `SFrame.apply(fn, dtype)` — row-wise UDF
8. `SFrame.flat_map(column_names, fn, column_types)` — one-to-many UDF
9. `SFrame.to_dataframe()` — pandas conversion
10. `aggregate.ARGMAX / ARGMIN / QUANTILE / DISTINCT / FREQ_COUNT`
11. `GroupedSFrame` — lazy grouped view with iteration

### Phase 4: Lower Priority
1. `SFrame.unstack()`
2. `SFrame.split_datetime()`
3. `SArray.to_numpy()`
4. `SArray.save()`
5. `read_csv_with_errors`

### Not Planned
- `SGraph` / `Vertex` / `Edge` — graph analytics out of scope
- `Sketch` class — Rust has equivalent direct methods
- `Image` type — out of scope
- ODBC / SQL connectors — use external tools
- Spark integration — obsolete
- `show()` visualization — out of scope for core library
