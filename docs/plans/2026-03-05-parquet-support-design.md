# Parquet Read/Write Support Design

**Date**: 2026-03-05
**Motivation**: Interoperability with Arrow/Spark/DuckDB/Pandas ecosystem

## Scope

- Read and write Parquet files using the `arrow-rs` (`arrow` + `parquet`) crates
- Read supports single file, glob pattern, or explicit file list (concatenated row-wise)
- Write supports single file or parallel sharded output
- No zonemap filtering, predicate pushdown, or Hive-style partitioned datasets

## Crate Structure

New crate: `crates/sframe-parquet/`

```
sframe-parquet/
  Cargo.toml
  src/
    lib.rs
    parquet_reader.rs
    parquet_writer.rs
    type_mapping.rs
```

Dependency graph:
```
sframe-parquet  depends on  arrow, parquet, sframe-types, sframe-query
sframe          depends on  sframe-parquet (always linked)
sframe-py       depends on  sframe-parquet (transitively via sframe)
```

## Type Mapping

### Write (SFrame -> Parquet)

| SFrame ColumnData       | Arrow Type                  | Parquet Type              |
|-------------------------|-----------------------------|---------------------------|
| Integer (Option<i64>)   | Int64Array                  | INT64                     |
| Float (Option<f64>)     | Float64Array                | DOUBLE                    |
| String (Option<Arc<str>>)| StringArray (Utf8)          | BYTE_ARRAY + UTF8         |
| DateTime                | TimestampMicrosecondArray   | INT64 + TIMESTAMP_MICROS  |
| Vector (Option<Arc<[f64]>>)| ListArray<Float64>       | repeated DOUBLE           |
| List                    | StringArray (JSON-encoded)  | BYTE_ARRAY + UTF8         |
| Dict                    | StringArray (JSON-encoded)  | BYTE_ARRAY + UTF8         |
| Flexible                | StringArray (JSON-encoded)  | BYTE_ARRAY + UTF8         |
| Null                    | Arrow null bitmap           | Parquet definition levels |

List, Dict, and Flexible columns are serialized as JSON strings. No metadata is written
to mark the original type — on read they come back as String.

### Read (Parquet -> SFrame)

| Parquet/Arrow Type              | SFrame Type                       |
|---------------------------------|-----------------------------------|
| INT8/16/32/64, UINT8/16/32     | Integer (i64)                     |
| UINT64                          | Integer (i64), error if > i64::MAX|
| FLOAT/DOUBLE                    | Float (f64)                       |
| BYTE_ARRAY + UTF8 / Utf8 / LargeUtf8 | String                      |
| BOOLEAN                         | Integer (0/1)                     |
| TIMESTAMP (any unit/timezone)   | DateTime                          |
| DATE32/DATE64                   | DateTime                          |
| LIST of DOUBLE                  | Vector                            |
| LIST of other                   | List (recursive conversion)       |
| STRUCT / MAP                    | Dict                              |
| Other nested types              | Error with descriptive message    |

## Read Path

### Rust API

```rust
// sframe-parquet public API
pub fn read_parquet_schema(path: &str) -> Result<(Vec<String>, Vec<FlexTypeEnum>)>;
pub fn read_parquet_batches(paths: &[PathBuf]) -> Result<BatchIterator>;

// sframe public API
SFrame::from_parquet(path: &str) -> Result<SFrame>
SFrame::from_parquet_files(paths: &[&str]) -> Result<SFrame>
```

### Flow

1. **Resolve files eagerly**: At `SFrame::from_parquet()` time, if `path` contains glob
   characters (`*`, `?`, `[`), expand to a `Vec<PathBuf>`. Otherwise treat as a single
   file. For `from_parquet_files`, convert the list directly. The resolved file list is
   stored in the plan node.
2. **Read schema from first file**: Open first Parquet file, read Arrow schema, map to
   SFrame types via `type_mapping`. All files are assumed to share the same schema.
3. **Build a lazy plan**: Create a `PlannerNode` source that, when executed, returns a
   `BatchIterator` concatenating row groups across all files.
4. **BatchIterator execution**: For each file, for each row group, use `ArrowReaderBuilder`
   to read a `RecordBatch`, convert to `SFrameRows`, yield the batch. Files and row groups
   are read sequentially to keep memory bounded.
5. **Schema mismatch**: If a subsequent file has a different schema than the first, return
   an error identifying the mismatched file.

## Write Path

### Rust API

```rust
// sframe public API
SFrame::to_parquet(path: &str) -> Result<()>
SFrame::to_parquet_sharded(prefix: &str) -> Result<()>
```

### Single File

Consume the SFrame's batch iterator sequentially. Each `SFrameRows` batch is converted to
an Arrow `RecordBatch` and written as a row group via `ArrowWriter`. One output file.

### Sharded

Uses the existing parallel execution pipeline. The total number of shards N is determined
by the number of worker threads. Each thread `n` (0-indexed) gets its own `ArrowWriter`
writing to `{prefix}_{n}_of_{N}.parquet`. No coordination between threads — each writes
whatever batches it receives.

### Writer Settings

- Parquet v2 data pages
- Snappy compression
- Automatic row group sizing
- No custom metadata

## Python Bindings

```python
# Read — str (single file or glob) or list[str] (explicit file list)
sf = SFrame.from_parquet("data.parquet")
sf = SFrame.from_parquet("data/*.parquet")
sf = SFrame.from_parquet(["file1.parquet", "file2.parquet"])

# Write — single file
sf.to_parquet("output.parquet")

# Write — sharded (prefix, not directory)
sf.to_parquet("output/data", sharded=True)
# produces: output/data_0_of_4.parquet, output/data_1_of_4.parquet, ...
```

`from_parquet` accepts `str | list[str]`. The Python binding is thin: argument parsing,
GIL release, delegate to Rust.

`to_parquet` takes a path and optional `sharded: bool` (default `False`).

## Error Handling

- File not found or glob matches nothing: error
- Schema mismatch across files on read: error on the file that differs
- Unsupported Parquet type on read: error with column name and type description
- UINT64 overflow (value > i64::MAX): error
- I/O errors: propagated from arrow-rs / filesystem

No silent data loss or silent type coercion.

## Out of Scope (Future Work)

- Zonemap / row group filtering and predicate pushdown
- Hive-style partitioned datasets
- Configurable compression (Zstd, LZ4, etc.)
- Round-trip metadata for List/Dict/Flexible columns
- Parquet schema evolution / column projection on read
