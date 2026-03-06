# Parquet Read/Write Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Parquet read/write support for interoperability with the Arrow/Spark/DuckDB/Pandas ecosystem.

**Architecture:** New `sframe-parquet` crate provides type mapping (Arrow ↔ SFrame), a Parquet reader that produces `BatchIterator`, and a Parquet writer that consumes `BatchIterator`. The `sframe` crate exposes `from_parquet`/`to_parquet` methods. Python bindings wire through to Rust.

**Tech Stack:** `arrow` and `parquet` crates from `arrow-rs`, `glob` crate for file pattern matching.

**Design doc:** `docs/plans/2026-03-05-parquet-support-design.md`

---

### Task 1: Create the `sframe-parquet` crate skeleton

**Files:**
- Create: `crates/sframe-parquet/Cargo.toml`
- Create: `crates/sframe-parquet/src/lib.rs`
- Create: `crates/sframe-parquet/src/type_mapping.rs`
- Create: `crates/sframe-parquet/src/parquet_reader.rs`
- Create: `crates/sframe-parquet/src/parquet_writer.rs`
- Modify: `Cargo.toml` (workspace root, add member)
- Modify: `crates/sframe/Cargo.toml` (add sframe-parquet dependency)

**Step 1: Create Cargo.toml for the new crate**

```toml
[package]
name = "sframe-parquet"
version = "0.1.0"
edition = "2021"

[dependencies]
sframe-types = { path = "../sframe-types" }
sframe-query = { path = "../sframe-query" }
arrow = { version = "54", default-features = false }
parquet = { version = "54", default-features = false, features = ["arrow", "snap"] }
glob = "0.3"
```

Note: Check the latest `arrow-rs` version at the time of implementation. Use matching versions for `arrow` and `parquet`.

**Step 2: Create stub source files**

`src/lib.rs`:
```rust
pub mod type_mapping;
pub mod parquet_reader;
pub mod parquet_writer;
```

`src/type_mapping.rs`:
```rust
//! Arrow ↔ SFrame type conversion.
```

`src/parquet_reader.rs`:
```rust
//! Parquet file reader producing BatchIterator.
```

`src/parquet_writer.rs`:
```rust
//! Parquet file writer consuming BatchIterator.
```

**Step 3: Add to workspace and wire dependency**

Add `"crates/sframe-parquet"` to `members` in the root `Cargo.toml`.

Add to `crates/sframe/Cargo.toml`:
```toml
sframe-parquet = { path = "../sframe-parquet" }
```

**Step 4: Verify it compiles**

Run: `cargo build -p sframe-parquet`
Expected: Compiles successfully.

**Step 5: Commit**

```
feat(parquet): add sframe-parquet crate skeleton
```

---

### Task 2: Implement Arrow → SFrame type mapping (read direction)

**Files:**
- Modify: `crates/sframe-parquet/src/type_mapping.rs`

This task implements converting Arrow `DataType` to `FlexTypeEnum`, and converting Arrow arrays to `ColumnData`.

**Step 1: Write tests for Arrow DataType → FlexTypeEnum mapping**

Add to `type_mapping.rs`:

```rust
use arrow::datatypes::DataType;
use sframe_types::flex_type::FlexTypeEnum;
use sframe_types::error::{Result, SFrameError};

/// Map an Arrow DataType to the corresponding SFrame FlexTypeEnum.
pub fn arrow_type_to_sframe(dt: &DataType) -> Result<FlexTypeEnum> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;

    #[test]
    fn test_int_types() {
        assert_eq!(arrow_type_to_sframe(&DataType::Int8).unwrap(), FlexTypeEnum::Integer);
        assert_eq!(arrow_type_to_sframe(&DataType::Int16).unwrap(), FlexTypeEnum::Integer);
        assert_eq!(arrow_type_to_sframe(&DataType::Int32).unwrap(), FlexTypeEnum::Integer);
        assert_eq!(arrow_type_to_sframe(&DataType::Int64).unwrap(), FlexTypeEnum::Integer);
        assert_eq!(arrow_type_to_sframe(&DataType::UInt8).unwrap(), FlexTypeEnum::Integer);
        assert_eq!(arrow_type_to_sframe(&DataType::UInt16).unwrap(), FlexTypeEnum::Integer);
        assert_eq!(arrow_type_to_sframe(&DataType::UInt32).unwrap(), FlexTypeEnum::Integer);
    }

    #[test]
    fn test_float_types() {
        assert_eq!(arrow_type_to_sframe(&DataType::Float32).unwrap(), FlexTypeEnum::Float);
        assert_eq!(arrow_type_to_sframe(&DataType::Float64).unwrap(), FlexTypeEnum::Float);
    }

    #[test]
    fn test_string_types() {
        assert_eq!(arrow_type_to_sframe(&DataType::Utf8).unwrap(), FlexTypeEnum::String);
        assert_eq!(arrow_type_to_sframe(&DataType::LargeUtf8).unwrap(), FlexTypeEnum::String);
    }

    #[test]
    fn test_boolean() {
        assert_eq!(arrow_type_to_sframe(&DataType::Boolean).unwrap(), FlexTypeEnum::Integer);
    }

    #[test]
    fn test_timestamp() {
        use arrow::datatypes::TimeUnit;
        assert_eq!(
            arrow_type_to_sframe(&DataType::Timestamp(TimeUnit::Microsecond, None)).unwrap(),
            FlexTypeEnum::DateTime,
        );
        assert_eq!(
            arrow_type_to_sframe(&DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()))).unwrap(),
            FlexTypeEnum::DateTime,
        );
    }

    #[test]
    fn test_date_types() {
        assert_eq!(arrow_type_to_sframe(&DataType::Date32).unwrap(), FlexTypeEnum::DateTime);
        assert_eq!(arrow_type_to_sframe(&DataType::Date64).unwrap(), FlexTypeEnum::DateTime);
    }

    #[test]
    fn test_list_of_f64() {
        let dt = DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float64, true)));
        assert_eq!(arrow_type_to_sframe(&dt).unwrap(), FlexTypeEnum::Vector);
    }

    #[test]
    fn test_list_of_other() {
        let dt = DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Utf8, true)));
        assert_eq!(arrow_type_to_sframe(&dt).unwrap(), FlexTypeEnum::List);
    }

    #[test]
    fn test_unsupported_type_errors() {
        // Binary without UTF8 should error
        assert!(arrow_type_to_sframe(&DataType::Duration(arrow::datatypes::TimeUnit::Second)).is_err());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p sframe-parquet`
Expected: All tests fail with `todo!()` panic.

**Step 3: Implement `arrow_type_to_sframe`**

```rust
pub fn arrow_type_to_sframe(dt: &DataType) -> Result<FlexTypeEnum> {
    match dt {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
        | DataType::UInt8 | DataType::UInt16 | DataType::UInt32 => Ok(FlexTypeEnum::Integer),

        DataType::UInt64 => Ok(FlexTypeEnum::Integer), // overflow checked at value level

        DataType::Float16 | DataType::Float32 | DataType::Float64 => Ok(FlexTypeEnum::Float),

        DataType::Boolean => Ok(FlexTypeEnum::Integer),

        DataType::Utf8 | DataType::LargeUtf8 => Ok(FlexTypeEnum::String),

        DataType::Timestamp(_, _) => Ok(FlexTypeEnum::DateTime),
        DataType::Date32 | DataType::Date64 => Ok(FlexTypeEnum::DateTime),

        DataType::List(field) | DataType::LargeList(field) => {
            if matches!(field.data_type(), DataType::Float64) {
                Ok(FlexTypeEnum::Vector)
            } else {
                Ok(FlexTypeEnum::List)
            }
        }

        DataType::Struct(_) | DataType::Map(_, _) => Ok(FlexTypeEnum::Dict),

        other => Err(SFrameError::Format(
            format!("Unsupported Parquet/Arrow type: {:?}", other),
        )),
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p sframe-parquet`
Expected: All tests pass.

**Step 5: Commit**

```
feat(parquet): implement Arrow → SFrame type mapping
```

---

### Task 3: Implement Arrow array → SFrame ColumnData conversion

**Files:**
- Modify: `crates/sframe-parquet/src/type_mapping.rs`

This converts an Arrow `ArrayRef` (one column of a `RecordBatch`) into a `ColumnData`.

**Step 1: Write tests**

```rust
use arrow::array::*;
use sframe_query::batch::ColumnData;
use sframe_types::flex_type::FlexType;

/// Convert an Arrow array to SFrame ColumnData.
pub fn arrow_array_to_column(array: &dyn arrow::array::Array, sframe_type: FlexTypeEnum) -> Result<ColumnData> {
    todo!()
}

#[cfg(test)]
mod tests {
    // ... existing tests ...

    #[test]
    fn test_convert_int64_array() {
        let arr = Int64Array::from(vec![Some(1), None, Some(3)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Integer).unwrap();
        match col {
            ColumnData::Integer(v) => {
                assert_eq!(v, vec![Some(1), None, Some(3)]);
            }
            _ => panic!("expected Integer column"),
        }
    }

    #[test]
    fn test_convert_int32_array_to_i64() {
        let arr = Int32Array::from(vec![Some(10), Some(20)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Integer).unwrap();
        match col {
            ColumnData::Integer(v) => {
                assert_eq!(v, vec![Some(10), Some(20)]);
            }
            _ => panic!("expected Integer column"),
        }
    }

    #[test]
    fn test_convert_float64_array() {
        let arr = Float64Array::from(vec![Some(1.5), None, Some(3.5)]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Float).unwrap();
        match col {
            ColumnData::Float(v) => {
                assert_eq!(v, vec![Some(1.5), None, Some(3.5)]);
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn test_convert_string_array() {
        let arr = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::String).unwrap();
        match col {
            ColumnData::String(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0].as_deref(), Some("hello"));
                assert!(v[1].is_none());
                assert_eq!(v[2].as_deref(), Some("world"));
            }
            _ => panic!("expected String column"),
        }
    }

    #[test]
    fn test_convert_boolean_array() {
        let arr = BooleanArray::from(vec![Some(true), Some(false), None]);
        let col = arrow_array_to_column(&arr, FlexTypeEnum::Integer).unwrap();
        match col {
            ColumnData::Integer(v) => {
                assert_eq!(v, vec![Some(1), Some(0), None]);
            }
            _ => panic!("expected Integer column"),
        }
    }

    #[test]
    fn test_convert_uint64_overflow() {
        let arr = UInt64Array::from(vec![Some(u64::MAX)]);
        assert!(arrow_array_to_column(&arr, FlexTypeEnum::Integer).is_err());
    }

    #[test]
    fn test_convert_list_f64_to_vector() {
        let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let offsets = OffsetSizeTrait... // build ListArray of [[1.0, 2.0], [3.0, 4.0]]
        // Exact construction depends on arrow API — use ListBuilder:
        let mut builder = ListBuilder::new(Float64Builder::new());
        builder.values().append_value(1.0);
        builder.values().append_value(2.0);
        builder.append(true);
        builder.values().append_value(3.0);
        builder.values().append_value(4.0);
        builder.append(true);
        let arr = builder.finish();

        let col = arrow_array_to_column(&arr, FlexTypeEnum::Vector).unwrap();
        match col {
            ColumnData::Vector(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0].as_deref(), Some([1.0, 2.0].as_slice()));
                assert_eq!(v[1].as_deref(), Some([3.0, 4.0].as_slice()));
            }
            _ => panic!("expected Vector column"),
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p sframe-parquet`
Expected: New tests fail.

**Step 3: Implement `arrow_array_to_column`**

Implement using Arrow's `as_any().downcast_ref::<T>()` pattern. Handle each SFrame type:
- Integer: downcast to Int8/16/32/64/UInt8/16/32/64/Boolean arrays, map values to `Option<i64>`. For UInt64, check each value fits in i64.
- Float: downcast to Float32/Float64 arrays, widen Float32 to f64.
- String: downcast to StringArray/LargeStringArray.
- DateTime: downcast to TimestampMicrosecondArray (and other units, converting to microseconds). For Date32, multiply by 86400 * 1_000_000. For Date64, multiply by 1000.
- Vector: downcast to ListArray, extract Float64 inner values, produce `Arc<[f64]>` per row.
- List: downcast to ListArray, recursively convert inner values to FlexType.
- Dict: downcast to StructArray/MapArray, convert fields to key-value pairs.

The exact implementation is non-trivial for timestamps and nested types, so this may need
to be broken into sub-steps during implementation. Focus on the common types first
(Integer, Float, String, Boolean) and add DateTime, Vector, List, Dict iteratively.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p sframe-parquet`
Expected: All tests pass.

**Step 5: Commit**

```
feat(parquet): implement Arrow array → SFrame ColumnData conversion
```

---

### Task 4: Implement SFrame → Arrow type mapping (write direction)

**Files:**
- Modify: `crates/sframe-parquet/src/type_mapping.rs`

This converts `FlexTypeEnum` → Arrow `DataType` and `ColumnData` → Arrow `ArrayRef`.

**Step 1: Write tests**

```rust
use arrow::datatypes::DataType;

/// Map SFrame FlexTypeEnum to Arrow DataType.
pub fn sframe_type_to_arrow(ft: FlexTypeEnum) -> DataType {
    todo!()
}

/// Convert SFrame ColumnData to an Arrow ArrayRef.
pub fn column_to_arrow_array(col: &ColumnData) -> Result<ArrayRef> {
    todo!()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sframe_to_arrow_types() {
        assert_eq!(sframe_type_to_arrow(FlexTypeEnum::Integer), DataType::Int64);
        assert_eq!(sframe_type_to_arrow(FlexTypeEnum::Float), DataType::Float64);
        assert_eq!(sframe_type_to_arrow(FlexTypeEnum::String), DataType::Utf8);
        // DateTime → Timestamp(Microsecond, None)
        // Vector → List(Float64)
        // List, Dict → Utf8 (JSON encoded)
    }

    #[test]
    fn test_integer_column_to_arrow() {
        let col = ColumnData::Integer(vec![Some(1), None, Some(3)]);
        let arr = column_to_arrow_array(&col).unwrap();
        let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), 1);
        assert!(arr.is_null(1));
        assert_eq!(arr.value(2), 3);
    }

    #[test]
    fn test_float_column_to_arrow() {
        let col = ColumnData::Float(vec![Some(1.5), None]);
        let arr = column_to_arrow_array(&col).unwrap();
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.value(0), 1.5);
        assert!(arr.is_null(1));
    }

    #[test]
    fn test_string_column_to_arrow() {
        let col = ColumnData::String(vec![Some("hello".into()), None]);
        let arr = column_to_arrow_array(&col).unwrap();
        let arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(arr.value(0), "hello");
        assert!(arr.is_null(1));
    }

    #[test]
    fn test_vector_column_to_arrow() {
        let col = ColumnData::Vector(vec![
            Some(vec![1.0, 2.0].into()),
            None,
        ]);
        let arr = column_to_arrow_array(&col).unwrap();
        // Should be ListArray<Float64>
        let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_arr.len(), 2);
        assert!(!list_arr.is_null(0));
        assert!(list_arr.is_null(1));
    }

    #[test]
    fn test_list_column_to_arrow_json() {
        let col = ColumnData::List(vec![
            Some(vec![FlexType::Integer(1), FlexType::String("two".into())].into()),
        ]);
        let arr = column_to_arrow_array(&col).unwrap();
        let arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
        // Should be JSON-encoded
        let val = arr.value(0);
        assert!(val.contains("1"));
        assert!(val.contains("two"));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p sframe-parquet`

**Step 3: Implement both functions**

`sframe_type_to_arrow`:
- Integer → Int64
- Float → Float64
- String → Utf8
- DateTime → Timestamp(Microsecond, None)
- Vector → List(Field("item", Float64, true))
- List → Utf8 (JSON)
- Dict → Utf8 (JSON)
- Undefined → Utf8 (JSON)

`column_to_arrow_array`:
- Integer: build `Int64Array` from `Vec<Option<i64>>`
- Float: build `Float64Array` from `Vec<Option<f64>>`
- String: build `StringArray` from values
- DateTime: build `TimestampMicrosecondArray` from posix_timestamp micros
- Vector: build `ListArray` with `Float64Builder` inner
- List/Dict/Flexible: JSON-serialize each value to `StringArray`

For List/Dict JSON serialization, use `serde_json`. Add `serde_json` to `sframe-parquet`
dependencies if not already pulled in transitively.

**Step 4: Run tests**

Run: `cargo test -p sframe-parquet`

**Step 5: Commit**

```
feat(parquet): implement SFrame → Arrow type mapping and array conversion
```

---

### Task 5: Implement `RecordBatch` ↔ `SFrameRows` conversion helpers

**Files:**
- Modify: `crates/sframe-parquet/src/type_mapping.rs`

Build on Tasks 3 and 4 to convert whole batches.

**Step 1: Write tests**

```rust
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, Field};

/// Convert an Arrow RecordBatch to SFrameRows.
pub fn record_batch_to_sframe_rows(batch: &RecordBatch) -> Result<SFrameRows> {
    todo!()
}

/// Convert SFrameRows to an Arrow RecordBatch given column names and types.
pub fn sframe_rows_to_record_batch(
    rows: &SFrameRows,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
) -> Result<RecordBatch> {
    todo!()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_record_batch_roundtrip() {
        // Build SFrameRows with Integer and String columns
        let int_col = ColumnData::Integer(vec![Some(1), Some(2), Some(3)]);
        let str_col = ColumnData::String(vec![
            Some("a".into()), Some("b".into()), Some("c".into()),
        ]);
        let rows = SFrameRows::new(vec![int_col, str_col]).unwrap();
        let names = vec!["id".to_string(), "name".to_string()];
        let types = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];

        // SFrameRows → RecordBatch → SFrameRows
        let batch = sframe_rows_to_record_batch(&rows, &names, &types).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);

        let rows2 = record_batch_to_sframe_rows(&batch).unwrap();
        assert_eq!(rows2.num_rows(), 3);
        assert_eq!(rows2.num_columns(), 2);
    }
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement both functions**

`record_batch_to_sframe_rows`: iterate columns, call `arrow_array_to_column` for each,
infer `FlexTypeEnum` from Arrow `DataType` via `arrow_type_to_sframe`. Construct `SFrameRows::new(columns)`.

`sframe_rows_to_record_batch`: iterate columns, call `column_to_arrow_array` for each.
Build Arrow `Schema` from names + `sframe_type_to_arrow(type)`. Call `RecordBatch::try_new(schema, arrays)`.

**Step 4: Run tests**

Run: `cargo test -p sframe-parquet`

**Step 5: Commit**

```
feat(parquet): add RecordBatch ↔ SFrameRows batch conversion
```

---

### Task 6: Implement Parquet reader

**Files:**
- Modify: `crates/sframe-parquet/src/parquet_reader.rs`

**Step 1: Write a test that reads a Parquet file**

We need a Parquet file for testing. The simplest approach: write one in the test using
the `parquet` crate's `ArrowWriter`, then read it back with our reader.

```rust
use std::fs::File;
use std::path::Path;
use parquet::arrow::ArrowWriter;
use arrow::datatypes::{Schema, Field, DataType};
use arrow::array::{Int64Array, StringArray, Float64Array};
use arrow::record_batch::RecordBatch;

fn write_test_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("alice"), None, Some("charlie")])),
            Arc::new(Float64Array::from(vec![Some(95.5), Some(87.0), None])),
        ],
    ).unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_parquet_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path);

        let (names, types) = read_parquet_schema(path.to_str().unwrap()).unwrap();
        assert_eq!(names, vec!["id", "name", "score"]);
        assert_eq!(types, vec![FlexTypeEnum::Integer, FlexTypeEnum::String, FlexTypeEnum::Float]);
    }

    #[test]
    fn test_read_parquet_batches_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path);

        let mut iter = read_parquet_batches(&[path]).unwrap();
        let batch = iter.next_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3);
        assert!(iter.next_batch().is_none()); // only one batch
    }

    #[test]
    fn test_read_parquet_multiple_files() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("part1.parquet");
        let path2 = dir.path().join("part2.parquet");
        write_test_parquet(&path1);
        write_test_parquet(&path2);

        let mut iter = read_parquet_batches(&[path1, path2]).unwrap();
        let mut total_rows = 0;
        while let Some(batch) = iter.next_batch() {
            total_rows += batch.unwrap().num_rows();
        }
        assert_eq!(total_rows, 6); // 3 + 3
    }
}
```

Add `tempfile` as a dev-dependency in `crates/sframe-parquet/Cargo.toml`.

**Step 2: Run tests to verify they fail**

**Step 3: Implement `read_parquet_schema` and `read_parquet_batches`**

`read_parquet_schema`:
```rust
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub fn read_parquet_schema(path: &str) -> Result<(Vec<String>, Vec<FlexTypeEnum>)> {
    let file = File::open(path).map_err(SFrameError::Io)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| SFrameError::Format(format!("Parquet error: {}", e)))?;
    let schema = builder.schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let types: Vec<FlexTypeEnum> = schema.fields().iter()
        .map(|f| arrow_type_to_sframe(f.data_type()))
        .collect::<Result<_>>()?;
    Ok((names, types))
}
```

`read_parquet_batches`:
```rust
use sframe_query::execute::batch_iter::{BatchIterator, BatchCo, BatchResponse, BatchCommand};

pub fn read_parquet_batches(paths: &[PathBuf]) -> Result<BatchIterator> {
    let paths = paths.to_vec();
    // Read schema from first file to infer SFrame types
    let first_path = paths.first()
        .ok_or_else(|| SFrameError::Format("No parquet files provided".into()))?;
    let (_, sframe_types) = read_parquet_schema(first_path.to_str().unwrap())?;

    Ok(BatchIterator::new(move |co: BatchCo| async move {
        let cmd = co.yield_(BatchResponse::Ready).await;
        if !matches!(cmd, BatchCommand::NextBatch) { return; }

        for path in &paths {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    co.yield_(BatchResponse::Batch(Err(SFrameError::Io(e)))).await;
                    return;
                }
            };
            let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
                Ok(b) => b,
                Err(e) => {
                    co.yield_(BatchResponse::Batch(Err(
                        SFrameError::Format(format!("Parquet error: {}", e))
                    ))).await;
                    return;
                }
            };
            let reader = match builder.build() {
                Ok(r) => r,
                Err(e) => {
                    co.yield_(BatchResponse::Batch(Err(
                        SFrameError::Format(format!("Parquet error: {}", e))
                    ))).await;
                    return;
                }
            };

            for record_batch_result in reader {
                let record_batch = match record_batch_result {
                    Ok(rb) => rb,
                    Err(e) => {
                        co.yield_(BatchResponse::Batch(Err(
                            SFrameError::Format(format!("Parquet error: {}", e))
                        ))).await;
                        return;
                    }
                };
                let sframe_rows = match record_batch_to_sframe_rows(&record_batch) {
                    Ok(rows) => rows,
                    Err(e) => {
                        co.yield_(BatchResponse::Batch(Err(e))).await;
                        return;
                    }
                };
                let cmd = co.yield_(BatchResponse::Batch(Ok(sframe_rows))).await;
                if !matches!(cmd, BatchCommand::NextBatch) { return; }
            }
        }
    }))
}
```

**Step 4: Run tests**

Run: `cargo test -p sframe-parquet`

**Step 5: Commit**

```
feat(parquet): implement Parquet file reader with BatchIterator
```

---

### Task 7: Implement glob expansion helper

**Files:**
- Modify: `crates/sframe-parquet/src/parquet_reader.rs`

**Step 1: Write tests**

```rust
/// Resolve a path to a list of Parquet files.
/// If path contains glob characters (*, ?, [), expand the glob.
/// Otherwise treat as a single file.
pub fn resolve_parquet_paths(path: &str) -> Result<Vec<PathBuf>> {
    todo!()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_resolve_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path);

        let paths = resolve_parquet_paths(path.to_str().unwrap()).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], path);
    }

    #[test]
    fn test_resolve_glob() {
        let dir = tempfile::tempdir().unwrap();
        write_test_parquet(&dir.path().join("a.parquet"));
        write_test_parquet(&dir.path().join("b.parquet"));

        let pattern = format!("{}/*.parquet", dir.path().to_str().unwrap());
        let mut paths = resolve_parquet_paths(&pattern).unwrap();
        paths.sort();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_resolve_no_matches() {
        let result = resolve_parquet_paths("/nonexistent/*.parquet");
        assert!(result.is_err());
    }
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `resolve_parquet_paths`**

```rust
pub fn resolve_parquet_paths(path: &str) -> Result<Vec<PathBuf>> {
    if path.contains('*') || path.contains('?') || path.contains('[') {
        let mut paths: Vec<PathBuf> = glob::glob(path)
            .map_err(|e| SFrameError::Format(format!("Invalid glob pattern: {}", e)))?
            .filter_map(|entry| entry.ok())
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(SFrameError::Format(format!("No files matched pattern: {}", path)));
        }
        Ok(paths)
    } else {
        let p = PathBuf::from(path);
        if !p.exists() {
            return Err(SFrameError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", path),
            )));
        }
        Ok(vec![p])
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p sframe-parquet`

**Step 5: Commit**

```
feat(parquet): add glob-based parquet file resolution
```

---

### Task 8: Implement Parquet writer (single file)

**Files:**
- Modify: `crates/sframe-parquet/src/parquet_writer.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.parquet");

        // Build SFrameRows
        let int_col = ColumnData::Integer(vec![Some(1), Some(2), None]);
        let str_col = ColumnData::String(vec![
            Some("a".into()), Some("b".into()), Some("c".into()),
        ]);
        let float_col = ColumnData::Float(vec![Some(1.5), None, Some(3.5)]);
        let rows = SFrameRows::new(vec![int_col, str_col, float_col]).unwrap();

        let names = vec!["id".into(), "name".into(), "score".into()];
        let types = vec![FlexTypeEnum::Integer, FlexTypeEnum::String, FlexTypeEnum::Float];

        // Write via BatchIterator
        let batch_rows = rows.clone();
        let iter = BatchIterator::new(move |co: BatchCo| async move {
            co.yield_(BatchResponse::Ready).await;
            co.yield_(BatchResponse::Batch(Ok(batch_rows))).await;
        });

        write_parquet(iter, &names, &types, path.as_path()).unwrap();

        // Read back with parquet crate and verify
        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap().build().unwrap();
        let batches: Vec<_> = reader.collect::<std::result::Result<_, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
    }

    #[test]
    fn test_write_parquet_v2() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.parquet");

        let col = ColumnData::Integer(vec![Some(42)]);
        let rows = SFrameRows::new(vec![col]).unwrap();
        let names = vec!["x".into()];
        let types = vec![FlexTypeEnum::Integer];

        let batch_rows = rows.clone();
        let iter = BatchIterator::new(move |co: BatchCo| async move {
            co.yield_(BatchResponse::Ready).await;
            co.yield_(BatchResponse::Batch(Ok(batch_rows))).await;
        });

        write_parquet(iter, &names, &types, path.as_path()).unwrap();

        // Verify file is valid parquet
        let metadata = parquet::file::reader::SerializedFileReader::try_from(
            File::open(&path).unwrap()
        ).unwrap();
        // Just verify it opens without error — v2 data pages are an internal detail
    }
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `write_parquet`**

```rust
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

pub fn write_parquet(
    mut iter: BatchIterator,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
    path: &Path,
) -> Result<()> {
    let arrow_fields: Vec<Field> = column_names.iter().zip(column_types.iter())
        .map(|(name, dtype)| Field::new(name, sframe_type_to_arrow(*dtype), true))
        .collect();
    let schema = Arc::new(Schema::new(arrow_fields));

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
        .build();

    let file = File::create(path).map_err(SFrameError::Io)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| SFrameError::Format(format!("Parquet write error: {}", e)))?;

    while let Some(batch_result) = iter.next_batch() {
        let sframe_rows = batch_result?;
        let record_batch = sframe_rows_to_record_batch(&sframe_rows, column_names, column_types)?;
        writer.write(&record_batch)
            .map_err(|e| SFrameError::Format(format!("Parquet write error: {}", e)))?;
    }

    writer.close()
        .map_err(|e| SFrameError::Format(format!("Parquet close error: {}", e)))?;

    Ok(())
}
```

**Step 4: Run tests**

Run: `cargo test -p sframe-parquet`

**Step 5: Commit**

```
feat(parquet): implement single-file Parquet writer
```

---

### Task 9: Implement sharded Parquet writer

**Files:**
- Modify: `crates/sframe-parquet/src/parquet_writer.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_write_sharded() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("data").to_str().unwrap().to_string();

        // Build two batches to ensure we get content
        let col1 = ColumnData::Integer(vec![Some(1), Some(2)]);
        let col2 = ColumnData::Integer(vec![Some(3), Some(4)]);
        let rows1 = SFrameRows::new(vec![col1]).unwrap();
        let rows2 = SFrameRows::new(vec![col2]).unwrap();

        let names = vec!["x".into()];
        let types = vec![FlexTypeEnum::Integer];

        write_parquet_sharded_simple(
            vec![rows1, rows2],
            &names,
            &types,
            &prefix,
        ).unwrap();

        // Check files exist with correct naming pattern
        let pattern = format!("{}*.parquet", prefix);
        let files = resolve_parquet_paths(&pattern).unwrap_or_default();
        // We should have files named prefix_0_of_N.parquet, prefix_1_of_N.parquet, ...
        assert!(!files.is_empty());
        for f in &files {
            let name = f.file_name().unwrap().to_str().unwrap();
            assert!(name.contains("_of_"));
            assert!(name.ends_with(".parquet"));
        }

        // Read all back and verify total row count
        let mut iter = read_parquet_batches(&files).unwrap();
        let mut total = 0;
        while let Some(b) = iter.next_batch() {
            total += b.unwrap().num_rows();
        }
        assert_eq!(total, 4);
    }
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement sharded writer**

The sharded writer needs to work with the parallel execution pipeline. Provide:

```rust
/// Write batches to sharded Parquet files.
/// Each shard is written independently. The naming convention is
/// `{prefix}_{n}_of_{N}.parquet`.
///
/// This is designed to be called from parallel workers, where each worker
/// calls `write_parquet_shard` with its own shard index.
pub fn write_parquet_shard(
    mut iter: BatchIterator,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
    prefix: &str,
    shard_index: usize,
    total_shards: usize,
) -> Result<()> {
    let path = PathBuf::from(format!("{}_{}_of_{}.parquet", prefix, shard_index, total_shards));
    write_parquet(iter, column_names, column_types, &path)
}
```

Also provide a convenience function for testing that takes a `Vec<SFrameRows>`:

```rust
/// Simple sharded write for testing — writes each batch as a separate shard.
fn write_parquet_sharded_simple(
    batches: Vec<SFrameRows>,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
    prefix: &str,
) -> Result<()> { ... }
```

**Step 4: Run tests**

Run: `cargo test -p sframe-parquet`

**Step 5: Commit**

```
feat(parquet): implement sharded Parquet writer
```

---

### Task 10: Wire into SFrame — `from_parquet` and `from_parquet_files`

**Files:**
- Modify: `crates/sframe/src/sframe.rs`
- Modify: `crates/sframe/Cargo.toml` (already done in Task 1)

**Step 1: Write tests**

Add to the `#[cfg(test)]` module in `sframe.rs`:

```rust
#[test]
fn test_from_parquet_single_file() {
    // Write a test parquet file
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.parquet");
    // Use arrow crate to write a simple parquet file
    // ... (same write_test_parquet helper)

    let sf = SFrame::from_parquet(path.to_str().unwrap()).unwrap();
    assert_eq!(sf.num_columns(), 3);
    assert_eq!(sf.num_rows().unwrap(), 3);
    assert_eq!(sf.column_names(), &["id", "name", "score"]);
}

#[test]
fn test_from_parquet_glob() {
    let dir = tempfile::tempdir().unwrap();
    write_test_parquet(&dir.path().join("part1.parquet"));
    write_test_parquet(&dir.path().join("part2.parquet"));

    let pattern = format!("{}/*.parquet", dir.path().to_str().unwrap());
    let sf = SFrame::from_parquet(&pattern).unwrap();
    assert_eq!(sf.num_rows().unwrap(), 6); // 3 + 3
}

#[test]
fn test_from_parquet_files() {
    let dir = tempfile::tempdir().unwrap();
    let p1 = dir.path().join("a.parquet");
    let p2 = dir.path().join("b.parquet");
    write_test_parquet(&p1);
    write_test_parquet(&p2);

    let sf = SFrame::from_parquet_files(&[
        p1.to_str().unwrap(),
        p2.to_str().unwrap(),
    ]).unwrap();
    assert_eq!(sf.num_rows().unwrap(), 6);
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `from_parquet` and `from_parquet_files`**

Follow the same pattern as `from_csv` — eagerly read schema, build an SFrameBuilder
that writes to cache, stream Parquet batches through it.

```rust
pub fn from_parquet(path: &str) -> Result<Self> {
    let paths = sframe_parquet::parquet_reader::resolve_parquet_paths(path)?;
    Self::from_parquet_resolved(paths)
}

pub fn from_parquet_files(paths: &[&str]) -> Result<Self> {
    let paths: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    Self::from_parquet_resolved(paths)
}

fn from_parquet_resolved(paths: Vec<PathBuf>) -> Result<Self> {
    let first = paths.first()
        .ok_or_else(|| SFrameError::Format("No parquet files".into()))?;
    let (col_names, col_types) = sframe_parquet::parquet_reader::read_parquet_schema(
        first.to_str().unwrap()
    )?;

    let mut builder = SFrameBuilder::anonymous(col_names.clone(), col_types.clone())?;
    let mut iter = sframe_parquet::parquet_reader::read_parquet_batches(&paths)?;
    while let Some(batch_result) = iter.next_batch() {
        let batch = batch_result?;
        builder.write_batch_chunked(&batch, DEFAULT_CHUNK_SIZE)?;
    }
    builder.finish()
}
```

**Step 4: Run tests**

Run: `cargo test -p sframe -- test_from_parquet`

**Step 5: Commit**

```
feat(sframe): add from_parquet and from_parquet_files methods
```

---

### Task 11: Wire into SFrame — `to_parquet` and `to_parquet_sharded`

**Files:**
- Modify: `crates/sframe/src/sframe.rs`

**Step 1: Write tests**

```rust
#[test]
fn test_to_parquet_roundtrip() {
    let sf = SFrame::from_columns(vec![
        ("id", SArray::from_vec(
            vec![FlexType::Integer(1), FlexType::Integer(2), FlexType::Integer(3)],
            FlexTypeEnum::Integer,
        ).unwrap()),
        ("name", SArray::from_vec(
            vec![FlexType::String("a".into()), FlexType::String("b".into()), FlexType::String("c".into())],
            FlexTypeEnum::String,
        ).unwrap()),
    ]).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.parquet");
    sf.to_parquet(path.to_str().unwrap()).unwrap();

    let sf2 = SFrame::from_parquet(path.to_str().unwrap()).unwrap();
    assert_eq!(sf2.num_rows().unwrap(), 3);
    assert_eq!(sf2.column_names(), &["id", "name"]);
}

#[test]
fn test_to_parquet_sharded_roundtrip() {
    let sf = SFrame::from_columns(vec![
        ("x", SArray::from_vec(
            (0..100).map(|i| FlexType::Integer(i)).collect(),
            FlexTypeEnum::Integer,
        ).unwrap()),
    ]).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("shard").to_str().unwrap().to_string();
    sf.to_parquet_sharded(&prefix).unwrap();

    // Read back via glob
    let pattern = format!("{}*.parquet", prefix);
    let sf2 = SFrame::from_parquet(&pattern).unwrap();
    assert_eq!(sf2.num_rows().unwrap(), 100);
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `to_parquet` and `to_parquet_sharded`**

`to_parquet` — follows the same pattern as `to_csv`:
```rust
pub fn to_parquet(&self, path: &str) -> Result<()> {
    let stream = self.compile_stream()?;
    let names = self.column_names.clone();
    let types = self.column_types();
    sframe_parquet::parquet_writer::write_parquet(
        stream,
        &names,
        &types,
        Path::new(path),
    )
}
```

`to_parquet_sharded` — uses rayon parallel execution. Each thread writes its own shard:
```rust
pub fn to_parquet_sharded(&self, prefix: &str) -> Result<()> {
    let fused = self.fuse_plan()?;
    let names = self.column_names.clone();
    let types = self.column_types();
    let n_workers = rayon::current_num_threads().max(1);

    // Try to get total row count for slicing
    let total_rows = self.len()?;

    // Build per-worker plans with row ranges
    let worker_plans: Vec<Arc<PlannerNode>> = (0..n_workers)
        .filter_map(|i| {
            let begin = (i as u64 * total_rows) / n_workers as u64;
            let end = ((i as u64 + 1) * total_rows) / n_workers as u64;
            if begin >= end { return None; }
            Some(clone_plan_with_row_range(&fused, begin, end))
        })
        .collect();

    let actual_shards = worker_plans.len();
    let prefix = prefix.to_string();

    worker_plans.into_par_iter().enumerate().try_for_each(|(i, plan)| {
        let iter = compile_single_threaded(&plan)?;
        sframe_parquet::parquet_writer::write_parquet_shard(
            iter, &names, &types, &prefix, i, actual_shards,
        )
    })?;

    Ok(())
}
```

Note: This needs `clone_plan_with_row_range` and `compile_single_threaded` to be
accessible. Check visibility — if they're `pub(crate)` in `sframe-query`, they may
need to be made `pub` or a helper added. The implementation may need to adapt based
on what's actually accessible. If the plan is not parallel-sliceable (e.g. it has
filters/transforms), fall back to single-file sequential write.

**Step 4: Run tests**

Run: `cargo test -p sframe -- test_to_parquet`

**Step 5: Commit**

```
feat(sframe): add to_parquet and to_parquet_sharded methods
```

---

### Task 12: Python bindings

**Files:**
- Modify: `crates/sframe-py/src/py_sframe.rs`

**Step 1: Write a Python test**

Create `crates/sframe-py/tests/test_parquet.py`:

```python
import sframe
import tempfile
import os

def test_parquet_roundtrip():
    sf = sframe.SFrame.from_columns({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.parquet")
        sf.to_parquet(path)
        sf2 = sframe.SFrame.from_parquet(path)
        assert sf2.num_rows() == 3
        assert sf2.column_names() == ["id", "name"]

def test_parquet_sharded_roundtrip():
    sf = sframe.SFrame.from_columns({"x": list(range(100))})
    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "shard")
        sf.to_parquet(prefix, sharded=True)
        sf2 = sframe.SFrame.from_parquet(os.path.join(d, "shard*.parquet"))
        assert sf2.num_rows() == 100

def test_parquet_from_list():
    sf = sframe.SFrame.from_columns({"x": [1, 2, 3]})
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "a.parquet")
        p2 = os.path.join(d, "b.parquet")
        sf.to_parquet(p1)
        sf.to_parquet(p2)
        sf2 = sframe.SFrame.from_parquet([p1, p2])
        assert sf2.num_rows() == 6
```

**Step 2: Add Python bindings**

```rust
/// Read Parquet file(s) into an SFrame.
/// Accepts a single path (with optional glob), or a list of paths.
#[staticmethod]
#[pyo3(signature = (path))]
fn from_parquet(path: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
    if let Ok(list) = path.downcast::<pyo3::types::PyList>() {
        let paths: Vec<String> = list.iter()
            .map(|item| item.extract::<String>())
            .collect::<std::result::Result<_, _>>()?;
        let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let sf = allow(py, move || SFrame::from_parquet_files(&path_refs))?;
        Ok(PySFrame::new(sf))
    } else {
        let path_str: String = path.extract()?;
        let sf = allow(py, move || SFrame::from_parquet(&path_str))?;
        Ok(PySFrame::new(sf))
    }
}

/// Write SFrame to Parquet.
#[pyo3(signature = (path, sharded=false))]
fn to_parquet(&self, path: &str, sharded: bool, py: Python<'_>) -> PyResult<()> {
    let inner = self.inner.clone();
    let path = path.to_string();
    if sharded {
        allow(py, move || inner.to_parquet_sharded(&path))
    } else {
        allow(py, move || inner.to_parquet(&path))
    }
}
```

**Step 3: Build and test**

Run: `cd crates/sframe-py && maturin develop && python -m pytest tests/test_parquet.py -v`

(Adapt the build command to whatever the project uses — check existing Makefile/scripts.)

**Step 4: Commit**

```
feat(sframe-py): add from_parquet and to_parquet Python bindings
```

---

### Task 13: Integration test — CSV ↔ Parquet roundtrip

**Files:**
- Modify: `crates/sframe/src/sframe.rs` (add test in `#[cfg(test)]` block)

**Step 1: Write integration test**

```rust
#[test]
fn test_csv_to_parquet_roundtrip() {
    let csv_path = format!("{}/business.csv", samples_dir());
    let sf = SFrame::from_csv(&csv_path, None).unwrap();
    let original_rows = sf.num_rows().unwrap();
    let original_names = sf.column_names().to_vec();

    let dir = tempfile::tempdir().unwrap();
    let parquet_path = dir.path().join("business.parquet");
    sf.to_parquet(parquet_path.to_str().unwrap()).unwrap();

    let sf2 = SFrame::from_parquet(parquet_path.to_str().unwrap()).unwrap();
    assert_eq!(sf2.num_rows().unwrap(), original_rows);
    assert_eq!(sf2.column_names(), &original_names);

    // Spot-check first few rows match
    let head1 = sf.head(5).unwrap();
    let head2 = sf2.head(5).unwrap();
    assert_eq!(head1.len(), head2.len());
}

#[test]
fn test_parquet_sharded_roundtrip_with_data() {
    let csv_path = format!("{}/business.csv", samples_dir());
    let sf = SFrame::from_csv(&csv_path, None).unwrap();
    let original_rows = sf.num_rows().unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("sharded").to_str().unwrap().to_string();
    sf.to_parquet_sharded(&prefix).unwrap();

    let pattern = format!("{}*.parquet", prefix);
    let sf2 = SFrame::from_parquet(&pattern).unwrap();
    assert_eq!(sf2.num_rows().unwrap(), original_rows);
}
```

**Step 2: Run tests**

Run: `cargo test -p sframe -- test_csv_to_parquet`
Run: `cargo test -p sframe -- test_parquet_sharded`

**Step 3: Commit**

```
test(parquet): add CSV ↔ Parquet roundtrip integration tests
```

---

### Task 14: Final verification

**Step 1: Run full test suite**

Run: `cargo test`
Expected: All tests pass, including existing tests (no regressions).

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: No new warnings.

**Step 3: Commit any fixes if needed**

---

## Implementation Notes

**Dependency versions**: At implementation time, check the latest `arrow-rs` release and
use matching versions for `arrow` and `parquet` crates. The `arrow` and `parquet` crate
versions must always match.

**Visibility**: Several functions in `sframe-query` (like `compile_single_threaded`,
`clone_plan_with_row_range`) may be `pub(crate)`. If `to_parquet_sharded` needs them,
either make them `pub` or move the parallel logic into `sframe-parquet` with
`sframe-query` exposing the needed primitives. Evaluate at implementation time.

**Error mapping**: Arrow/Parquet errors don't implement `From` for `SFrameError`.
Use `.map_err(|e| SFrameError::Format(format!(...)))` consistently.

**Task 3 complexity**: The Arrow array → ColumnData conversion for DateTime and nested
types (List, Dict) is the most complex part. Consider splitting into sub-PRs if it
gets unwieldy. Get Integer/Float/String/Boolean working first, then add the rest.
