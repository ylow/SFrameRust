// Parquet file reader producing SFrame batches.

use std::fs::File;
use std::path::PathBuf;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use sframe_query::execute::{BatchCo, BatchCommand, BatchIterator, BatchResponse};
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexTypeEnum;

use crate::type_mapping::{arrow_type_to_sframe, record_batch_to_sframe_rows};

// ---------------------------------------------------------------------------
// Schema reading
// ---------------------------------------------------------------------------

/// Read the schema from a single Parquet file, returning column names and
/// their corresponding SFrame types.
pub fn read_parquet_schema(path: &str) -> Result<(Vec<String>, Vec<FlexTypeEnum>)> {
    let file = File::open(path).map_err(SFrameError::Io)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| SFrameError::Format(format!("Parquet error: {e}")))?;
    let schema = builder.schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let types: Vec<FlexTypeEnum> = schema
        .fields()
        .iter()
        .map(|f| arrow_type_to_sframe(f.data_type()))
        .collect::<Result<_>>()?;
    Ok((names, types))
}

// ---------------------------------------------------------------------------
// Row counting (metadata only, no data read)
// ---------------------------------------------------------------------------

/// Count total rows across all Parquet files by reading file metadata only.
pub fn count_parquet_rows(paths: &[PathBuf]) -> Result<u64> {
    let mut total = 0u64;
    for path in paths {
        let file = File::open(path).map_err(SFrameError::Io)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| SFrameError::Format(format!("Parquet error: {e}")))?;
        let metadata = builder.metadata();
        for rg in metadata.row_groups() {
            total += rg.num_rows() as u64;
        }
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Batch reading
// ---------------------------------------------------------------------------

/// Create a `BatchIterator` that reads row groups from all given Parquet files
/// sequentially, converting each `RecordBatch` into `SFrameRows`.
pub fn read_parquet_batches(paths: &[PathBuf]) -> Result<BatchIterator> {
    if paths.is_empty() {
        return Err(SFrameError::Format(
            "No parquet files provided".to_string(),
        ));
    }
    let paths = paths.to_vec();

    Ok(BatchIterator::new(move |co: BatchCo| async move {
        let mut cmd = co.yield_(BatchResponse::Ready).await;
        if !matches!(cmd, BatchCommand::NextBatch | BatchCommand::SkipBatch) {
            return;
        }

        for path in &paths {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    co.yield_(BatchResponse::Batch(Err(SFrameError::Io(e))))
                        .await;
                    return;
                }
            };
            let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
                Ok(b) => b,
                Err(e) => {
                    co.yield_(BatchResponse::Batch(Err(SFrameError::Format(
                        format!("Parquet error: {e}"),
                    ))))
                    .await;
                    return;
                }
            };
            let reader = match builder.build() {
                Ok(r) => r,
                Err(e) => {
                    co.yield_(BatchResponse::Batch(Err(SFrameError::Format(
                        format!("Parquet error: {e}"),
                    ))))
                    .await;
                    return;
                }
            };

            for record_batch_result in reader {
                let record_batch = match record_batch_result {
                    Ok(rb) => rb,
                    Err(e) => {
                        co.yield_(BatchResponse::Batch(Err(SFrameError::Format(
                            format!("Parquet error: {e}"),
                        ))))
                        .await;
                        return;
                    }
                };

                match cmd {
                    BatchCommand::NextBatch => {
                        let sframe_rows = match record_batch_to_sframe_rows(&record_batch) {
                            Ok(rows) => rows,
                            Err(e) => {
                                co.yield_(BatchResponse::Batch(Err(e))).await;
                                return;
                            }
                        };
                        cmd = co.yield_(BatchResponse::Batch(Ok(sframe_rows))).await;
                    }
                    BatchCommand::SkipBatch => {
                        cmd = co.yield_(BatchResponse::Skipped).await;
                    }
                    _ => return,
                }
            }
        }
    }))
}

// ---------------------------------------------------------------------------
// Glob expansion
// ---------------------------------------------------------------------------

/// Resolve a path string to a sorted list of Parquet file paths.
///
/// If the path contains glob characters (`*`, `?`, or `[`), expand the glob
/// pattern. Otherwise treat it as a single file path.
///
/// Returns an error if no files match the pattern, or if a non-glob path
/// does not exist.
pub fn resolve_parquet_paths(path: &str) -> Result<Vec<PathBuf>> {
    if path.contains('*') || path.contains('?') || path.contains('[') {
        let mut paths: Vec<PathBuf> = glob::glob(path)
            .map_err(|e| SFrameError::Format(format!("Invalid glob pattern: {e}")))?
            .filter_map(|entry| entry.ok())
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(SFrameError::Format(format!(
                "No files matched pattern: {path}"
            )));
        }
        Ok(paths)
    } else {
        let p = PathBuf::from(path);
        if !p.exists() {
            return Err(SFrameError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {path}"),
            )));
        }
        Ok(vec![p])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    use sframe_types::flex_type::FlexType;

    /// Helper: write a simple test Parquet file with columns (id: i64, name: string).
    fn write_test_parquet(path: &std::path::Path, ids: &[i64], names: &[&str]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let id_array = Int64Array::from(ids.to_vec());
        let name_array = StringArray::from(names.to_vec());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(id_array), Arc::new(name_array)],
        )
        .unwrap();

        let file = File::create(path).unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Helper: write a minimal Parquet file (used for glob tests that only need
    /// file existence).
    fn write_minimal_parquet(path: &std::path::Path) {
        write_test_parquet(path, &[1], &["x"]);
    }

    // ===== Task 6: read_parquet_schema =====

    #[test]
    fn test_read_parquet_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path, &[1, 2, 3], &["a", "b", "c"]);

        let (names, types) = read_parquet_schema(path.to_str().unwrap()).unwrap();
        assert_eq!(names, vec!["id", "name"]);
        assert_eq!(types, vec![FlexTypeEnum::Integer, FlexTypeEnum::String]);
    }

    #[test]
    fn test_read_parquet_schema_nonexistent() {
        let result = read_parquet_schema("/nonexistent/path/file.parquet");
        assert!(result.is_err());
    }

    // ===== Task 6: count_parquet_rows =====

    #[test]
    fn test_count_parquet_rows_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path, &[1, 2, 3], &["a", "b", "c"]);

        let count = count_parquet_rows(&[path]).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_count_parquet_rows_multiple_files() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("a.parquet");
        let path2 = dir.path().join("b.parquet");
        write_test_parquet(&path1, &[1, 2, 3], &["a", "b", "c"]);
        write_test_parquet(&path2, &[4, 5], &["d", "e"]);

        let count = count_parquet_rows(&[path1, path2]).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_count_parquet_rows_empty_list() {
        let count = count_parquet_rows(&[]).unwrap();
        assert_eq!(count, 0);
    }

    // ===== Task 6: read_parquet_batches =====

    #[test]
    fn test_read_parquet_batches_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path, &[10, 20, 30], &["alpha", "beta", "gamma"]);

        let mut iter = read_parquet_batches(&[path]).unwrap();
        let mut total_rows = 0;
        let mut all_ids = Vec::new();
        let mut all_names = Vec::new();

        while let Some(batch_result) = iter.next_batch() {
            let batch = batch_result.unwrap();
            total_rows += batch.num_rows();
            for i in 0..batch.num_rows() {
                let row = batch.row(i);
                if let FlexType::Integer(id) = &row[0] {
                    all_ids.push(*id);
                }
                if let FlexType::String(name) = &row[1] {
                    all_names.push(name.to_string());
                }
            }
        }

        assert_eq!(total_rows, 3);
        assert_eq!(all_ids, vec![10, 20, 30]);
        assert_eq!(all_names, vec!["alpha", "beta", "gamma"]);
        assert!(iter.is_done());
    }

    #[test]
    fn test_read_parquet_batches_multiple_files() {
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("part1.parquet");
        let path2 = dir.path().join("part2.parquet");
        write_test_parquet(&path1, &[1, 2, 3], &["a", "b", "c"]);
        write_test_parquet(&path2, &[4, 5, 6], &["d", "e", "f"]);

        let mut iter = read_parquet_batches(&[path1, path2]).unwrap();
        let mut total_rows = 0;
        while let Some(batch_result) = iter.next_batch() {
            total_rows += batch_result.unwrap().num_rows();
        }
        assert_eq!(total_rows, 6);
    }

    #[test]
    fn test_read_parquet_batches_empty_paths() {
        let result = read_parquet_batches(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_parquet_batches_skip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_test_parquet(&path, &[1, 2, 3], &["a", "b", "c"]);

        let mut iter = read_parquet_batches(&[path]).unwrap();
        // Skip the first (and only) batch.
        let skipped = iter.skip_batch();
        assert_eq!(skipped, Some(()));
        // Should be done now.
        assert!(iter.next_batch().is_none());
    }

    #[test]
    fn test_read_parquet_batches_float_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("floats.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, true),
        ]));
        let arr = Float64Array::from(vec![Some(1.5), None, Some(3.14)]);
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let mut iter = read_parquet_batches(&[path]).unwrap();
        let batch = iter.next_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 1);

        let row0 = batch.row(0);
        assert_eq!(row0, vec![FlexType::Float(1.5)]);
        let row1 = batch.row(1);
        assert_eq!(row1, vec![FlexType::Undefined]);
        let row2 = batch.row(2);
        assert_eq!(row2, vec![FlexType::Float(3.14)]);
    }

    #[test]
    fn test_read_parquet_batches_multiple_row_groups() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_rg.parquet");

        let schema = Arc::new(Schema::new(vec![Field::new(
            "x",
            DataType::Int64,
            false,
        )]));

        // Write multiple row groups by calling write() multiple times.
        let props = WriterProperties::builder().build();
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();

        let arr1 = Int64Array::from(vec![1, 2]);
        let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr1)]).unwrap();
        writer.write(&batch1).unwrap();
        writer.flush().unwrap(); // flush forces a new row group

        let arr2 = Int64Array::from(vec![3, 4, 5]);
        let batch2 = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr2)]).unwrap();
        writer.write(&batch2).unwrap();
        writer.close().unwrap();

        // Verify metadata shows 2 row groups.
        let count = count_parquet_rows(&[path.clone()]).unwrap();
        assert_eq!(count, 5);

        // Read all data and verify values.
        let mut iter = read_parquet_batches(&[path]).unwrap();
        let mut all_values = Vec::new();
        while let Some(result) = iter.next_batch() {
            let b = result.unwrap();
            for i in 0..b.num_rows() {
                if let FlexType::Integer(v) = b.row(i)[0] {
                    all_values.push(v);
                }
            }
        }
        assert_eq!(all_values, vec![1, 2, 3, 4, 5]);
    }

    // ===== Task 7: resolve_parquet_paths =====

    #[test]
    fn test_resolve_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        write_minimal_parquet(&path);

        let paths = resolve_parquet_paths(path.to_str().unwrap()).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], path);
    }

    #[test]
    fn test_resolve_single_file_not_found() {
        let result = resolve_parquet_paths("/nonexistent/file.parquet");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SFrameError::Io(_)),
            "Expected Io error, got: {err:?}"
        );
    }

    #[test]
    fn test_resolve_glob_star() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_parquet(&dir.path().join("a.parquet"));
        write_minimal_parquet(&dir.path().join("b.parquet"));
        write_minimal_parquet(&dir.path().join("c.txt")); // should not match

        let pattern = format!("{}/*.parquet", dir.path().to_str().unwrap());
        let paths = resolve_parquet_paths(&pattern).unwrap();
        assert_eq!(paths.len(), 2);
        // Results should be sorted.
        assert!(paths[0] < paths[1]);
        assert!(paths[0].to_str().unwrap().contains("a.parquet"));
        assert!(paths[1].to_str().unwrap().contains("b.parquet"));
    }

    #[test]
    fn test_resolve_glob_question_mark() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_parquet(&dir.path().join("part0.parquet"));
        write_minimal_parquet(&dir.path().join("part1.parquet"));
        write_minimal_parquet(&dir.path().join("part12.parquet")); // two digits, should not match ?

        let pattern = format!("{}/part?.parquet", dir.path().to_str().unwrap());
        let paths = resolve_parquet_paths(&pattern).unwrap();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_resolve_glob_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        // No files created, glob should fail.
        let pattern = format!("{}/*.parquet", dir.path().to_str().unwrap());
        let result = resolve_parquet_paths(&pattern);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SFrameError::Format(ref msg) if msg.contains("No files matched")),
            "Expected Format error about no matches, got: {err:?}"
        );
    }

    #[test]
    fn test_resolve_glob_bracket() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_parquet(&dir.path().join("data_a.parquet"));
        write_minimal_parquet(&dir.path().join("data_b.parquet"));
        write_minimal_parquet(&dir.path().join("data_c.parquet"));

        let pattern = format!("{}/data_[ab].parquet", dir.path().to_str().unwrap());
        let paths = resolve_parquet_paths(&pattern).unwrap();
        assert_eq!(paths.len(), 2);
    }

    // ===== Integration: resolve + read =====

    #[test]
    fn test_resolve_then_read() {
        let dir = tempfile::tempdir().unwrap();
        write_test_parquet(&dir.path().join("part0.parquet"), &[1, 2], &["a", "b"]);
        write_test_parquet(&dir.path().join("part1.parquet"), &[3, 4], &["c", "d"]);

        let pattern = format!("{}/*.parquet", dir.path().to_str().unwrap());
        let paths = resolve_parquet_paths(&pattern).unwrap();
        assert_eq!(paths.len(), 2);

        // Read schema from first file.
        let (names, types) = read_parquet_schema(paths[0].to_str().unwrap()).unwrap();
        assert_eq!(names, vec!["id", "name"]);
        assert_eq!(types, vec![FlexTypeEnum::Integer, FlexTypeEnum::String]);

        // Count rows.
        let count = count_parquet_rows(&paths).unwrap();
        assert_eq!(count, 4);

        // Read all batches.
        let mut iter = read_parquet_batches(&paths).unwrap();
        let mut total = 0;
        while let Some(result) = iter.next_batch() {
            total += result.unwrap().num_rows();
        }
        assert_eq!(total, 4);
    }
}
