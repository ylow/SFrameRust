// Parquet file writer consuming SFrame batches.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{WriterProperties, WriterVersion};

use sframe_query::execute::BatchIterator;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexTypeEnum;

use crate::type_mapping::{sframe_rows_to_record_batch, sframe_type_to_arrow};

// ---------------------------------------------------------------------------
// Task 8: Single-file Parquet writer
// ---------------------------------------------------------------------------

/// Write all batches from `iter` into a single Parquet file at `path`.
///
/// The Arrow schema is built from `column_names` and `column_types`.  Parquet
/// v2 data pages and Snappy compression are used.
pub fn write_parquet(
    mut iter: BatchIterator,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
    path: &Path,
) -> Result<()> {
    if column_names.len() != column_types.len() {
        return Err(SFrameError::Format(format!(
            "column_names length {} != column_types length {}",
            column_names.len(),
            column_types.len()
        )));
    }

    // 1. Build Arrow Schema
    let fields: Vec<Field> = column_names
        .iter()
        .zip(column_types.iter())
        .map(|(name, typ)| Field::new(name, sframe_type_to_arrow(*typ), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    // 2. Writer properties: Parquet v2, Snappy compression
    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::SNAPPY)
        .build();

    // 3. Create ArrowWriter
    let file = File::create(path).map_err(SFrameError::Io)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| SFrameError::Format(format!("Parquet writer error: {}", e)))?;

    // 4. Consume BatchIterator, convert each batch, write
    while let Some(batch_result) = iter.next_batch() {
        let sframe_rows = batch_result?;
        let record_batch = sframe_rows_to_record_batch(&sframe_rows, column_names, column_types)?;
        writer
            .write(&record_batch)
            .map_err(|e| SFrameError::Format(format!("Parquet write error: {}", e)))?;
    }

    // 5. Close writer
    writer
        .close()
        .map_err(|e| SFrameError::Format(format!("Parquet close error: {}", e)))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Task 9: Sharded Parquet writer
// ---------------------------------------------------------------------------

/// Write all batches from `iter` into a sharded Parquet file.
///
/// The output file is named `{prefix}_{shard_index}_of_{total_shards}.parquet`.
/// Delegates to [`write_parquet`].
pub fn write_parquet_shard(
    iter: BatchIterator,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
    prefix: &str,
    shard_index: usize,
    total_shards: usize,
) -> Result<()> {
    let filename = format!("{}_{}_of_{}.parquet", prefix, shard_index, total_shards);
    let path = Path::new(&filename);
    write_parquet(iter, column_names, column_types, path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    use sframe_query::batch::{ColumnData, SFrameRows};
    use sframe_query::execute::{BatchCo, BatchCommand, BatchIterator, BatchResponse};
    use sframe_types::flex_type::{FlexType, FlexTypeEnum};

    use crate::parquet_reader::{count_parquet_rows, read_parquet_batches, read_parquet_schema};

    /// Helper: build a BatchIterator that yields one batch of the given SFrameRows.
    fn single_batch_iter(rows: SFrameRows) -> BatchIterator {
        BatchIterator::new(move |co: BatchCo| async move {
            let cmd = co.yield_(BatchResponse::Ready).await;
            match cmd {
                BatchCommand::NextBatch => {
                    co.yield_(BatchResponse::Batch(Ok(rows))).await;
                }
                _ => {}
            }
        })
    }

    /// Helper: build a BatchIterator that yields multiple batches.
    fn multi_batch_iter(batches: Vec<SFrameRows>) -> BatchIterator {
        BatchIterator::new(move |co: BatchCo| async move {
            let mut cmd = co.yield_(BatchResponse::Ready).await;
            for batch in batches {
                match cmd {
                    BatchCommand::NextBatch => {
                        cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await;
                    }
                    BatchCommand::SkipBatch => {
                        cmd = co.yield_(BatchResponse::Skipped).await;
                    }
                    _ => return,
                }
            }
        })
    }

    /// Helper: build a BatchIterator that yields zero batches.
    fn empty_iter() -> BatchIterator {
        BatchIterator::new(move |co: BatchCo| async move {
            co.yield_(BatchResponse::Ready).await;
        })
    }

    // ===== Task 8: write_parquet =====

    #[test]
    fn test_write_parquet_basic_int_string() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.parquet");

        let rows = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1), Some(2), Some(3)]),
            ColumnData::String(vec![
                Some(Arc::from("alpha")),
                Some(Arc::from("beta")),
                Some(Arc::from("gamma")),
            ]),
        ])
        .unwrap();

        let names = vec!["id".to_string(), "name".to_string()];
        let types = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];

        write_parquet(single_batch_iter(rows), &names, &types, &path).unwrap();

        // Verify by reading back
        let (rnames, rtypes) = read_parquet_schema(path.to_str().unwrap()).unwrap();
        assert_eq!(rnames, names);
        assert_eq!(rtypes, types);

        let count = count_parquet_rows(&[path.clone()]).unwrap();
        assert_eq!(count, 3);

        let mut iter = read_parquet_batches(&[path]).unwrap();
        let batch = iter.next_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);

        // Check values
        assert_eq!(batch.row(0)[0], FlexType::Integer(1));
        assert_eq!(batch.row(1)[0], FlexType::Integer(2));
        assert_eq!(batch.row(2)[0], FlexType::Integer(3));
        assert_eq!(batch.row(0)[1], FlexType::String(Arc::from("alpha")));
        assert_eq!(batch.row(1)[1], FlexType::String(Arc::from("beta")));
        assert_eq!(batch.row(2)[1], FlexType::String(Arc::from("gamma")));

        assert!(iter.next_batch().is_none());
    }

    #[test]
    fn test_write_parquet_float_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("floats.parquet");

        let rows = SFrameRows::new(vec![ColumnData::Float(vec![
            Some(1.5),
            None,
            Some(3.14),
        ])])
        .unwrap();

        let names = vec!["value".to_string()];
        let types = vec![FlexTypeEnum::Float];

        write_parquet(single_batch_iter(rows), &names, &types, &path).unwrap();

        let mut iter = read_parquet_batches(&[path]).unwrap();
        let batch = iter.next_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.row(0), vec![FlexType::Float(1.5)]);
        assert_eq!(batch.row(1), vec![FlexType::Undefined]);
        assert_eq!(batch.row(2), vec![FlexType::Float(3.14)]);
    }

    #[test]
    fn test_write_parquet_empty_batches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.parquet");

        let names = vec!["x".to_string()];
        let types = vec![FlexTypeEnum::Integer];

        write_parquet(empty_iter(), &names, &types, &path).unwrap();

        let count = count_parquet_rows(&[path.clone()]).unwrap();
        assert_eq!(count, 0);

        let (rnames, rtypes) = read_parquet_schema(path.to_str().unwrap()).unwrap();
        assert_eq!(rnames, names);
        assert_eq!(rtypes, types);
    }

    #[test]
    fn test_write_parquet_multiple_batches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.parquet");

        let batch1 = SFrameRows::new(vec![ColumnData::Integer(vec![Some(1), Some(2)])])
            .unwrap();
        let batch2 = SFrameRows::new(vec![ColumnData::Integer(vec![Some(3), Some(4), Some(5)])])
            .unwrap();

        let names = vec!["x".to_string()];
        let types = vec![FlexTypeEnum::Integer];

        write_parquet(multi_batch_iter(vec![batch1, batch2]), &names, &types, &path).unwrap();

        let count = count_parquet_rows(&[path.clone()]).unwrap();
        assert_eq!(count, 5);

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

    #[test]
    fn test_write_parquet_mismatched_names_types() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.parquet");

        let names = vec!["a".to_string(), "b".to_string()];
        let types = vec![FlexTypeEnum::Integer]; // only 1 type, but 2 names

        let result = write_parquet(empty_iter(), &names, &types, &path);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_parquet_with_nulls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nulls.parquet");

        let rows = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(10), None, Some(30)]),
            ColumnData::String(vec![None, Some(Arc::from("hello")), None]),
        ])
        .unwrap();

        let names = vec!["id".to_string(), "name".to_string()];
        let types = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];

        write_parquet(single_batch_iter(rows), &names, &types, &path).unwrap();

        let mut iter = read_parquet_batches(&[path]).unwrap();
        let batch = iter.next_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);

        assert_eq!(batch.row(0)[0], FlexType::Integer(10));
        assert_eq!(batch.row(0)[1], FlexType::Undefined);
        assert_eq!(batch.row(1)[0], FlexType::Undefined);
        assert_eq!(batch.row(1)[1], FlexType::String(Arc::from("hello")));
        assert_eq!(batch.row(2)[0], FlexType::Integer(30));
        assert_eq!(batch.row(2)[1], FlexType::Undefined);
    }

    // ===== Task 9: write_parquet_shard =====

    #[test]
    fn test_write_parquet_shard_naming() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("data").to_str().unwrap().to_string();

        let rows = SFrameRows::new(vec![ColumnData::Integer(vec![Some(42)])])
            .unwrap();

        let names = vec!["val".to_string()];
        let types = vec![FlexTypeEnum::Integer];

        write_parquet_shard(single_batch_iter(rows), &names, &types, &prefix, 0, 3).unwrap();

        // Check the file was created with the expected name
        let expected_path = format!("{}_0_of_3.parquet", prefix);
        assert!(
            Path::new(&expected_path).exists(),
            "Expected file at {}",
            expected_path
        );

        // Read back and verify content
        let mut iter = read_parquet_batches(&[PathBuf::from(&expected_path)]).unwrap();
        let batch = iter.next_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.row(0)[0], FlexType::Integer(42));
    }

    #[test]
    fn test_write_parquet_shard_multiple_shards() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("part").to_str().unwrap().to_string();

        let names = vec!["x".to_string()];
        let types = vec![FlexTypeEnum::Integer];

        // Write shard 0
        let rows0 = SFrameRows::new(vec![ColumnData::Integer(vec![Some(1), Some(2)])])
            .unwrap();
        write_parquet_shard(single_batch_iter(rows0), &names, &types, &prefix, 0, 2).unwrap();

        // Write shard 1
        let rows1 = SFrameRows::new(vec![ColumnData::Integer(vec![Some(3), Some(4)])])
            .unwrap();
        write_parquet_shard(single_batch_iter(rows1), &names, &types, &prefix, 1, 2).unwrap();

        // Both files should exist
        let path0 = PathBuf::from(format!("{}_0_of_2.parquet", prefix));
        let path1 = PathBuf::from(format!("{}_1_of_2.parquet", prefix));
        assert!(path0.exists());
        assert!(path1.exists());

        // Total rows across both
        let count = count_parquet_rows(&[path0.clone(), path1.clone()]).unwrap();
        assert_eq!(count, 4);

        // Read both and verify content
        let mut iter = read_parquet_batches(&[path0, path1]).unwrap();
        let mut all_values = Vec::new();
        while let Some(result) = iter.next_batch() {
            let b = result.unwrap();
            for i in 0..b.num_rows() {
                if let FlexType::Integer(v) = b.row(i)[0] {
                    all_values.push(v);
                }
            }
        }
        assert_eq!(all_values, vec![1, 2, 3, 4]);
    }

    // ===== Roundtrip test: write then read with all column types =====

    #[test]
    fn test_roundtrip_all_basic_types() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("all_types.parquet");

        let rows = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(100), Some(200)]),
            ColumnData::Float(vec![Some(1.1), Some(2.2)]),
            ColumnData::String(vec![Some(Arc::from("foo")), Some(Arc::from("bar"))]),
        ])
        .unwrap();

        let names = vec!["i".to_string(), "f".to_string(), "s".to_string()];
        let types = vec![
            FlexTypeEnum::Integer,
            FlexTypeEnum::Float,
            FlexTypeEnum::String,
        ];

        write_parquet(single_batch_iter(rows), &names, &types, &path).unwrap();

        // Read back schema
        let (rnames, rtypes) = read_parquet_schema(path.to_str().unwrap()).unwrap();
        assert_eq!(rnames, names);
        assert_eq!(rtypes, types);

        // Read back data
        let mut iter = read_parquet_batches(&[path]).unwrap();
        let batch = iter.next_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);

        assert_eq!(batch.row(0)[0], FlexType::Integer(100));
        assert_eq!(batch.row(0)[1], FlexType::Float(1.1));
        assert_eq!(batch.row(0)[2], FlexType::String(Arc::from("foo")));
        assert_eq!(batch.row(1)[0], FlexType::Integer(200));
        assert_eq!(batch.row(1)[1], FlexType::Float(2.2));
        assert_eq!(batch.row(1)[2], FlexType::String(Arc::from("bar")));
    }
}
