//! Stream consumers that drive a BatchStream to a destination.

use std::io::Write;

use futures::StreamExt;

use sframe_storage::segment_writer::SegmentWriter;
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::FlexTypeEnum;

use super::BatchStream;

/// Drive a `BatchStream` and write all output into a single segment file.
///
/// Each batch's columns are written as blocks into the `SegmentWriter`.
/// Blocks from different columns are interleaved in the file (all columns
/// for batch 0, then all columns for batch 1, etc.), which is correct
/// because `SegmentReader` uses the block index footer for seeking.
///
/// Returns `(segment_sizes, total_rows)` where `segment_sizes` is the
/// per-column element count vector from `SegmentWriter::finish()`.
pub fn consume_to_segment<W: Write>(
    stream: BatchStream,
    mut seg_writer: SegmentWriter<W>,
    dtypes: &[FlexTypeEnum],
) -> Result<(Vec<u64>, u64)> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| SFrameError::Format(format!("Failed to create tokio runtime: {}", e)))?;

    let mut total_rows: u64 = 0;

    rt.block_on(async {
        let mut stream = stream;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let n = batch.num_rows();
            if n == 0 {
                continue;
            }
            for (col_idx, col) in batch.columns().iter().enumerate() {
                let values = col.to_flex_vec();
                seg_writer.write_column_block(col_idx, &values, dtypes[col_idx])?;
            }
            total_rows += n as u64;
        }
        Ok::<(), SFrameError>(())
    })?;

    let segment_sizes = seg_writer.finish()?;
    Ok((segment_sizes, total_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{ColumnData, SFrameRows};
    use futures::stream;
    use sframe_storage::segment_reader::SegmentReader;
    use sframe_types::flex_type::FlexType;

    #[test]
    fn test_consume_to_segment_basic() {
        // Create two batches
        let batch1 = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(1), Some(2)]),
            ColumnData::String(vec![Some("a".into()), Some("b".into())]),
        ])
        .unwrap();
        let batch2 = SFrameRows::new(vec![
            ColumnData::Integer(vec![Some(3)]),
            ColumnData::String(vec![Some("c".into())]),
        ])
        .unwrap();

        let input: BatchStream = Box::pin(stream::iter(vec![Ok(batch1), Ok(batch2)]));

        // Write to a temp file
        let dir = tempfile::tempdir().unwrap();
        let seg_path = dir.path().join("test.segment");
        let file = std::fs::File::create(&seg_path).unwrap();
        let seg_writer = SegmentWriter::new(std::io::BufWriter::new(file), 2);
        let dtypes = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];

        let (segment_sizes, total_rows) =
            consume_to_segment(input, seg_writer, &dtypes).unwrap();

        assert_eq!(total_rows, 3);
        assert_eq!(segment_sizes[0], 3);
        assert_eq!(segment_sizes[1], 3);

        // Read back with SegmentReader and verify
        let file = std::fs::File::open(&seg_path).unwrap();
        let file_size = file.metadata().unwrap().len();
        let mut reader =
            SegmentReader::open(Box::new(file), file_size, dtypes).unwrap();

        let col0 = reader.read_column(0).unwrap();
        assert_eq!(
            col0,
            vec![
                FlexType::Integer(1),
                FlexType::Integer(2),
                FlexType::Integer(3),
            ]
        );
        let col1 = reader.read_column(1).unwrap();
        assert_eq!(
            col1,
            vec![
                FlexType::String("a".into()),
                FlexType::String("b".into()),
                FlexType::String("c".into()),
            ]
        );
    }

    #[test]
    fn test_consume_to_segment_empty_stream() {
        let input: BatchStream = Box::pin(stream::empty());

        let dir = tempfile::tempdir().unwrap();
        let seg_path = dir.path().join("empty.segment");
        let file = std::fs::File::create(&seg_path).unwrap();
        let seg_writer = SegmentWriter::new(std::io::BufWriter::new(file), 1);
        let dtypes = vec![FlexTypeEnum::Integer];

        let (segment_sizes, total_rows) =
            consume_to_segment(input, seg_writer, &dtypes).unwrap();

        assert_eq!(total_rows, 0);
        assert_eq!(segment_sizes[0], 0);
    }
}
