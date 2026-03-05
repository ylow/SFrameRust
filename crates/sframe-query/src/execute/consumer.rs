//! Stream consumers that drive a BatchIterator to a destination.

use std::io::Write;

use sframe_storage::segment_writer::SegmentWriter;
use sframe_types::error::Result;
use sframe_types::flex_type::FlexTypeEnum;

use super::batch_iter::BatchIterator;

/// Drive a `BatchIterator` and write all output into a single segment file.
///
/// Each batch's columns are written as blocks into the `SegmentWriter`.
/// Blocks from different columns are interleaved in the file (all columns
/// for batch 0, then all columns for batch 1, etc.), which is correct
/// because `SegmentReader` uses the block index footer for seeking.
///
/// Returns `(segment_sizes, total_rows)` where `segment_sizes` is the
/// per-column element count vector from `SegmentWriter::finish()`.
pub fn consume_to_segment<W: Write>(
    iter: &mut BatchIterator,
    mut seg_writer: SegmentWriter<W>,
    dtypes: &[FlexTypeEnum],
) -> Result<(Vec<u64>, u64)> {
    let mut total_rows: u64 = 0;

    while let Some(batch_result) = iter.next_batch() {
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

    let segment_sizes = seg_writer.finish()?;
    Ok((segment_sizes, total_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{ColumnData, SFrameRows};
    use crate::execute::batch_iter::{BatchCo, BatchResponse};
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

        let batches = vec![batch1, batch2];
        let mut input = BatchIterator::new(move |co: BatchCo| async move {
            let mut cmd = co.yield_(BatchResponse::Ready).await;
            for batch in batches {
                match cmd {
                    crate::execute::BatchCommand::NextBatch => {
                        cmd = co.yield_(BatchResponse::Batch(Ok(batch))).await;
                    }
                    crate::execute::BatchCommand::SkipBatch => {
                        cmd = co.yield_(BatchResponse::Skipped).await;
                    }
                    _ => break,
                }
            }
        });

        // Write to a temp file
        let dir = tempfile::tempdir().unwrap();
        let seg_path = dir.path().join("test.segment");
        let file = std::fs::File::create(&seg_path).unwrap();
        let seg_writer = SegmentWriter::new(std::io::BufWriter::new(file), 2);
        let dtypes = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];

        let (segment_sizes, total_rows) =
            consume_to_segment(&mut input, seg_writer, &dtypes).unwrap();

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
        let mut input = BatchIterator::new(|co: BatchCo| async move {
            co.yield_(BatchResponse::Ready).await;
            // No batches — return immediately.
        });

        let dir = tempfile::tempdir().unwrap();
        let seg_path = dir.path().join("empty.segment");
        let file = std::fs::File::create(&seg_path).unwrap();
        let seg_writer = SegmentWriter::new(std::io::BufWriter::new(file), 1);
        let dtypes = vec![FlexTypeEnum::Integer];

        let (segment_sizes, total_rows) =
            consume_to_segment(&mut input, seg_writer, &dtypes).unwrap();

        assert_eq!(total_rows, 0);
        assert_eq!(segment_sizes[0], 0);
    }
}
