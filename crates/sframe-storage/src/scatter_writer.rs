//! ScatterWriter: writes column values to M segment files by (column, segment) pair.
//!
//! Used by EC Sort to scatter column values into buckets. Values for any
//! column can be written at any time — blocks in segment files may be
//! interleaved across columns, and SegmentReader handles this via the
//! offset field in BlockInfo.

use sframe_io::vfs::{VirtualFileSystem, WritableFile};
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::segment_writer::{
    estimate_bytes_per_value, SegmentWriter, MAX_ROWS_PER_BLOCK, MIN_ROWS_PER_BLOCK,
    TARGET_BLOCK_SIZE,
};
use crate::sframe_writer::segment_filename;

/// Result returned by `ScatterWriter::finish()`.
pub struct ScatterResult {
    /// Segment file names (relative to base_path).
    pub segment_files: Vec<String>,
    /// Per-segment, per-column row counts: `all_segment_sizes[seg][col]`.
    pub all_segment_sizes: Vec<Vec<u64>>,
}

/// Writes column values to M segment files, supporting scatter by (column, segment).
///
/// Values for any column can be written in any order. Per-(column, segment)
/// buffers are flushed as blocks when they reach the target block size.
/// Blocks from different columns may be interleaved in segment files;
/// SegmentReader handles this via offset-based seeking.
pub struct ScatterWriter {
    segment_writers: Vec<SegmentWriter<Box<dyn WritableFile>>>,
    segment_files: Vec<String>,
    column_types: Vec<FlexTypeEnum>,
    num_columns: usize,
    num_segments: usize,
    /// Per-column, per-segment buffers: `buffers[col][seg]`.
    buffers: Vec<Vec<Vec<FlexType>>>,
    /// Per-column target block size (rows).
    rows_per_block: Vec<usize>,
    /// Per-segment, per-column row counts: `column_counts[seg][col]`.
    column_counts: Vec<Vec<u64>>,
}

impl ScatterWriter {
    /// Create a new ScatterWriter.
    ///
    /// Creates `num_segments` segment files at `{base_path}/{data_prefix}.NNNN`.
    pub fn new(
        vfs: &dyn VirtualFileSystem,
        base_path: &str,
        data_prefix: &str,
        column_types: &[FlexTypeEnum],
        num_segments: usize,
    ) -> Result<Self> {
        let num_columns = column_types.len();
        let mut segment_writers = Vec::with_capacity(num_segments);
        let mut segment_files = Vec::with_capacity(num_segments);

        for seg_idx in 0..num_segments {
            let seg_file = segment_filename(data_prefix, seg_idx);
            let seg_path = format!("{base_path}/{seg_file}");
            let file = vfs.open_write(&seg_path)?;
            segment_writers.push(SegmentWriter::new(file, num_columns));
            segment_files.push(seg_file);
        }

        let column_counts = vec![vec![0u64; num_columns]; num_segments];

        let rows_per_block: Vec<usize> = column_types
            .iter()
            .map(|&dt| {
                let est = estimate_bytes_per_value(dt).max(1);
                (TARGET_BLOCK_SIZE / est).clamp(MIN_ROWS_PER_BLOCK, MAX_ROWS_PER_BLOCK)
            })
            .collect();

        let buffers = (0..num_columns)
            .map(|_| (0..num_segments).map(|_| Vec::new()).collect())
            .collect();

        Ok(ScatterWriter {
            segment_writers,
            segment_files,
            column_types: column_types.to_vec(),
            num_columns,
            num_segments,
            buffers,
            rows_per_block,
            column_counts,
        })
    }

    /// Write a value to a specific (column, segment) pair.
    ///
    /// Values for any column can be written in any order.
    pub fn write_to_segment(
        &mut self,
        column_id: usize,
        segment_id: usize,
        value: FlexType,
    ) -> Result<()> {
        if segment_id >= self.num_segments {
            return Err(SFrameError::Format(format!(
                "ScatterWriter: segment_id {segment_id} out of range ({})",
                self.num_segments
            )));
        }

        self.buffers[column_id][segment_id].push(value);

        if self.buffers[column_id][segment_id].len() >= self.rows_per_block[column_id] {
            self.flush_buffer(column_id, segment_id)?;
        }

        Ok(())
    }

    /// Flush all remaining buffers for a specific column across all segments.
    pub fn flush_column(&mut self, column_id: usize) -> Result<()> {
        for seg_id in 0..self.num_segments {
            self.flush_buffer(column_id, seg_id)?;
        }
        Ok(())
    }

    /// Finalize: flush all buffers, finish all segment writers, return results.
    pub fn finish(mut self) -> Result<ScatterResult> {
        for col_id in 0..self.num_columns {
            for seg_id in 0..self.num_segments {
                self.flush_buffer(col_id, seg_id)?;
            }
        }

        let mut all_segment_sizes = Vec::with_capacity(self.num_segments);
        for seg_writer in self.segment_writers {
            let segment_sizes = seg_writer.finish()?;
            all_segment_sizes.push(segment_sizes);
        }

        Ok(ScatterResult {
            segment_files: self.segment_files,
            all_segment_sizes,
        })
    }

    /// Flush one (column, segment) buffer.
    fn flush_buffer(&mut self, column_id: usize, segment_id: usize) -> Result<()> {
        let values: Vec<FlexType> = self.buffers[column_id][segment_id].drain(..).collect();
        if values.is_empty() {
            return Ok(());
        }
        let count = values.len() as u64;
        self.segment_writers[segment_id].write_column_block(
            column_id,
            &values,
            self.column_types[column_id],
        )?;
        self.column_counts[segment_id][column_id] += count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_reader::SegmentReader;
    use sframe_io::local_fs::LocalFileSystem;
    use std::sync::Arc;

    #[test]
    fn test_scatter_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().to_str().unwrap();
        let vfs = LocalFileSystem;
        let column_types = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let num_segments = 3;
        let data_prefix = "scatter_test";

        let mut sw = ScatterWriter::new(
            &vfs,
            base_path,
            data_prefix,
            &column_types,
            num_segments,
        )
        .unwrap();

        // Write 9 integer values with round-robin distribution across 3 segments
        for i in 0..9i64 {
            let seg = (i % 3) as usize;
            sw.write_to_segment(0, seg, FlexType::Integer(i)).unwrap();
        }

        // Write 9 string values with round-robin distribution
        for i in 0..9i64 {
            let seg = (i % 3) as usize;
            sw.write_to_segment(1, seg, FlexType::String(Arc::from(format!("val_{i}"))))
                .unwrap();
        }

        let result = sw.finish().unwrap();

        assert_eq!(result.segment_files.len(), 3);
        assert_eq!(result.all_segment_sizes.len(), 3);

        for seg_sizes in &result.all_segment_sizes {
            assert_eq!(seg_sizes[0], 3, "Expected 3 integers per segment");
            assert_eq!(seg_sizes[1], 3, "Expected 3 strings per segment");
        }

        for seg_idx in 0..num_segments {
            let seg_path = format!("{}/{}", base_path, result.segment_files[seg_idx]);
            let file = vfs.open_read(&seg_path).unwrap();
            let file_size = file.size().unwrap();
            let mut reader =
                SegmentReader::open(Box::new(file), file_size, column_types.to_vec()).unwrap();

            let ints = reader.read_column(0).unwrap();
            assert_eq!(ints.len(), 3);
            for (j, val) in ints.iter().enumerate() {
                let expected = (seg_idx + j * 3) as i64;
                assert_eq!(*val, FlexType::Integer(expected));
            }

            let strs = reader.read_column(1).unwrap();
            assert_eq!(strs.len(), 3);
            for (j, val) in strs.iter().enumerate() {
                let expected_i = seg_idx + j * 3;
                let expected = FlexType::String(Arc::from(format!("val_{expected_i}")));
                assert_eq!(*val, expected);
            }
        }
    }

    #[test]
    fn test_scatter_interleaved_columns() {
        // Write columns in interleaved order (col0, col1, col0, col1, ...)
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().to_str().unwrap();
        let vfs = LocalFileSystem;
        let column_types = [FlexTypeEnum::Integer, FlexTypeEnum::String];

        let mut sw =
            ScatterWriter::new(&vfs, base_path, "scatter_interleave", &column_types, 2).unwrap();

        // Interleave writes: for each "row", write both columns
        for i in 0..6i64 {
            let seg = (i % 2) as usize;
            sw.write_to_segment(0, seg, FlexType::Integer(i)).unwrap();
            sw.write_to_segment(1, seg, FlexType::String(Arc::from(format!("s{i}"))))
                .unwrap();
        }

        let result = sw.finish().unwrap();

        // Each segment should have 3 rows per column
        assert_eq!(result.all_segment_sizes[0][0], 3);
        assert_eq!(result.all_segment_sizes[0][1], 3);

        // Verify readback
        let seg_path = format!("{}/{}", base_path, result.segment_files[0]);
        let file = vfs.open_read(&seg_path).unwrap();
        let file_size = file.size().unwrap();
        let mut reader =
            SegmentReader::open(Box::new(file), file_size, column_types.to_vec()).unwrap();

        let ints = reader.read_column(0).unwrap();
        assert_eq!(ints, vec![FlexType::Integer(0), FlexType::Integer(2), FlexType::Integer(4)]);

        let strs = reader.read_column(1).unwrap();
        assert_eq!(strs, vec![
            FlexType::String(Arc::from("s0")),
            FlexType::String(Arc::from("s2")),
            FlexType::String(Arc::from("s4")),
        ]);
    }

    #[test]
    fn test_scatter_empty_segments() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().to_str().unwrap();
        let vfs = LocalFileSystem;
        let column_types = [FlexTypeEnum::Integer];

        let mut sw =
            ScatterWriter::new(&vfs, base_path, "scatter_empty", &column_types, 3).unwrap();

        for i in 0..10i64 {
            sw.write_to_segment(0, 1, FlexType::Integer(i)).unwrap();
        }

        let result = sw.finish().unwrap();

        assert_eq!(result.all_segment_sizes[0][0], 0);
        assert_eq!(result.all_segment_sizes[1][0], 10);
        assert_eq!(result.all_segment_sizes[2][0], 0);
    }
}
