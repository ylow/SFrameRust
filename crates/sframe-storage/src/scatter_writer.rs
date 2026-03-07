//! ScatterWriter: writes column values to M segment files by (column, segment) pair.
//!
//! Used by EC Sort to scatter column values into buckets. Columns must be
//! written in sequential order so that blocks appear in column order within
//! each segment file.

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
/// Columns must be written in strictly increasing order. Within each column,
/// values can be written to any segment in any order. Buffering and adaptive
/// block sizing are handled per-segment for the active column.
pub struct ScatterWriter {
    segment_writers: Vec<SegmentWriter<Box<dyn WritableFile>>>,
    segment_files: Vec<String>,
    column_types: Vec<FlexTypeEnum>,
    num_segments: usize,
    /// Per-segment buffer for the active column.
    buffers: Vec<Vec<FlexType>>,
    /// Adaptive block size (rows) for the active column.
    rows_per_block: usize,
    /// Type of the active column.
    active_dtype: FlexTypeEnum,
    /// Index of the active column, or None if no column is active.
    active_column: Option<usize>,
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
        let mut segment_writers = Vec::with_capacity(num_segments);
        let mut segment_files = Vec::with_capacity(num_segments);

        for seg_idx in 0..num_segments {
            let seg_file = segment_filename(data_prefix, seg_idx);
            let seg_path = format!("{base_path}/{seg_file}");
            let file = vfs.open_write(&seg_path)?;
            segment_writers.push(SegmentWriter::new(file, column_types.len()));
            segment_files.push(seg_file);
        }

        let column_counts = vec![vec![0u64; column_types.len()]; num_segments];

        Ok(ScatterWriter {
            segment_writers,
            segment_files,
            column_types: column_types.to_vec(),
            num_segments,
            buffers: vec![Vec::new(); num_segments],
            rows_per_block: MIN_ROWS_PER_BLOCK,
            active_dtype: FlexTypeEnum::Undefined,
            active_column: None,
            column_counts,
        })
    }

    /// Begin writing a new column. Validates ordering and flushes the previous column.
    fn begin_column(&mut self, column_id: usize) -> Result<()> {
        // Validate column order: must be strictly greater than previous
        if let Some(prev) = self.active_column {
            if column_id <= prev {
                return Err(SFrameError::Format(format!(
                    "ScatterWriter: column {column_id} is not after previous column {prev}; \
                     columns must be written in strictly increasing order"
                )));
            }
            // Flush the previous column
            self.flush_active_column()?;
        }

        // Set up for the new column
        let dtype = self.column_types[column_id];
        let est = estimate_bytes_per_value(dtype).max(1);
        let rows_per_block = (TARGET_BLOCK_SIZE / est).clamp(MIN_ROWS_PER_BLOCK, MAX_ROWS_PER_BLOCK);

        self.active_column = Some(column_id);
        self.active_dtype = dtype;
        self.rows_per_block = rows_per_block;

        // Clear buffers (should already be empty after flush, but be safe)
        for buf in &mut self.buffers {
            buf.clear();
        }

        Ok(())
    }

    /// Write a value to a specific segment for the given column.
    ///
    /// Auto-begins the column if it is not yet active. Columns must be written
    /// in strictly increasing order.
    pub fn write_to_segment(
        &mut self,
        column_id: usize,
        segment_id: usize,
        value: FlexType,
    ) -> Result<()> {
        // Auto-begin column if needed
        match self.active_column {
            None => self.begin_column(column_id)?,
            Some(active) if active != column_id => self.begin_column(column_id)?,
            _ => {}
        }

        if segment_id >= self.num_segments {
            return Err(SFrameError::Format(format!(
                "ScatterWriter: segment_id {segment_id} out of range ({})",
                self.num_segments
            )));
        }

        self.buffers[segment_id].push(value);

        // Flush if buffer reaches rows_per_block
        if self.buffers[segment_id].len() >= self.rows_per_block {
            self.flush_segment_buffer(column_id, segment_id)?;
        }

        Ok(())
    }

    /// Flush all remaining buffers for the active column and deactivate it.
    pub fn flush_column(&mut self, column_id: usize) -> Result<()> {
        match self.active_column {
            Some(active) if active == column_id => {
                self.flush_active_column()?;
                self.active_column = None;
                Ok(())
            }
            Some(active) => Err(SFrameError::Format(format!(
                "ScatterWriter::flush_column({column_id}) but active column is {active}"
            ))),
            None => {
                // No active column; nothing to flush. This is not an error.
                Ok(())
            }
        }
    }

    /// Finalize: flush the active column, finish all segment writers, return results.
    pub fn finish(mut self) -> Result<ScatterResult> {
        // Flush active column if any
        if self.active_column.is_some() {
            self.flush_active_column()?;
        }

        // Finish all segment writers and collect segment sizes
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

    /// Flush one segment's buffer for the active column.
    fn flush_segment_buffer(&mut self, column_id: usize, segment_id: usize) -> Result<()> {
        let values: Vec<FlexType> = self.buffers[segment_id].drain(..).collect();
        if values.is_empty() {
            return Ok(());
        }
        let count = values.len() as u64;
        self.segment_writers[segment_id].write_column_block(
            column_id,
            &values,
            self.active_dtype,
        )?;
        self.column_counts[segment_id][column_id] += count;
        Ok(())
    }

    /// Flush all segment buffers for the active column.
    fn flush_active_column(&mut self) -> Result<()> {
        let column_id = match self.active_column {
            Some(id) => id,
            None => return Ok(()),
        };

        for seg_id in 0..self.num_segments {
            self.flush_segment_buffer(column_id, seg_id)?;
        }

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

        // Create ScatterWriter
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

        // Verify segment files
        assert_eq!(result.segment_files.len(), 3);
        assert_eq!(result.all_segment_sizes.len(), 3);

        // Each segment should have 3 values per column
        for seg_sizes in &result.all_segment_sizes {
            assert_eq!(seg_sizes[0], 3, "Expected 3 integers per segment");
            assert_eq!(seg_sizes[1], 3, "Expected 3 strings per segment");
        }

        // Read back and verify with SegmentReader
        for seg_idx in 0..num_segments {
            let seg_path = format!(
                "{}/{}",
                base_path, result.segment_files[seg_idx]
            );
            let file = vfs.open_read(&seg_path).unwrap();
            let file_size = file.size().unwrap();
            let mut reader =
                SegmentReader::open(Box::new(file), file_size, column_types.to_vec()).unwrap();

            // Check integers
            let ints = reader.read_column(0).unwrap();
            assert_eq!(ints.len(), 3);
            for (j, val) in ints.iter().enumerate() {
                let expected = (seg_idx + j * 3) as i64;
                assert_eq!(
                    *val,
                    FlexType::Integer(expected),
                    "Segment {seg_idx}, int index {j}: expected {expected}"
                );
            }

            // Check strings
            let strs = reader.read_column(1).unwrap();
            assert_eq!(strs.len(), 3);
            for (j, val) in strs.iter().enumerate() {
                let expected_i = seg_idx + j * 3;
                let expected = FlexType::String(Arc::from(format!("val_{expected_i}")));
                assert_eq!(
                    *val, expected,
                    "Segment {seg_idx}, string index {j}"
                );
            }
        }
    }

    #[test]
    fn test_scatter_empty_segments() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().to_str().unwrap();
        let vfs = LocalFileSystem;
        let column_types = [FlexTypeEnum::Integer];
        let num_segments = 3;
        let data_prefix = "scatter_empty";

        let mut sw = ScatterWriter::new(
            &vfs,
            base_path,
            data_prefix,
            &column_types,
            num_segments,
        )
        .unwrap();

        // Write all values to segment 1 only
        for i in 0..10i64 {
            sw.write_to_segment(0, 1, FlexType::Integer(i)).unwrap();
        }

        let result = sw.finish().unwrap();

        // Segment 0 and 2 should have 0 rows
        assert_eq!(result.all_segment_sizes[0][0], 0);
        assert_eq!(result.all_segment_sizes[1][0], 10);
        assert_eq!(result.all_segment_sizes[2][0], 0);

        // Verify segment 1 has the right values
        let seg_path = format!("{}/{}", base_path, result.segment_files[1]);
        let file = vfs.open_read(&seg_path).unwrap();
        let file_size = file.size().unwrap();
        let mut reader =
            SegmentReader::open(Box::new(file), file_size, column_types.to_vec()).unwrap();

        let ints = reader.read_column(0).unwrap();
        assert_eq!(ints.len(), 10);
        for (i, val) in ints.iter().enumerate() {
            assert_eq!(*val, FlexType::Integer(i as i64));
        }
    }

    #[test]
    fn test_scatter_column_order_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().to_str().unwrap();
        let vfs = LocalFileSystem;
        let column_types = [FlexTypeEnum::Integer, FlexTypeEnum::String];
        let num_segments = 2;
        let data_prefix = "scatter_order";

        let mut sw = ScatterWriter::new(
            &vfs,
            base_path,
            data_prefix,
            &column_types,
            num_segments,
        )
        .unwrap();

        // Write to column 1 first
        sw.write_to_segment(1, 0, FlexType::String(Arc::from("hello")))
            .unwrap();

        // Now try to write to column 0 — should fail
        let result = sw.write_to_segment(0, 0, FlexType::Integer(42));
        assert!(
            result.is_err(),
            "Writing column 0 after column 1 should fail"
        );

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("not after previous column"),
            "Error should mention column ordering: {err_msg}"
        );
    }
}
