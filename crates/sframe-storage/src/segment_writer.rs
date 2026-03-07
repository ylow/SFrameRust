//! Writes columns to a V2 segment file.
//!
//! A segment file contains blocks for multiple columns. Each block is
//! optionally LZ4-compressed and padded to 4K alignment. The footer
//! stores block metadata for seeking during reads.

use std::io::Write;

use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::block_encode::encode_typed_block;
use crate::block_info::{
    write_block_index, BlockInfo, BLOCK_ENCODING_EXTENSION, IS_FLEXIBLE_TYPE, LZ4_COMPRESSION,
};

/// LZ4 compression is skipped if compressed size >= 90% of original.
const COMPRESSION_DISABLE_THRESHOLD: f64 = 0.9;

/// 4K alignment for blocks.
const BLOCK_ALIGNMENT: u64 = 4096;

/// Writer for a single V2 segment file.
pub struct SegmentWriter<W: Write> {
    writer: W,
    block_index: Vec<Vec<BlockInfo>>,
    num_columns: usize,
    bytes_written: u64,
}

impl<W: Write> SegmentWriter<W> {
    /// Create a new segment writer for the given number of columns.
    pub fn new(writer: W, num_columns: usize) -> Self {
        SegmentWriter {
            writer,
            block_index: vec![Vec::new(); num_columns],
            num_columns,
            bytes_written: 0,
        }
    }

    /// Write a block of values for a specific column.
    ///
    /// Returns the on-disk (post-compression, pre-padding) byte count,
    /// which callers can use for online block-size estimation.
    pub fn write_column_block(
        &mut self,
        column: usize,
        values: &[FlexType],
        dtype: FlexTypeEnum,
    ) -> Result<u64> {
        if column >= self.num_columns {
            return Err(sframe_types::error::SFrameError::Format(format!(
                "Column index {} out of range ({})",
                column, self.num_columns
            )));
        }

        // Encode the block
        let encoded = encode_typed_block(values)?;
        let block_size = encoded.len() as u64;

        // Try LZ4 compression
        let compressed = lz4_flex::compress(&encoded);

        let use_compression =
            (compressed.len() as f64) < COMPRESSION_DISABLE_THRESHOLD * (block_size as f64);
        let (data_to_write, flags, on_disk_len) = if use_compression {
            let len = compressed.len() as u64;
            (compressed, LZ4_COMPRESSION | IS_FLEXIBLE_TYPE | BLOCK_ENCODING_EXTENSION, len)
        } else {
            (encoded, IS_FLEXIBLE_TYPE | BLOCK_ENCODING_EXTENSION, block_size)
        };

        let offset = self.bytes_written;

        // Write block data
        self.writer.write_all(&data_to_write)?;
        self.bytes_written += on_disk_len;

        // 4K alignment padding
        let padded = self.bytes_written.div_ceil(BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT;
        let padding = padded - self.bytes_written;
        if padding > 0 {
            let zeros = vec![0u8; padding as usize];
            self.writer.write_all(&zeros)?;
            self.bytes_written = padded;
        }

        // Record block info
        self.block_index[column].push(BlockInfo {
            offset,
            length: on_disk_len,
            block_size,
            num_elem: values.len() as u64,
            flags,
            content_type: dtype as u16,
        });

        Ok(on_disk_len)
    }

    /// Write a pre-encoded (and possibly compressed) block for a column.
    ///
    /// Used by parallel encoding: the CPU-intensive encode+compress work is
    /// done in parallel, then results are written sequentially via this method.
    pub fn write_pre_encoded_block(
        &mut self,
        column: usize,
        data: &[u8],
        uncompressed_size: u64,
        num_elem: u64,
        is_compressed: bool,
        dtype: FlexTypeEnum,
    ) -> Result<u64> {
        let on_disk_len = data.len() as u64;
        let flags = if is_compressed {
            LZ4_COMPRESSION | IS_FLEXIBLE_TYPE | BLOCK_ENCODING_EXTENSION
        } else {
            IS_FLEXIBLE_TYPE | BLOCK_ENCODING_EXTENSION
        };

        let offset = self.bytes_written;

        self.writer.write_all(data)?;
        self.bytes_written += on_disk_len;

        // 4K alignment padding
        let padded = self.bytes_written.div_ceil(BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT;
        let padding = padded - self.bytes_written;
        if padding > 0 {
            let zeros = vec![0u8; padding as usize];
            self.writer.write_all(&zeros)?;
            self.bytes_written = padded;
        }

        self.block_index[column].push(BlockInfo {
            offset,
            length: on_disk_len,
            block_size: uncompressed_size,
            num_elem,
            flags,
            content_type: dtype as u16,
        });

        Ok(on_disk_len)
    }

    /// Finalize the segment: write the footer and return column element counts.
    ///
    /// Returns the per-column element counts (segment_sizes).
    pub fn finish(mut self) -> Result<Vec<u64>> {
        // Write footer (serialized block index)
        let footer_size = write_block_index(&mut self.writer, &self.block_index)?;

        // Write footer_size as last 8 bytes
        self.writer.write_all(&footer_size.to_le_bytes())?;

        // Compute per-column element counts
        let segment_sizes: Vec<u64> = self
            .block_index
            .iter()
            .map(|blocks| blocks.iter().map(|b| b.num_elem).sum())
            .collect();

        Ok(segment_sizes)
    }
}

// ============================================================================
// BufferedSegmentWriter — adaptive block sizing
// ============================================================================

/// Target on-disk block size in bytes (64 KiB).
pub(crate) const TARGET_BLOCK_SIZE: usize = 64 * 1024;

/// Minimum rows per block (lower bound for adaptive sizing).
pub(crate) const MIN_ROWS_PER_BLOCK: usize = 8;

/// Maximum rows per block (upper bound for adaptive sizing).
pub(crate) const MAX_ROWS_PER_BLOCK: usize = 256 * 1024;

/// Rough estimate of bytes per value for initial block sizing.
pub(crate) fn estimate_bytes_per_value(dtype: FlexTypeEnum) -> usize {
    match dtype {
        FlexTypeEnum::Integer => 8,
        FlexTypeEnum::Float => 8,
        FlexTypeEnum::String => 32,
        FlexTypeEnum::Vector => 64,
        FlexTypeEnum::List => 64,
        FlexTypeEnum::Dict => 64,
        FlexTypeEnum::DateTime => 12,
        FlexTypeEnum::Undefined => 1,
    }
}

/// A buffered wrapper around `SegmentWriter` that coalesces small writes
/// into blocks of approximately `TARGET_BLOCK_SIZE` bytes.
///
/// The block size is adaptive: after each flush the writer updates its
/// bytes-per-value estimate using a cumulative average and recalculates
/// the number of rows to buffer before the next flush.
pub struct BufferedSegmentWriter<W: Write> {
    inner: SegmentWriter<W>,
    dtypes: Vec<FlexTypeEnum>,
    num_columns: usize,
    /// Per-column value buffer, filled by `write_column_block`.
    buffers: Vec<Vec<FlexType>>,
    /// Per-column adaptive block size (number of rows).
    rows_per_block: Vec<usize>,
    /// Per-column cumulative encoded bytes (for adaptive estimate).
    encoded_bytes: Vec<u64>,
    /// Per-column cumulative encoded values (for adaptive estimate).
    encoded_values: Vec<u64>,
}

impl<W: Write> BufferedSegmentWriter<W> {
    /// Create a new `BufferedSegmentWriter`.
    ///
    /// `dtypes` determines the number of columns and their types.
    /// Initial rows-per-block is estimated from `estimate_bytes_per_value`.
    pub fn new(writer: W, dtypes: &[FlexTypeEnum]) -> Self {
        let num_columns = dtypes.len();
        let rows_per_block: Vec<usize> = dtypes
            .iter()
            .map(|&dt| {
                let est = estimate_bytes_per_value(dt).max(1);
                (TARGET_BLOCK_SIZE / est).clamp(MIN_ROWS_PER_BLOCK, MAX_ROWS_PER_BLOCK)
            })
            .collect();

        BufferedSegmentWriter {
            inner: SegmentWriter::new(writer, num_columns),
            dtypes: dtypes.to_vec(),
            num_columns,
            buffers: vec![Vec::new(); num_columns],
            rows_per_block,
            encoded_bytes: vec![0u64; num_columns],
            encoded_values: vec![0u64; num_columns],
        }
    }

    /// Append values for a column, flushing full blocks as needed.
    ///
    /// Returns the total on-disk bytes written during any flushes triggered
    /// by this call.
    pub fn write_column_block(
        &mut self,
        column: usize,
        values: &[FlexType],
        dtype: FlexTypeEnum,
    ) -> Result<u64> {
        self.buffers[column].extend_from_slice(values);

        let mut total_bytes = 0u64;

        while self.buffers[column].len() >= self.rows_per_block[column] {
            let rpb = self.rows_per_block[column];
            // Drain one full block from the front.
            let block: Vec<FlexType> = self.buffers[column].drain(..rpb).collect();
            let num_values = block.len() as u64;
            let on_disk = self.inner.write_column_block(column, &block, dtype)?;
            total_bytes += on_disk;
            self.update_block_size(column, num_values, on_disk);
        }

        Ok(total_bytes)
    }

    /// Flush all remaining buffered data and finalize the segment.
    ///
    /// Returns per-column element counts (same as `SegmentWriter::finish`).
    pub fn finish(mut self) -> Result<Vec<u64>> {
        for col in 0..self.num_columns {
            if !self.buffers[col].is_empty() {
                let dtype = self.dtypes[col];
                let remaining: Vec<FlexType> = std::mem::take(&mut self.buffers[col]);
                self.inner.write_column_block(col, &remaining, dtype)?;
            }
        }
        self.inner.finish()
    }

    /// Update the adaptive block size for a column after flushing.
    fn update_block_size(&mut self, col: usize, num_values: u64, on_disk_bytes: u64) {
        self.encoded_bytes[col] += on_disk_bytes;
        self.encoded_values[col] += num_values;

        if self.encoded_values[col] > 0 {
            let avg_bytes_per_value =
                (self.encoded_bytes[col] as f64) / (self.encoded_values[col] as f64);
            let est = avg_bytes_per_value.max(1.0);
            let new_rpb = ((TARGET_BLOCK_SIZE as f64) / est) as usize;
            self.rows_per_block[col] = new_rpb.clamp(MIN_ROWS_PER_BLOCK, MAX_ROWS_PER_BLOCK);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_reader::SegmentReader;
    use std::sync::Arc;

    /// Helper: write to a Vec<u8>, then read back with SegmentReader.
    fn make_reader(buf: Vec<u8>, dtypes: &[FlexTypeEnum]) -> SegmentReader {
        let len = buf.len() as u64;
        let cursor = std::io::Cursor::new(buf);
        SegmentReader::open(Box::new(cursor), len, dtypes.to_vec()).unwrap()
    }

    #[test]
    fn test_buffered_writer_basic() {
        let dtypes = vec![FlexTypeEnum::Integer, FlexTypeEnum::String];
        let mut buf = Vec::new();

        {
            let mut writer = BufferedSegmentWriter::new(&mut buf, &dtypes);

            let ints: Vec<FlexType> = (0..10).map(|i| FlexType::Integer(i)).collect();
            writer
                .write_column_block(0, &ints, FlexTypeEnum::Integer)
                .unwrap();

            let strings: Vec<FlexType> = (0..10)
                .map(|i| FlexType::String(Arc::from(format!("val_{i}"))))
                .collect();
            writer
                .write_column_block(1, &strings, FlexTypeEnum::String)
                .unwrap();

            let sizes = writer.finish().unwrap();
            assert_eq!(sizes, vec![10, 10]);
        }

        let mut reader = make_reader(buf, &dtypes);

        // Verify integers
        let col0 = reader.read_column(0).unwrap();
        assert_eq!(col0.len(), 10);
        for (i, v) in col0.iter().enumerate() {
            assert_eq!(*v, FlexType::Integer(i as i64), "int mismatch at {i}");
        }

        // Verify strings
        let col1 = reader.read_column(1).unwrap();
        assert_eq!(col1.len(), 10);
        for (i, v) in col1.iter().enumerate() {
            let expected = FlexType::String(Arc::from(format!("val_{i}")));
            assert_eq!(*v, expected, "string mismatch at {i}");
        }
    }

    #[test]
    fn test_buffered_writer_many_small_writes() {
        let dtypes = vec![FlexTypeEnum::Integer];
        let mut buf = Vec::new();

        let n = 1000usize;
        {
            let mut writer = BufferedSegmentWriter::new(&mut buf, &dtypes);

            // Write one value at a time.
            for i in 0..n {
                writer
                    .write_column_block(
                        0,
                        &[FlexType::Integer(i as i64)],
                        FlexTypeEnum::Integer,
                    )
                    .unwrap();
            }

            let sizes = writer.finish().unwrap();
            assert_eq!(sizes, vec![n as u64]);
        }

        let mut reader = make_reader(buf, &dtypes);

        // Verify all values round-trip.
        let col0 = reader.read_column(0).unwrap();
        assert_eq!(col0.len(), n);
        for (i, v) in col0.iter().enumerate() {
            assert_eq!(*v, FlexType::Integer(i as i64), "mismatch at {i}");
        }

        // Verify that values were coalesced into fewer blocks than 1000.
        let num_blocks = reader.num_blocks(0);
        assert!(
            num_blocks < n,
            "Expected coalescing: {num_blocks} blocks for {n} values"
        );
    }

    #[test]
    fn test_buffered_writer_empty() {
        let dtypes = vec![FlexTypeEnum::Integer];
        let mut buf = Vec::new();

        {
            let writer = BufferedSegmentWriter::new(&mut buf, &dtypes);
            let sizes = writer.finish().unwrap();
            assert_eq!(sizes, vec![0]);
        }
    }
}
