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
        let padded = ((self.bytes_written + BLOCK_ALIGNMENT - 1) / BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT;
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
