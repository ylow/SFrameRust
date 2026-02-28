//! Reads columns from a V2 segment file.
//!
//! A segment file contains blocks for multiple columns. The footer
//! (read by block_info::read_block_index) maps columns to block offsets.
//! This module reads and decodes individual blocks and entire columns.

use std::io::{Read, Seek, SeekFrom};

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::block_decode::decode_typed_block;
use crate::block_info::{read_block_index, BlockInfo};

/// Reader for a single V2 segment file.
pub struct SegmentReader {
    file: Box<dyn ReadSeek>,
    pub block_index: Vec<Vec<BlockInfo>>,
    column_types: Vec<FlexTypeEnum>,
}

/// Helper trait combining Read + Seek for boxed storage.
pub trait ReadSeek: Read + Seek + Send {}
impl<T: Read + Seek + Send> ReadSeek for T {}

impl SegmentReader {
    /// Open a segment file and read its footer.
    pub fn open(
        mut file: Box<dyn ReadSeek>,
        file_size: u64,
        column_types: Vec<FlexTypeEnum>,
    ) -> Result<Self> {
        let block_index = read_block_index(&mut *file, file_size)?;

        if block_index.len() != column_types.len() {
            return Err(SFrameError::Format(format!(
                "Block index has {} columns but {} types provided",
                block_index.len(),
                column_types.len()
            )));
        }

        Ok(SegmentReader {
            file,
            block_index,
            column_types,
        })
    }

    /// Number of columns in this segment.
    pub fn num_columns(&self) -> usize {
        self.block_index.len()
    }

    /// Total number of elements in a column (sum across all blocks).
    pub fn column_len(&self, column: usize) -> u64 {
        self.block_index[column].iter().map(|b| b.num_elem).sum()
    }

    /// Number of blocks in a column.
    pub fn num_blocks(&self, column: usize) -> usize {
        self.block_index[column].len()
    }

    /// Number of rows in a specific block.
    pub fn block_num_elem(&self, column: usize, block_idx: usize) -> u64 {
        self.block_index[column][block_idx].num_elem
    }

    /// Read and decode a single block by column and block index.
    pub fn read_block(&mut self, column: usize, block_idx: usize) -> Result<Vec<FlexType>> {
        let block_info = self.block_index[column][block_idx].clone();
        let raw = self.read_raw_block(&block_info)?;
        let decompressed = Self::decompress_block(&raw, &block_info)?;
        decode_typed_block(&decompressed, &block_info)
    }

    /// Read and decode a single raw block.
    pub fn read_raw_block(&mut self, block_info: &BlockInfo) -> Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(block_info.offset))?;
        let mut buf = vec![0u8; block_info.length as usize];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Decompress a block if LZ4-compressed.
    pub fn decompress_block(raw: &[u8], block_info: &BlockInfo) -> Result<Vec<u8>> {
        if block_info.is_lz4_compressed() {
            let decompressed_size = block_info.block_size as usize;
            let decompressed = lz4_flex::decompress(raw, decompressed_size).map_err(|e| {
                SFrameError::Format(format!("LZ4 decompression failed: {}", e))
            })?;
            Ok(decompressed)
        } else {
            Ok(raw.to_vec())
        }
    }

    /// Read all values in a column across all blocks.
    pub fn read_column(&mut self, column: usize) -> Result<Vec<FlexType>> {
        if column >= self.num_columns() {
            return Err(SFrameError::Format(format!(
                "Column index {} out of range ({})",
                column,
                self.num_columns()
            )));
        }

        let total_elems: u64 = self.column_len(column);
        let mut result = Vec::with_capacity(total_elems as usize);

        for block_idx in 0..self.num_blocks(column) {
            let values = self.read_block(column, block_idx)?;

            let expected = self.block_num_elem(column, block_idx) as usize;
            if values.len() != expected {
                return Err(SFrameError::Format(format!(
                    "Column {} block {}: decoded {} values, expected {}",
                    column, block_idx, values.len(), expected
                )));
            }

            result.extend(values);
        }

        Ok(result)
    }
}
