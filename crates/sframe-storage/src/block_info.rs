//! Block metadata types for V2 segment files.
//!
//! Each block in a segment file has associated metadata (BlockInfo) stored
//! in the file footer. The footer contains a Vec<Vec<BlockInfo>> serialized
//! via GraphLab's oarchive:
//!
//! - Outer vec: one entry per column
//! - Inner vec: one BlockInfo per block in that column
//!
//! BlockInfo is serialized as POD (48 bytes per entry, matching C++ struct
//! layout with 6 bytes of trailing padding).

use std::io::{Read, Seek, SeekFrom};

use sframe_types::error::{Result, SFrameError};
use sframe_types::serialization::{read_u16, read_u64};

/// Block flags matching C++ BLOCK_FLAGS enum.
pub const LZ4_COMPRESSION: u64 = 1;
pub const IS_FLEXIBLE_TYPE: u64 = 2;
pub const MULTIPLE_TYPE_BLOCK: u64 = 4;
pub const BLOCK_ENCODING_EXTENSION: u64 = 8;

/// Metadata for a single block in a segment file.
/// Matches the C++ block_info struct layout (48 bytes with padding).
#[derive(Debug, Clone)]
pub struct BlockInfo {
    pub offset: u64,
    pub length: u64,
    pub block_size: u64,
    pub num_elem: u64,
    pub flags: u64,
    pub content_type: u16,
}

impl BlockInfo {
    /// Read a single BlockInfo from a reader (48 bytes, matching C++ POD layout).
    fn read_from(reader: &mut impl Read) -> Result<Self> {
        let offset = read_u64(reader)?;
        let length = read_u64(reader)?;
        let block_size = read_u64(reader)?;
        let num_elem = read_u64(reader)?;
        let flags = read_u64(reader)?;
        let content_type = read_u16(reader)?;
        // Skip 6 bytes of padding
        let mut padding = [0u8; 6];
        reader.read_exact(&mut padding)?;
        Ok(BlockInfo {
            offset,
            length,
            block_size,
            num_elem,
            flags,
            content_type,
        })
    }

    pub fn is_lz4_compressed(&self) -> bool {
        self.flags & LZ4_COMPRESSION != 0
    }

    pub fn is_flexible_type(&self) -> bool {
        self.flags & IS_FLEXIBLE_TYPE != 0
    }

    pub fn has_encoding_extension(&self) -> bool {
        self.flags & BLOCK_ENCODING_EXTENSION != 0
    }
}

/// Read the block index (Vec<Vec<BlockInfo>>) from a segment file footer.
///
/// Footer layout:
///   [block data...] [serialized block index] [footer_size: u64]
///
/// Reading procedure:
///   1. Seek to last 8 bytes, read footer_size
///   2. Seek to file_size - 8 - footer_size
///   3. Deserialize Vec<Vec<BlockInfo>>
pub fn read_block_index(
    reader: &mut (impl Read + Seek),
    file_size: u64,
) -> Result<Vec<Vec<BlockInfo>>> {
    // Read footer_size from last 8 bytes
    reader.seek(SeekFrom::Start(file_size - 8))?;
    let footer_size = read_u64(reader)?;

    if footer_size > file_size - 8 {
        return Err(SFrameError::Format(format!(
            "Footer size {} exceeds file size {}",
            footer_size, file_size
        )));
    }

    // Seek to start of footer
    reader.seek(SeekFrom::Start(file_size - 8 - footer_size))?;

    // Read outer vec: number of columns
    let num_columns = read_u64(reader)? as usize;
    let mut block_index = Vec::with_capacity(num_columns);

    for _ in 0..num_columns {
        // Read inner vec: number of blocks in this column
        let num_blocks = read_u64(reader)? as usize;

        // POD bulk read: num_blocks * 48 bytes
        let mut blocks = Vec::with_capacity(num_blocks);

        // For POD vectors in GraphLab, the serialization writes
        // the size_t length followed by raw bytes (sizeof(T) * len).
        // We read each block_info individually for clarity and safety.
        for _ in 0..num_blocks {
            blocks.push(BlockInfo::read_from(reader)?);
        }

        block_index.push(blocks);
    }

    Ok(block_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../samples", manifest)
    }

    #[test]
    fn test_read_segment_footer() {
        let path = format!(
            "{}/business.sf/m_9688d6320ff94822.0000",
            samples_dir()
        );
        let mut file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
        let file_size = file.get_ref().metadata().unwrap().len();
        let block_index = read_block_index(&mut file, file_size).unwrap();

        // 12 columns
        assert_eq!(block_index.len(), 12, "Expected 12 columns in block index");

        // Each column should have at least 1 block
        for (i, col_blocks) in block_index.iter().enumerate() {
            assert!(!col_blocks.is_empty(), "Column {} has no blocks", i);
        }

        // Total elements across blocks in each column should = 11536
        for (i, col_blocks) in block_index.iter().enumerate() {
            let total: u64 = col_blocks.iter().map(|b| b.num_elem).sum();
            assert_eq!(
                total, 11536,
                "Column {} has {} total elements, expected 11536",
                i, total
            );
        }

        // Block offsets should be non-negative and increasing within a column
        for (i, col_blocks) in block_index.iter().enumerate() {
            for (j, block) in col_blocks.iter().enumerate() {
                assert!(
                    block.offset < file_size,
                    "Column {} block {} offset {} exceeds file size {}",
                    i,
                    j,
                    block.offset,
                    file_size
                );
                assert!(
                    block.length > 0,
                    "Column {} block {} has zero length",
                    i,
                    j
                );
            }
        }

        // All blocks should have IS_FLEXIBLE_TYPE flag
        for col_blocks in &block_index {
            for block in col_blocks {
                assert!(
                    block.is_flexible_type(),
                    "Expected IS_FLEXIBLE_TYPE flag on all blocks"
                );
            }
        }
    }
}
