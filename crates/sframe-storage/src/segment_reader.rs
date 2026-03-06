//! Reads columns from a V2 segment file.
//!
//! A segment file contains blocks for multiple columns. The footer
//! (read by block_info::read_block_index) maps columns to block offsets.
//! This module reads and decodes individual blocks and entire columns.
//!
//! `CachedSegmentReader` wraps `SegmentReader` with a per-block cache that
//! uses `EncodedBlockRange` for streaming decode. Sequential reads stream
//! through blocks incrementally; backward access falls back to full decode.

use std::io::{Read, Seek, SeekFrom};

use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::block_decode::decode_typed_block;
use crate::block_info::{read_block_index, BlockInfo};
use crate::encoded_block_range::EncodedBlockRange;

/// Reader for a single V2 segment file.
pub struct SegmentReader {
    file: Box<dyn ReadSeek>,
    pub block_index: Vec<Vec<BlockInfo>>,
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
                SFrameError::Format(format!("LZ4 decompression failed: {e}"))
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

// ============================================================================
// CachedSegmentReader
// ============================================================================

/// Cache slot for a single block.
enum CacheSlot {
    /// No cached data.
    Empty,
    /// Streaming decoder positioned at some row within the block.
    Encoded(EncodedBlockRange),
    /// Fully decoded values (fallback for backward access).
    Decoded { data: Vec<FlexType> },
}

/// A caching wrapper around `SegmentReader` that uses `EncodedBlockRange`
/// for incremental streaming decode.
///
/// Sequential access streams through blocks via the `Encoded` state.
/// Backward access (reading before the decoder's current position) falls
/// back to the `Decoded` state with full block decode.
pub struct CachedSegmentReader {
    inner: SegmentReader,
    /// cumulative_rows[col] has length num_blocks(col) + 1.
    /// cumulative_rows[col][0] = 0, cumulative_rows[col][i] = sum of num_elem for blocks 0..i.
    cumulative_rows: Vec<Vec<u64>>,
    /// Per-column, per-block cache.
    cache: Vec<Vec<CacheSlot>>,
    /// Number of non-Empty cache slots.
    cached_block_count: usize,
    /// Maximum cached blocks before eviction.
    max_cached_blocks: usize,
}

impl CachedSegmentReader {
    /// Create a new cached reader wrapping a `SegmentReader`.
    pub fn new(inner: SegmentReader, max_cached_blocks: usize) -> Self {
        let num_cols = inner.num_columns();
        let mut cumulative_rows = Vec::with_capacity(num_cols);
        let mut cache = Vec::with_capacity(num_cols);

        for col in 0..num_cols {
            let num_blocks = inner.num_blocks(col);
            let mut cum = Vec::with_capacity(num_blocks + 1);
            cum.push(0u64);
            for blk in 0..num_blocks {
                let prev = *cum.last().unwrap();
                cum.push(prev + inner.block_num_elem(col, blk));
            }
            cumulative_rows.push(cum);

            let slots: Vec<CacheSlot> = (0..num_blocks).map(|_| CacheSlot::Empty).collect();
            cache.push(slots);
        }

        CachedSegmentReader {
            inner,
            cumulative_rows,
            cache,
            cached_block_count: 0,
            max_cached_blocks,
        }
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.inner.num_columns()
    }

    /// Total rows in a column (from precomputed cumulative sums).
    pub fn column_len(&self, col: usize) -> u64 {
        *self.cumulative_rows[col].last().unwrap()
    }

    /// Number of blocks in a column.
    pub fn num_blocks(&self, col: usize) -> usize {
        self.inner.num_blocks(col)
    }

    /// Access the inner SegmentReader.
    pub fn inner(&self) -> &SegmentReader {
        &self.inner
    }

    /// Access the inner SegmentReader mutably.
    pub fn inner_mut(&mut self) -> &mut SegmentReader {
        &mut self.inner
    }

    /// Current number of cached (non-Empty) block slots.
    pub fn cached_block_count(&self) -> usize {
        self.cached_block_count
    }

    /// Clear all cached blocks.
    pub fn clear_cache(&mut self) {
        for col_cache in &mut self.cache {
            for slot in col_cache.iter_mut() {
                *slot = CacheSlot::Empty;
            }
        }
        self.cached_block_count = 0;
    }

    /// Find the block index containing `row` in `cumulative_rows[col]`.
    /// Returns the largest `i` where `cumulative_rows[col][i] <= row`.
    fn find_block_containing_row(&self, col: usize, row: u64) -> usize {
        let cum = &self.cumulative_rows[col];
        // Binary search: find rightmost i where cum[i] <= row.
        // partition_point returns the first i where cum[i] > row.
        let pos = cum.partition_point(|&x| x <= row);
        // pos is at least 1 (cum[0] = 0 <= any row), we want pos - 1.
        // But clamp to valid block range.
        pos.saturating_sub(1).min(cum.len().saturating_sub(2))
    }

    /// Read rows `[begin, end)` from a single column.
    pub fn read_column_rows(&mut self, col: usize, begin: u64, end: u64) -> Result<Vec<FlexType>> {
        let col_len = self.column_len(col);
        let end = end.min(col_len);
        if begin >= end {
            return Ok(Vec::new());
        }

        let total = (end - begin) as usize;
        let mut result = Vec::with_capacity(total);

        let first_block = self.find_block_containing_row(col, begin);
        let last_block = self.find_block_containing_row(col, end - 1);

        for blk in first_block..=last_block {
            let block_start = self.cumulative_rows[col][blk];
            let block_end = self.cumulative_rows[col][blk + 1];

            let local_begin = (begin.max(block_start) - block_start) as usize;
            let local_end = (end.min(block_end) - block_start) as usize;

            self.read_from_block(col, blk, local_begin, local_end, &mut result)?;
        }

        Ok(result)
    }

    /// Read rows `[begin, end)` from multiple columns.
    pub fn read_columns_rows(
        &mut self,
        columns: &[usize],
        begin: u64,
        end: u64,
    ) -> Result<Vec<Vec<FlexType>>> {
        columns
            .iter()
            .map(|&col| self.read_column_rows(col, begin, end))
            .collect()
    }

    /// Read `[local_begin, local_end)` within block `blk` of column `col`,
    /// appending to `result`.
    fn read_from_block(
        &mut self,
        col: usize,
        blk: usize,
        local_begin: usize,
        local_end: usize,
        result: &mut Vec<FlexType>,
    ) -> Result<()> {
        let count = local_end - local_begin;
        if count == 0 {
            return Ok(());
        }

        // Take the slot out so we can work with it without borrowing self.
        let slot = std::mem::replace(&mut self.cache[col][blk], CacheSlot::Empty);

        match slot {
            CacheSlot::Empty => {
                // Cache miss: load block into Encoded state.
                self.evict_if_needed();
                let range = self.load_encoded_block(col, blk)?;
                self.cache[col][blk] = CacheSlot::Encoded(range);
                self.cached_block_count += 1;
                // Now serve from the newly loaded Encoded slot.
                self.serve_encoded(col, blk, local_begin, local_end, result)
            }
            CacheSlot::Encoded(range) => {
                self.cache[col][blk] = CacheSlot::Encoded(range);
                self.serve_encoded(col, blk, local_begin, local_end, result)
            }
            CacheSlot::Decoded { data } => {
                result.extend_from_slice(&data[local_begin..local_end]);
                self.cache[col][blk] = CacheSlot::Decoded { data };
                Ok(())
            }
        }
    }

    /// Serve rows from an Encoded cache slot. Handles forward access (skip+decode)
    /// and backward access (transition to Decoded).
    fn serve_encoded(
        &mut self,
        col: usize,
        blk: usize,
        local_begin: usize,
        local_end: usize,
        result: &mut Vec<FlexType>,
    ) -> Result<()> {
        // Extract the range to work with it.
        let slot = std::mem::replace(&mut self.cache[col][blk], CacheSlot::Empty);
        let CacheSlot::Encoded(mut range) = slot else {
            unreachable!("serve_encoded called on non-Encoded slot");
        };

        let cursor = range.current_row();
        if local_begin >= cursor {
            // Forward access.
            if local_begin > cursor {
                range.skip(local_begin - cursor);
            }
            let values = range.decode_next(local_end - local_begin);
            result.extend(values);

            if range.is_exhausted() {
                // Auto-evict: block fully consumed.
                self.cache[col][blk] = CacheSlot::Empty;
                self.cached_block_count -= 1;
            } else {
                self.cache[col][blk] = CacheSlot::Encoded(range);
            }
            Ok(())
        } else {
            // Backward access: full decode into Decoded state.
            drop(range);
            let data = self.inner.read_block(col, blk)?;
            result.extend_from_slice(&data[local_begin..local_end]);
            self.cache[col][blk] = CacheSlot::Decoded { data };
            // cached_block_count stays the same (was Encoded, now Decoded).
            Ok(())
        }
    }

    /// Load a block into an EncodedBlockRange.
    fn load_encoded_block(&mut self, col: usize, blk: usize) -> Result<EncodedBlockRange> {
        let block_info = self.inner.block_index[col][blk].clone();
        let raw = self.inner.read_raw_block(&block_info)?;
        let decompressed = SegmentReader::decompress_block(&raw, &block_info)?;
        Ok(EncodedBlockRange::new(decompressed, block_info))
    }

    /// Evict a random cache slot if at capacity.
    fn evict_if_needed(&mut self) {
        if self.cached_block_count < self.max_cached_blocks {
            return;
        }
        // Count total slots for random probe.
        let total_slots: usize = self.cache.iter().map(|c| c.len()).sum();
        if total_slots == 0 {
            return;
        }

        // Simple deterministic probe: scan from a pseudo-random start.
        // Use cached_block_count as a cheap seed — varies across calls.
        let start = self.cached_block_count.wrapping_mul(6364136223846793005) % total_slots;
        let mut idx = start;
        loop {
            // Map flat index to (col, blk).
            let (col, blk) = self.flat_index_to_col_blk(idx);
            if !matches!(self.cache[col][blk], CacheSlot::Empty) {
                self.cache[col][blk] = CacheSlot::Empty;
                self.cached_block_count -= 1;
                return;
            }
            idx = (idx + 1) % total_slots;
            if idx == start {
                break; // Full loop, nothing to evict (shouldn't happen).
            }
        }
    }

    /// Convert a flat index into (col, blk).
    fn flat_index_to_col_blk(&self, mut flat: usize) -> (usize, usize) {
        for (col, col_cache) in self.cache.iter().enumerate() {
            if flat < col_cache.len() {
                return (col, flat);
            }
            flat -= col_cache.len();
        }
        // Fallback (shouldn't reach here).
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sframe_reader::SFrameReader;

    fn samples_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{manifest}/../../samples")
    }

    fn open_business_sf() -> SFrameReader {
        SFrameReader::open(&format!("{}/business.sf", samples_dir())).unwrap()
    }

    fn make_cached_reader(max_cached_blocks: usize) -> CachedSegmentReader {
        let mut sf = open_business_sf();
        let seg = sf.segment_readers.remove(0);
        CachedSegmentReader::new(seg, max_cached_blocks)
    }

    #[test]
    fn test_cumulative_rows() {
        let cached = make_cached_reader(64);
        for col in 0..cached.num_columns() {
            let cum = &cached.cumulative_rows[col];
            assert_eq!(cum[0], 0, "cumulative_rows[{col}][0] should be 0");
            assert_eq!(
                *cum.last().unwrap(),
                cached.column_len(col),
                "cumulative_rows[{col}] last element should match column_len"
            );
            // Cumulative rows should be monotonically non-decreasing.
            for i in 1..cum.len() {
                assert!(
                    cum[i] >= cum[i - 1],
                    "cumulative_rows[{col}] not monotonic at index {i}"
                );
            }
            // Length should be num_blocks + 1.
            assert_eq!(cum.len(), cached.num_blocks(col) + 1);
        }
    }

    #[test]
    fn test_find_block_containing_row() {
        // Test indirectly: read row 0, last row, and block boundaries.
        let mut cached = make_cached_reader(64);
        let col = 0;
        let col_len = cached.column_len(col);
        assert!(col_len > 0);

        // Row 0: should return the first value.
        let first = cached.read_column_rows(col, 0, 1).unwrap();
        assert_eq!(first.len(), 1);

        // Last row.
        let last = cached.read_column_rows(col, col_len - 1, col_len).unwrap();
        assert_eq!(last.len(), 1);

        // Block boundary: read across the boundary of block 0 and block 1.
        if cached.num_blocks(col) > 1 {
            let boundary = cached.cumulative_rows[col][1];
            // Read one row before and one row after the boundary.
            let across = cached.read_column_rows(col, boundary - 1, boundary + 1).unwrap();
            assert_eq!(across.len(), 2);
        }
    }

    #[test]
    fn test_full_column_read() {
        let mut sf = open_business_sf();
        let seg = sf.segment_readers.remove(0);
        let num_cols = seg.num_columns();

        // Read full columns via SegmentReader first.
        let mut reference_columns = Vec::new();
        {
            // Need a separate SegmentReader for reference data since we move it.
            let mut sf2 = open_business_sf();
            let seg2 = &mut sf2.segment_readers[0];
            for col in 0..num_cols {
                reference_columns.push(seg2.read_column(col).unwrap());
            }
        }

        let mut cached = CachedSegmentReader::new(seg, 64);
        for col in 0..num_cols {
            let total = cached.column_len(col);
            let result = cached.read_column_rows(col, 0, total).unwrap();
            assert_eq!(
                result.len(),
                reference_columns[col].len(),
                "Column {col} length mismatch"
            );
            for (i, (a, b)) in result.iter().zip(reference_columns[col].iter()).enumerate() {
                assert_eq!(a, b, "Column {col} mismatch at row {i}");
            }
        }
    }

    #[test]
    fn test_single_block_range() {
        let mut sf = open_business_sf();
        let mut sf2 = open_business_sf();
        let seg_ref = &mut sf2.segment_readers[0];
        let seg = sf.segment_readers.remove(0);

        let col = 0;
        let full = seg_ref.read_column(col).unwrap();
        let mut cached = CachedSegmentReader::new(seg, 64);

        // Read a range within the first block.
        let block0_end = cached.cumulative_rows[col][1] as usize;
        let begin = 10.min(block0_end);
        let end = 50.min(block0_end);
        let result = cached.read_column_rows(col, begin as u64, end as u64).unwrap();
        assert_eq!(result, &full[begin..end]);
    }

    #[test]
    fn test_cross_block_boundary() {
        let mut sf = open_business_sf();
        let mut sf2 = open_business_sf();
        let seg_ref = &mut sf2.segment_readers[0];
        let seg = sf.segment_readers.remove(0);

        let col = 0;
        let full = seg_ref.read_column(col).unwrap();
        let mut cached = CachedSegmentReader::new(seg, 64);

        if cached.num_blocks(col) > 1 {
            let boundary = cached.cumulative_rows[col][1];
            // Read a range that spans two blocks.
            let begin = boundary.saturating_sub(20);
            let end = (boundary + 20).min(cached.column_len(col));
            let result = cached.read_column_rows(col, begin, end).unwrap();
            assert_eq!(result, &full[begin as usize..end as usize]);
        }
    }

    #[test]
    fn test_partial_first_last_block() {
        let mut sf = open_business_sf();
        let mut sf2 = open_business_sf();
        let seg_ref = &mut sf2.segment_readers[0];
        let seg = sf.segment_readers.remove(0);

        let col = 0;
        let full = seg_ref.read_column(col).unwrap();
        let mut cached = CachedSegmentReader::new(seg, 64);

        if cached.num_blocks(col) >= 3 {
            // Start mid-first-block, end mid-last-block (spanning 3+ blocks).
            let block1_start = cached.cumulative_rows[col][1];
            let block2_end = cached.cumulative_rows[col][3];
            let begin = block1_start.saturating_sub(10);
            let end = (block2_end - 10).max(block1_start + 1);
            let result = cached.read_column_rows(col, begin, end).unwrap();
            assert_eq!(result, &full[begin as usize..end as usize]);
        } else {
            // Few blocks: just read a middle slice.
            let total = cached.column_len(col);
            let begin = total / 4;
            let end = 3 * total / 4;
            let result = cached.read_column_rows(col, begin, end).unwrap();
            assert_eq!(result, &full[begin as usize..end as usize]);
        }
    }

    #[test]
    fn test_sequential_auto_eviction() {
        let mut sf = open_business_sf();
        let seg = sf.segment_readers.remove(0);

        // Find a column with multiple blocks.
        let col = 0;
        let mut cached = CachedSegmentReader::new(seg, 64);
        let num_blocks = cached.num_blocks(col);
        if num_blocks < 2 {
            return; // Skip if only one block.
        }

        // Read sequentially block by block; after fully consuming a block,
        // its cache should be auto-evicted.
        for blk in 0..num_blocks {
            let blk_start = cached.cumulative_rows[col][blk];
            let blk_end = cached.cumulative_rows[col][blk + 1];
            let _ = cached.read_column_rows(col, blk_start, blk_end).unwrap();
        }
        // After sequential full-block reads, all blocks should be auto-evicted
        // (each block is fully consumed in one read).
        assert_eq!(
            cached.cached_block_count(),
            0,
            "All blocks should be auto-evicted after full sequential reads"
        );
    }

    #[test]
    fn test_backward_access_triggers_decoded() {
        let mut sf = open_business_sf();
        let mut sf2 = open_business_sf();
        let seg_ref = &mut sf2.segment_readers[0];
        let seg = sf.segment_readers.remove(0);

        let col = 0;
        let full = seg_ref.read_column(col).unwrap();
        let mut cached = CachedSegmentReader::new(seg, 64);

        let col_len = cached.column_len(col);
        if col_len < 200 {
            return;
        }

        // Forward read: [100, 200).
        let r1 = cached.read_column_rows(col, 100, 200).unwrap();
        assert_eq!(r1, &full[100..200]);

        let count_before = cached.cached_block_count();

        // Backward read: [50, 150) overlaps with already-advanced cursor.
        let r2 = cached.read_column_rows(col, 50, 150).unwrap();
        assert_eq!(r2, &full[50..150]);

        // The block(s) involved should have transitioned Encoded -> Decoded.
        // cached_block_count should remain the same (state change, not eviction).
        assert_eq!(
            cached.cached_block_count(),
            count_before,
            "Backward access should not change cached_block_count"
        );

        // Verify the block containing row 50 is now Decoded.
        let blk = cached.find_block_containing_row(col, 50);
        assert!(
            matches!(cached.cache[col][blk], CacheSlot::Decoded { .. }),
            "Block {blk} should be in Decoded state after backward access"
        );
    }

    #[test]
    fn test_eviction_respects_bound() {
        let mut sf = open_business_sf();
        let seg = sf.segment_readers.remove(0);

        let max_cached = 2;
        let mut cached = CachedSegmentReader::new(seg, max_cached);

        // Read from enough distinct blocks to exceed the limit.
        let num_cols = cached.num_columns();
        let mut reads_done = 0;
        for col in 0..num_cols {
            for blk in 0..cached.num_blocks(col) {
                let blk_start = cached.cumulative_rows[col][blk];
                // Read just a few rows to populate cache without exhausting blocks.
                let blk_end = cached.cumulative_rows[col][blk + 1];
                let end = blk_start + 1.min(blk_end - blk_start);
                let _ = cached.read_column_rows(col, blk_start, end).unwrap();
                reads_done += 1;
                assert!(
                    cached.cached_block_count() <= max_cached,
                    "cached_block_count {} exceeds max {} after {} reads",
                    cached.cached_block_count(),
                    max_cached,
                    reads_done
                );
                if reads_done > max_cached + 2 {
                    return; // Enough to verify.
                }
            }
        }
    }

    #[test]
    fn test_multi_column_read() {
        let mut sf = open_business_sf();
        let mut sf2 = open_business_sf();
        let seg = sf.segment_readers.remove(0);
        let seg_ref = &mut sf2.segment_readers[0];

        let num_cols = seg.num_columns();
        if num_cols < 2 {
            return;
        }

        let columns: Vec<usize> = (0..num_cols.min(3)).collect();
        let mut cached = CachedSegmentReader::new(seg, 64);
        let begin = 10u64;
        let end = 100u64;

        let multi = cached.read_columns_rows(&columns, begin, end).unwrap();
        assert_eq!(multi.len(), columns.len());

        for &col in &columns {
            let full = seg_ref.read_column(col).unwrap();
            assert_eq!(
                multi[col],
                &full[begin as usize..end as usize],
                "Multi-column read mismatch for column {col}"
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        let mut cached = make_cached_reader(64);
        let col = 0;
        let col_len = cached.column_len(col);

        // Empty range: begin == end.
        let empty1 = cached.read_column_rows(col, 10, 10).unwrap();
        assert!(empty1.is_empty(), "begin == end should return empty");

        // Empty range: begin > end.
        let empty2 = cached.read_column_rows(col, 20, 10).unwrap();
        assert!(empty2.is_empty(), "begin > end should return empty");

        // Range past end is clamped.
        let clamped = cached.read_column_rows(col, col_len - 5, col_len + 100).unwrap();
        assert_eq!(clamped.len(), 5, "Range past end should be clamped");

        // begin >= column_len returns empty.
        let past = cached.read_column_rows(col, col_len, col_len + 10).unwrap();
        assert!(past.is_empty(), "begin >= column_len should return empty");

        let way_past = cached.read_column_rows(col, col_len + 100, col_len + 200).unwrap();
        assert!(way_past.is_empty(), "Range entirely past end should return empty");
    }

    #[test]
    fn test_chunked_sequential_matches_full() {
        let mut sf = open_business_sf();
        let seg = sf.segment_readers.remove(0);
        let num_cols = seg.num_columns();

        let mut sf2 = open_business_sf();
        let seg_ref = &mut sf2.segment_readers[0];

        let mut cached = CachedSegmentReader::new(seg, 64);

        for col in 0..num_cols {
            let full = seg_ref.read_column(col).unwrap();
            let total = cached.column_len(col);

            let mut chunked = Vec::new();
            let chunk_size = 100u64;
            let mut offset = 0u64;
            while offset < total {
                let end = (offset + chunk_size).min(total);
                let chunk = cached.read_column_rows(col, offset, end).unwrap();
                chunked.extend(chunk);
                offset = end;
            }

            assert_eq!(
                chunked.len(),
                full.len(),
                "Chunked read length mismatch for column {col}"
            );
            for (i, (a, b)) in chunked.iter().zip(full.iter()).enumerate() {
                assert_eq!(a, b, "Column {col} chunked mismatch at row {i}");
            }
        }
    }
}
