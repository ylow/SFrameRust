//! SFrameSource operator: pull-based streaming reads from disk.
//!
//! Source nodes are fully pull-based: each `.next()` on the stream reads
//! the next chunk from a `CachedSegmentReader`. No background threads,
//! no channels. The `CachedSegmentReader`'s block cache provides natural
//! read-ahead at the block level via `EncodedBlockRange` streaming decode.

use std::sync::Arc;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::local_fs::LocalFileSystem;
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_storage::sframe_reader::SFrameMetadata;
use sframe_storage::segment_reader::{CachedSegmentReader, SegmentReader};
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::{ColumnData, SFrameRows};

use super::batch_iter::{BatchCommand, BatchCo, BatchIterator, BatchResponse};

/// Resolve a path to the appropriate VFS backend.
fn resolve_vfs(path: &str) -> Arc<dyn VirtualFileSystem> {
    if path.starts_with("cache://") {
        Arc::new(ArcCacheFsVfs(global_cache_fs().clone()))
    } else {
        Arc::new(LocalFileSystem)
    }
}

/// Describes the row range to read from one segment.
struct SegmentSlice {
    segment_path: String,
    /// Row offset within the segment to start reading (inclusive).
    local_begin: u64,
    /// Row offset within the segment to stop reading (exclusive).
    local_end: u64,
}

/// Compute which segments overlap `[begin_row, end_row)` and what
/// local row range to read from each.
fn compute_segment_slices(
    segment_paths: &[String],
    segment_sizes: &[u64],
    begin_row: u64,
    end_row: u64,
) -> Vec<SegmentSlice> {
    let mut slices = Vec::new();
    let mut cumulative = 0u64;

    for (i, &seg_size) in segment_sizes.iter().enumerate() {
        let seg_start = cumulative;
        let seg_end = cumulative + seg_size;
        cumulative = seg_end;

        // No overlap?
        if seg_end <= begin_row || seg_start >= end_row {
            continue;
        }

        let local_begin = begin_row.saturating_sub(seg_start);
        let local_end = (end_row - seg_start).min(seg_size);

        slices.push(SegmentSlice {
            segment_path: segment_paths[i].clone(),
            local_begin,
            local_end,
        });
    }

    slices
}

// ============================================================================
// Pull-based source state
// ============================================================================

/// State for the pull-based source.
///
/// Holds the VFS, segment slices to read, and the current
/// `CachedSegmentReader` (lazily opened per segment).
struct PullSourceState {
    vfs: Arc<dyn VirtualFileSystem>,
    column_types: Vec<FlexTypeEnum>,
    cols_to_read: Vec<usize>,
    dtypes: Vec<FlexTypeEnum>,
    chunk_size: u64,
    max_blocks: usize,
    slices: Vec<SegmentSlice>,
    /// Index into `slices` for the current segment.
    slice_idx: usize,
    /// Current row offset within the current segment.
    offset: u64,
    /// End row for the current segment slice.
    slice_end: u64,
    /// Cached reader for the current segment (None if not yet opened).
    reader: Option<CachedSegmentReader>,
}

impl PullSourceState {
    /// Open the next segment, advancing slice_idx.
    /// Returns Err on I/O failure, Ok(false) if no more slices.
    fn open_next_segment(&mut self) -> Result<bool> {
        if self.slice_idx >= self.slices.len() {
            return Ok(false);
        }
        let slice = &self.slices[self.slice_idx];
        let file = self.vfs.open_read(&slice.segment_path)?;
        let file_size = file.size()?;
        let seg_reader = SegmentReader::open(
            Box::new(file),
            file_size,
            self.column_types.clone(),
        )?;
        self.reader = Some(CachedSegmentReader::new(seg_reader, self.max_blocks));
        self.offset = slice.local_begin;
        self.slice_end = slice.local_end;
        Ok(true)
    }

    /// Advance to the next chunk position, ensuring a segment is open.
    /// Returns Ok(true) if a chunk is available, Ok(false) if done.
    fn ensure_segment(&mut self) -> Result<bool> {
        if self.reader.is_none() || self.offset >= self.slice_end {
            if self.reader.is_some() {
                self.reader = None;
                self.slice_idx += 1;
            }
            if !self.open_next_segment()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Read the next chunk. Returns None when all slices are exhausted.
    fn next_batch(&mut self) -> Option<Result<SFrameRows>> {
        loop {
            match self.ensure_segment() {
                Err(e) => return Some(Err(e)),
                Ok(false) => return None,
                Ok(true) => {}
            }

            let chunk_end = (self.offset + self.chunk_size).min(self.slice_end);
            let reader = self.reader.as_mut().unwrap();
            let result = reader.read_columns_rows(&self.cols_to_read, self.offset, chunk_end);
            self.offset = chunk_end;

            match result {
                Err(e) => return Some(Err(e)),
                Ok(data) => {
                    if data.is_empty() || data[0].is_empty() {
                        continue; // skip empty chunks (shouldn't happen, but be safe)
                    }
                    return Some(columns_to_batch(&data, &self.dtypes));
                }
            }
        }
    }

    /// Skip the next chunk without decoding data.
    /// Returns true if a chunk was skipped, false if done.
    fn skip_batch(&mut self) -> bool {
        match self.ensure_segment() {
            Err(_) => false,
            Ok(false) => false,
            Ok(true) => {
                let chunk_end = (self.offset + self.chunk_size).min(self.slice_end);
                self.offset = chunk_end;
                true
            }
        }
    }
}

/// Build a pull-based source BatchIterator from segment slices.
fn make_pull_source(
    vfs: Arc<dyn VirtualFileSystem>,
    column_types: Vec<FlexTypeEnum>,
    cols_to_read: Vec<usize>,
    dtypes: Vec<FlexTypeEnum>,
    slices: Vec<SegmentSlice>,
) -> BatchIterator {
    let batch_size = sframe_config::global().source_batch_size;
    let max_blocks = sframe_config::global().max_blocks_in_cache;

    let mut state = PullSourceState {
        vfs,
        column_types,
        cols_to_read,
        dtypes,
        chunk_size: batch_size as u64,
        max_blocks,
        slices,
        slice_idx: 0,
        offset: 0,
        slice_end: 0,
        reader: None,
    };

    BatchIterator::new(move |co: BatchCo| async move {
        let mut cmd = co.yield_(BatchResponse::Ready).await;
        loop {
            match cmd {
                BatchCommand::NextBatch => {
                    match state.next_batch() {
                        None => return,
                        Some(result) => {
                            cmd = co.yield_(BatchResponse::Batch(result)).await;
                        }
                    }
                }
                BatchCommand::SkipBatch => {
                    if state.skip_batch() {
                        cmd = co.yield_(BatchResponse::Skipped).await;
                    } else {
                        return;
                    }
                }
                BatchCommand::Start => unreachable!(),
            }
        }
    })
}

// ============================================================================
// Public compilation functions
// ============================================================================

/// Compile an SFrame source as a pull-based BatchIterator.
///
/// Each `.next_batch()` reads the next chunk directly from a `CachedSegmentReader`.
/// No background threads or channels — fully pull-based.
///
/// When `begin_row > 0` or `end_row < total`, only overlapping segments
/// are read, and partial segments use block-level skipping.
pub(super) fn compile_sframe_source(
    path: &str,
    column_types: &[FlexTypeEnum],
    begin_row: u64,
    end_row: u64,
) -> Result<BatchIterator> {
    let vfs = resolve_vfs(path);
    let meta = SFrameMetadata::open_with_fs(&*vfs, path)?;
    let dtypes: Vec<FlexTypeEnum> = column_types.to_vec();
    let num_cols = dtypes.len();

    let segment_paths: Vec<String> = meta
        .group_index
        .segment_files
        .iter()
        .map(|f| format!("{path}/{f}"))
        .collect();

    let segment_sizes: Vec<u64> = meta
        .group_index
        .columns[0]
        .segment_sizes
        .clone();

    let slices = compute_segment_slices(&segment_paths, &segment_sizes, begin_row, end_row);
    let cols_to_read: Vec<usize> = (0..num_cols).collect();

    Ok(make_pull_source(vfs, dtypes.clone(), cols_to_read, dtypes, slices))
}

/// Compile an SFrame source reading only the projected columns.
///
/// Fuses `Project → SFrameSource` at the I/O level: only the requested
/// columns are decoded from each segment, avoiding disk reads for
/// columns that are projected away.
pub(super) fn compile_sframe_source_projected(
    path: &str,
    column_types: &[FlexTypeEnum],
    begin_row: u64,
    end_row: u64,
    column_indices: &[usize],
) -> Result<BatchIterator> {
    let vfs = resolve_vfs(path);
    let meta = SFrameMetadata::open_with_fs(&*vfs, path)?;
    let all_col_types: Vec<FlexTypeEnum> = column_types.to_vec();
    let projected_dtypes: Vec<FlexTypeEnum> =
        column_indices.iter().map(|&i| column_types[i]).collect();
    let cols_to_read: Vec<usize> = column_indices.to_vec();

    let segment_paths: Vec<String> = meta
        .group_index
        .segment_files
        .iter()
        .map(|f| format!("{path}/{f}"))
        .collect();

    let segment_sizes: Vec<u64> = meta
        .group_index
        .columns[0]
        .segment_sizes
        .clone();

    let slices = compute_segment_slices(&segment_paths, &segment_sizes, begin_row, end_row);

    Ok(make_pull_source(vfs, all_col_types, cols_to_read, projected_dtypes, slices))
}

// ============================================================================
// Helpers
// ============================================================================

/// Convert column vectors to an SFrameRows batch.
fn columns_to_batch(
    data: &[Vec<FlexType>],
    dtypes: &[FlexTypeEnum],
) -> Result<SFrameRows> {
    let columns: Vec<ColumnData> = data.iter()
        .zip(dtypes.iter())
        .map(|(col_data, &dtype)| ColumnData::from_flex_slice(col_data, dtype))
        .collect();
    SFrameRows::new(columns)
}

/// Read a row range from a segment, reading only projected columns.
fn read_segment_columns_projected_row_range(
    vfs: &dyn VirtualFileSystem,
    segment_path: &str,
    column_types: &[FlexTypeEnum],
    column_indices: &[usize],
    local_begin: u64,
    local_end: u64,
) -> Result<Vec<Vec<FlexType>>> {
    let file = vfs.open_read(segment_path)?;
    let file_size = file.size()?;
    let seg_reader = SegmentReader::open(
        Box::new(file),
        file_size,
        column_types.to_vec(),
    )?;
    let max_blocks = sframe_config::global().max_blocks_in_cache;
    let mut cached = CachedSegmentReader::new(seg_reader, max_blocks);

    let cols: Vec<usize> = column_indices.to_vec();
    cached.read_columns_rows(&cols, local_begin, local_end)
}

/// Read projected columns for a global row range `[begin_row, end_row)`
/// across potentially multiple segments.
///
/// Computes which segments overlap the range, reads each with block-level
/// skipping for partial segments, and concatenates the results.
pub(super) fn read_projected_row_range(
    vfs: &dyn VirtualFileSystem,
    segment_paths: &[String],
    segment_sizes: &[u64],
    column_types: &[FlexTypeEnum],
    column_indices: &[usize],
    begin_row: u64,
    end_row: u64,
) -> Result<Vec<Vec<FlexType>>> {
    let slices = compute_segment_slices(segment_paths, segment_sizes, begin_row, end_row);
    let num_projected = column_indices.len();
    let expected_rows = (end_row - begin_row) as usize;
    let mut result: Vec<Vec<FlexType>> = (0..num_projected)
        .map(|_| Vec::with_capacity(expected_rows))
        .collect();

    for slice in slices {
        let seg_columns = read_segment_columns_projected_row_range(
            vfs,
            &slice.segment_path,
            column_types,
            column_indices,
            slice.local_begin,
            slice.local_end,
        )?;
        for (i, col_data) in seg_columns.into_iter().enumerate() {
            result[i].extend(col_data);
        }
    }

    Ok(result)
}
