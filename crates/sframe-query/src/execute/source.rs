//! SFrameSource operator: segment-prefetching reads from disk.

use std::sync::Arc;

use futures::stream::{self, Stream};

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::local_fs::LocalFileSystem;
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_storage::sframe_reader::SFrameMetadata;
use sframe_storage::segment_reader::SegmentReader;
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::batch::{ColumnData, SFrameRows};

use super::BatchStream;

/// State for segment-prefetching source reading.
struct PrefetchSourceState {
    rx: std::sync::mpsc::Receiver<Result<Vec<Vec<FlexType>>>>,
    dtypes: Vec<FlexTypeEnum>,
    batch_size: usize,
    seg_data: Option<Vec<Vec<FlexType>>>,
    row_offset: usize,
}

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

/// Compile an SFrame source with segment prefetching and optional row range.
///
/// A background thread reads segments ahead into a bounded channel. The
/// stream consumer pulls from the channel and slices segments into batches.
///
/// When `begin_row > 0` or `end_row < total`, only overlapping segments
/// are read, and partial segments are sliced at block boundaries.
pub(super) fn compile_sframe_source(
    path: &str,
    column_types: &[FlexTypeEnum],
    begin_row: u64,
    end_row: u64,
) -> Result<BatchStream> {
    let vfs = resolve_vfs(path);
    let meta = SFrameMetadata::open_with_fs(&*vfs, path)?;
    let dtypes: Vec<FlexTypeEnum> = column_types.to_vec();
    let batch_size = sframe_config::global().source_batch_size;
    let n_prefetch = sframe_config::global().source_prefetch_segments.max(1);

    let segment_paths: Vec<String> = meta
        .group_index
        .segment_files
        .iter()
        .map(|f| format!("{}/{}", path, f))
        .collect();

    // Get segment sizes from first column (all columns have same segment sizes).
    let segment_sizes: Vec<u64> = meta
        .group_index
        .columns[0]
        .segment_sizes
        .clone();

    let slices = compute_segment_slices(&segment_paths, &segment_sizes, begin_row, end_row);

    let (tx, rx) =
        std::sync::mpsc::sync_channel::<Result<Vec<Vec<FlexType>>>>(n_prefetch);
    let dtypes_bg = dtypes.clone();
    let vfs_bg = vfs.clone();

    std::thread::spawn(move || {
        for slice in slices {
            let result = read_segment_row_range(
                &*vfs_bg,
                &slice.segment_path,
                &dtypes_bg,
                slice.local_begin,
                slice.local_end,
            );
            if tx.send(result).is_err() {
                break; // receiver dropped
            }
        }
    });

    let state = PrefetchSourceState {
        rx,
        dtypes,
        batch_size,
        seg_data: None,
        row_offset: 0,
    };

    Ok(Box::pin(unfold_prefetch(state)))
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
) -> Result<BatchStream> {
    let vfs = resolve_vfs(path);
    let meta = SFrameMetadata::open_with_fs(&*vfs, path)?;
    let projected_dtypes: Vec<FlexTypeEnum> =
        column_indices.iter().map(|&i| column_types[i]).collect();
    let all_col_types: Vec<FlexTypeEnum> = column_types.to_vec();
    let proj_indices: Vec<usize> = column_indices.to_vec();
    let batch_size = sframe_config::global().source_batch_size;
    let n_prefetch = sframe_config::global().source_prefetch_segments.max(1);

    let segment_paths: Vec<String> = meta
        .group_index
        .segment_files
        .iter()
        .map(|f| format!("{}/{}", path, f))
        .collect();

    let segment_sizes: Vec<u64> = meta
        .group_index
        .columns[0]
        .segment_sizes
        .clone();

    let slices = compute_segment_slices(&segment_paths, &segment_sizes, begin_row, end_row);

    let (tx, rx) =
        std::sync::mpsc::sync_channel::<Result<Vec<Vec<FlexType>>>>(n_prefetch);
    let vfs_bg = vfs.clone();

    std::thread::spawn(move || {
        for slice in slices {
            let result = read_segment_columns_projected_row_range(
                &*vfs_bg,
                &slice.segment_path,
                &all_col_types,
                &proj_indices,
                slice.local_begin,
                slice.local_end,
            );
            if tx.send(result).is_err() {
                break;
            }
        }
    });

    let state = PrefetchSourceState {
        rx,
        dtypes: projected_dtypes,
        batch_size,
        seg_data: None,
        row_offset: 0,
    };

    Ok(Box::pin(unfold_prefetch(state)))
}

/// Shared unfold logic for prefetched segment reading.
fn unfold_prefetch(
    state: PrefetchSourceState,
) -> impl Stream<Item = Result<SFrameRows>> {
    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(ref data) = state.seg_data {
                let seg_rows = if data.is_empty() { 0 } else { data[0].len() };
                if state.row_offset < seg_rows {
                    let end = (state.row_offset + state.batch_size).min(seg_rows);
                    let batch = slice_columns_to_batch(
                        data,
                        &state.dtypes,
                        state.row_offset,
                        end,
                    );
                    state.row_offset = end;
                    return Some((batch, state));
                }
                state.seg_data = None;
                state.row_offset = 0;
            }

            match state.rx.recv() {
                Err(_) => return None, // channel closed, all segments done
                Ok(Ok(data)) => {
                    if data.is_empty() || data.first().map(|c| c.is_empty()).unwrap_or(true) {
                        continue;
                    }
                    state.seg_data = Some(data);
                }
                Ok(Err(e)) => return Some((Err(e), state)),
            }
        }
    })
}

/// Slice column data into an SFrameRows batch for rows [start..end).
fn slice_columns_to_batch(
    data: &[Vec<FlexType>],
    dtypes: &[FlexTypeEnum],
    start: usize,
    end: usize,
) -> Result<SFrameRows> {
    let columns: Vec<ColumnData> = data.iter()
        .zip(dtypes.iter())
        .map(|(col_data, &dtype)| ColumnData::from_flex_slice(&col_data[start..end], dtype))
        .collect();
    SFrameRows::new(columns)
}

/// Read a row range `[local_begin, local_end)` from a segment.
///
/// Fast path: when reading the entire segment, delegates to
/// `read_segment_independently`. Otherwise, walks the block index
/// to skip non-overlapping blocks and slices partial blocks at boundaries.
fn read_segment_row_range(
    vfs: &dyn VirtualFileSystem,
    segment_path: &str,
    column_types: &[FlexTypeEnum],
    local_begin: u64,
    local_end: u64,
) -> Result<Vec<Vec<FlexType>>> {
    let file = vfs.open_read(segment_path)?;
    let file_size = file.size()?;
    let mut seg_reader = SegmentReader::open(
        Box::new(file),
        file_size,
        column_types.to_vec(),
    )?;

    // Fast path: full segment read
    let seg_total = seg_reader.column_len(0);
    if local_begin == 0 && local_end >= seg_total {
        let num_cols = seg_reader.num_columns();
        let mut columns = Vec::with_capacity(num_cols);
        for col in 0..num_cols {
            columns.push(seg_reader.read_column(col)?);
        }
        return Ok(columns);
    }

    read_columns_block_range(&mut seg_reader, local_begin, local_end, None)
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
    let mut seg_reader = SegmentReader::open(
        Box::new(file),
        file_size,
        column_types.to_vec(),
    )?;

    // Fast path: full segment read
    let seg_total = seg_reader.column_len(0);
    if local_begin == 0 && local_end >= seg_total {
        let mut columns = Vec::with_capacity(column_indices.len());
        for &col_idx in column_indices {
            columns.push(seg_reader.read_column(col_idx)?);
        }
        return Ok(columns);
    }

    read_columns_block_range(&mut seg_reader, local_begin, local_end, Some(column_indices))
}

/// Read columns from a segment for rows `[local_begin, local_end)`,
/// using block-level skipping.
///
/// Each column's block structure is walked independently since different
/// columns can have different block sizes in SFrame V2.
///
/// If `column_indices` is None, reads all columns; otherwise only the
/// specified columns (in given order).
fn read_columns_block_range(
    seg_reader: &mut SegmentReader,
    local_begin: u64,
    local_end: u64,
    column_indices: Option<&[usize]>,
) -> Result<Vec<Vec<FlexType>>> {
    let total_rows = (local_end - local_begin) as usize;

    let cols_to_read: Vec<usize> = match column_indices {
        Some(indices) => indices.to_vec(),
        None => (0..seg_reader.num_columns()).collect(),
    };

    let mut result = Vec::with_capacity(cols_to_read.len());

    for &col_idx in &cols_to_read {
        let mut col_data = Vec::with_capacity(total_rows);
        let num_blocks = seg_reader.num_blocks(col_idx);
        let mut row_cursor = 0u64;

        for block_idx in 0..num_blocks {
            let block_rows = seg_reader.block_num_elem(col_idx, block_idx);
            let block_start = row_cursor;
            let block_end = row_cursor + block_rows;
            row_cursor = block_end;

            // No overlap with [local_begin, local_end)?
            if block_end <= local_begin || block_start >= local_end {
                continue;
            }

            let block_data = seg_reader.read_block(col_idx, block_idx)?;
            let skip = local_begin.saturating_sub(block_start) as usize;
            let take_end = (local_end - block_start).min(block_rows) as usize;
            col_data.extend_from_slice(&block_data[skip..take_end]);
        }

        result.push(col_data);
    }

    Ok(result)
}

/// Read only projected columns from a segment file.
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
