use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use sframe_io::vfs::VirtualFileSystem;
use sframe_storage::scatter_writer::ScatterWriter;
use sframe_storage::segment_reader::SegmentReader;
use sframe_storage::segment_writer::SegmentWriter;
use sframe_storage::sframe_reader::SFrameMetadata;
use sframe_storage::sframe_writer::segment_filename;
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::sarray::SArray;
use crate::sframe::SFrame;

use super::{
    classify_into_buckets, estimate_bytes_per_value, extract_physical_source,
    flush_to_segment_writer, read_fmap_chunk, PhysicalSource,
};

/// Scatter all input columns (plus the forward map) into segment files.
///
/// Two strategies:
/// - **Direct path** (materialized inputs): Per-thread `SFrameReader`s
///   read column blocks and scatter directly to `Mutex`-wrapped segment
///   writers. No intermediate accumulation — memory is bounded to one
///   block per thread plus the forward_map chunk.
/// - **Lazy path**: Uses plan slicing (`try_slice`) + ScatterWriter for
///   any plan type.
pub(super) fn scatter_columns(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
    input: &SFrame,
    forward_map: &SArray,
    num_buckets: usize,
    rows_per_bucket: u64,
    num_rows: u64,
    budget: usize,
) -> Result<sframe_storage::scatter_writer::ScatterResult> {
    let input_source = extract_physical_source(input);
    let fmap_sf = SFrame::new_with_columns(
        vec![forward_map.clone()],
        vec!["__fmap__".to_string()],
    );
    let fmap_source = extract_physical_source(&fmap_sf);

    if input_source.is_some() && fmap_source.is_some() {
        eprintln!("[sframe] ec_sort scatter: using direct reader path");
        scatter_columns_direct(
            vfs, base_path, data_prefix, input,
            &fmap_sf, input_source.as_ref().unwrap(), fmap_source.as_ref().unwrap(),
            num_buckets, rows_per_bucket, num_rows, budget,
        )
    } else {
        scatter_columns_lazy(
            vfs, base_path, data_prefix, input, forward_map,
            &fmap_sf, &fmap_source,
            num_buckets, rows_per_bucket, num_rows, budget,
        )
    }
}

/// Direct scatter path for materialized inputs.
///
/// Uses Mutex-wrapped SegmentWriters so rayon threads can write directly.
/// Each thread reads its column block-by-block, classifying and flushing
/// to per-segment buffers as it goes. Memory per thread is bounded to
/// one decoded block + per-segment flush buffers (~64KB each).
fn scatter_columns_direct(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
    input: &SFrame,
    fmap_sf: &SFrame,
    input_source: &PhysicalSource,
    fmap_source: &PhysicalSource,
    num_buckets: usize,
    rows_per_bucket: u64,
    num_rows: u64,
    budget: usize,
) -> Result<sframe_storage::scatter_writer::ScatterResult> {
    let input_dtypes = input.column_types();
    let num_input_cols = input_dtypes.len();
    let fmap_col_idx = num_input_cols;

    let mut scatter_dtypes = input_dtypes.clone();
    scatter_dtypes.push(FlexTypeEnum::Integer);

    // Create M Mutex-wrapped SegmentWriters
    let mut segment_files = Vec::with_capacity(num_buckets);
    let segment_writers: Vec<Mutex<SegmentWriter<Box<dyn sframe_io::vfs::WritableFile>>>> =
        (0..num_buckets)
            .map(|seg_idx| {
                let seg_file = segment_filename(data_prefix, seg_idx);
                let seg_path = format!("{base_path}/{seg_file}");
                let file = vfs.open_write(&seg_path)?;
                segment_files.push(seg_file);
                Ok(Mutex::new(SegmentWriter::new(file, scatter_dtypes.len())))
            })
            .collect::<Result<_>>()?;

    // Per-column flush threshold (rows per output block)
    let flush_threshold: Vec<usize> = scatter_dtypes
        .iter()
        .map(|&dt| {
            let est = estimate_bytes_per_value(dt).max(1);
            (64 * 1024 / est).clamp(8, 256 * 1024)
        })
        .collect();

    // Pre-compute segment layout: cumulative row offsets per input segment.
    let input_meta = SFrameMetadata::open_with_fs(&*input_source.vfs, &input_source.path)?;
    let input_seg_files: Vec<String> = input_meta.group_index.segment_files.clone();
    let col_types: Vec<FlexTypeEnum> = input_meta.group_index.columns
        .iter().map(|c| c.dtype).collect();
    let num_input_segs = input_seg_files.len();

    let ref_col = input_source.column_map[0];
    let seg_row_offsets: Vec<u64> = {
        let mut offsets = Vec::with_capacity(num_input_segs + 1);
        offsets.push(0);
        for seg_idx in 0..num_input_segs {
            let prev = *offsets.last().unwrap();
            offsets.push(prev + input_meta.group_index.columns[ref_col].segment_sizes[seg_idx]);
        }
        offsets
    };

    // Size fmap chunks so we can process multiple concurrently.
    // We want enough work items (chunk x column x segment) to keep all
    // threads busy. Each concurrent chunk holds fmap data in memory.
    let num_threads = rayon::current_num_threads().max(1);
    let items_per_chunk = (num_input_cols * num_input_segs).max(1);
    let concurrent_chunks = ((num_threads * 2) / items_per_chunk).max(1);
    let fmap_chunk_size = {
        let by_budget = budget / 2 / 8 / concurrent_chunks;
        let by_count = ((num_rows as usize + concurrent_chunks - 1) / concurrent_chunks).max(1);
        (by_budget.min(by_count)).max(1024) as u64
    };

    // Process in batches of concurrent_chunks fmap chunks at a time.
    let mut batch_start_row = 0u64;
    while batch_start_row < num_rows {
        // Determine which fmap chunks are in this batch
        let mut batch_chunks: Vec<(u64, u64)> = Vec::new(); // (chunk_start, chunk_end)
        let mut row = batch_start_row;
        for _ in 0..concurrent_chunks {
            if row >= num_rows {
                break;
            }
            let end = (row + fmap_chunk_size).min(num_rows);
            batch_chunks.push((row, end));
            row = end;
        }
        batch_start_row = row;

        // Read all fmap chunks for this batch (can be parallelized too)
        let fmap_arcs: Vec<Arc<Vec<u64>>> = batch_chunks
            .iter()
            .map(|&(cs, ce)| {
                let vals = read_fmap_chunk(
                    fmap_sf, &Some(fmap_source.clone()), cs, ce,
                )?;
                Ok(Arc::new(vals))
            })
            .collect::<Result<_>>()?;

        // Build work items across all chunks in the batch:
        // (chunk_idx, col_idx, input_seg_idx)
        struct WorkItem {
            fmap: Arc<Vec<u64>>,
            chunk_start: u64,
            chunk_end: u64,
            col_idx: usize,
            seg_idx: usize,
        }

        let mut work_items: Vec<WorkItem> = Vec::new();
        for (ci, &(cs, ce)) in batch_chunks.iter().enumerate() {
            for col_idx in 0..num_input_cols {
                for seg_idx in 0..num_input_segs {
                    let seg_start = seg_row_offsets[seg_idx];
                    let seg_end = seg_row_offsets[seg_idx + 1];
                    if seg_end > cs && seg_start < ce {
                        work_items.push(WorkItem {
                            fmap: fmap_arcs[ci].clone(),
                            chunk_start: cs,
                            chunk_end: ce,
                            col_idx,
                            seg_idx,
                        });
                    }
                }
            }
        }

        // Process all work items in parallel across all chunks.
        let work_errors: Vec<Result<()>> = work_items
            .par_iter()
            .map(|item| {
                let phys_col = input_source.column_map[item.col_idx];
                let dtype = input_dtypes[item.col_idx];
                let thresh = flush_threshold[item.col_idx];
                let seg_start = seg_row_offsets[item.seg_idx];

                let seg_path = format!("{}/{}", input_source.path, input_seg_files[item.seg_idx]);
                let file = input_source.vfs.open_read(&seg_path)?;
                let file_size = file.size()?;
                let mut seg_reader = SegmentReader::open(
                    Box::new(file), file_size, col_types.clone(),
                )?;

                let mut seg_bufs: Vec<Vec<FlexType>> = vec![Vec::new(); num_buckets];

                let mut block_row = seg_start;
                for blk_idx in 0..seg_reader.num_blocks(phys_col) {
                    let blk_len = seg_reader.block_num_elem(phys_col, blk_idx);
                    let blk_end = block_row + blk_len;

                    if blk_end <= item.chunk_start {
                        block_row = blk_end;
                        continue;
                    }
                    if block_row >= item.chunk_end {
                        break;
                    }

                    let block_data = seg_reader.read_block(phys_col, blk_idx)?;
                    let local_begin = item.chunk_start.saturating_sub(block_row) as usize;
                    let local_end = (item.chunk_end - block_row).min(blk_len) as usize;

                    for i in local_begin..local_end {
                        let fmap_idx = (block_row + i as u64 - item.chunk_start) as usize;
                        let bucket = (item.fmap[fmap_idx] / rows_per_bucket)
                            .min(num_buckets as u64 - 1) as usize;
                        seg_bufs[bucket].push(block_data[i].clone());

                        if seg_bufs[bucket].len() >= thresh {
                            let buf = std::mem::take(&mut seg_bufs[bucket]);
                            flush_to_segment_writer(
                                &segment_writers, bucket, item.col_idx, &buf, dtype,
                            )?;
                        }
                    }

                    block_row = blk_end;
                }

                for (out_seg, buf) in seg_bufs.into_iter().enumerate() {
                    if !buf.is_empty() {
                        flush_to_segment_writer(
                            &segment_writers, out_seg, item.col_idx, &buf, dtype,
                        )?;
                    }
                }

                Ok(())
            })
            .collect();

        for err in work_errors {
            err?;
        }

        // Scatter the forward_map as the last column for all chunks in batch
        let fmap_dtype = FlexTypeEnum::Integer;
        let fmap_thresh = flush_threshold[fmap_col_idx];
        let mut fmap_seg_bufs: Vec<Vec<FlexType>> = vec![Vec::new(); num_buckets];
        for fmap_arc in &fmap_arcs {
            for &fmap_val in fmap_arc.iter() {
                let bucket = (fmap_val / rows_per_bucket).min(num_buckets as u64 - 1) as usize;
                fmap_seg_bufs[bucket].push(FlexType::Integer(fmap_val as i64));

                if fmap_seg_bufs[bucket].len() >= fmap_thresh {
                    let buf = std::mem::take(&mut fmap_seg_bufs[bucket]);
                    flush_to_segment_writer(
                        &segment_writers, bucket, fmap_col_idx, &buf, fmap_dtype,
                    )?;
                }
            }
        }
        for (seg_idx, buf) in fmap_seg_bufs.into_iter().enumerate() {
            if !buf.is_empty() {
                flush_to_segment_writer(
                    &segment_writers, seg_idx, fmap_col_idx, &buf, fmap_dtype,
                )?;
            }
        }
    }

    // Finish all segment writers
    let mut all_segment_sizes = Vec::with_capacity(num_buckets);
    for sw_mutex in segment_writers {
        let sw = sw_mutex.into_inner().unwrap();
        all_segment_sizes.push(sw.finish()?);
    }

    Ok(sframe_storage::scatter_writer::ScatterResult {
        segment_files,
        all_segment_sizes,
    })
}

/// Lazy scatter path using plan slicing + ScatterWriter.
fn scatter_columns_lazy(
    vfs: &dyn VirtualFileSystem,
    base_path: &str,
    data_prefix: &str,
    input: &SFrame,
    _forward_map: &SArray,
    fmap_sf: &SFrame,
    fmap_source: &Option<PhysicalSource>,
    num_buckets: usize,
    rows_per_bucket: u64,
    num_rows: u64,
    budget: usize,
) -> Result<sframe_storage::scatter_writer::ScatterResult> {
    use sframe_query::execute::materialize_sync;

    let input_dtypes = input.column_types();
    let num_input_cols = input_dtypes.len();
    let fmap_col_idx = num_input_cols;

    let mut scatter_dtypes = input_dtypes.clone();
    scatter_dtypes.push(FlexTypeEnum::Integer);

    let mut scatter_writer = ScatterWriter::new(
        vfs, base_path, data_prefix, &scatter_dtypes, num_buckets,
    )?;

    let num_threads = rayon::current_num_threads().max(1);
    let max_val_size = input.column_types().iter()
        .map(|&dt| estimate_bytes_per_value(dt))
        .max()
        .unwrap_or(8);
    // Budget: fmap chunk + num_threads x chunk_values x 2 (values + buckets)
    let per_row_cost = 8 + num_threads * max_val_size * 2;
    let fmap_chunk_size = (budget / per_row_cost.max(1)).max(1024) as u64;

    let mut chunk_start = 0u64;
    while chunk_start < num_rows {
        let chunk_end = (chunk_start + fmap_chunk_size).min(num_rows);

        let fmap_values = read_fmap_chunk(fmap_sf, fmap_source, chunk_start, chunk_end)?;
        let chunk_len = fmap_values.len();

        let input_columns = input.columns();
        let per_col_results: Vec<Result<Vec<Vec<FlexType>>>> = (0..num_input_cols)
            .into_par_iter()
            .map(|col_idx| {
                let col_sa = input_columns[col_idx].try_slice(chunk_start, chunk_end)?;
                let col_sf = SFrame::new_with_columns(
                    vec![col_sa], vec!["__c__".to_string()],
                );
                let col_batch = materialize_sync(col_sf.compile_stream()?)?;
                let col_data = col_batch.column(0);
                let values: Vec<FlexType> =
                    (0..col_batch.num_rows()).map(|r| col_data.get(r)).collect();
                classify_into_buckets(&values, &fmap_values, num_buckets, rows_per_bucket)
            })
            .collect();

        for (col_idx, result) in per_col_results.into_iter().enumerate() {
            let seg_bufs = result?;
            for (seg_idx, values) in seg_bufs.into_iter().enumerate() {
                for value in values {
                    scatter_writer.write_to_segment(col_idx, seg_idx, value)?;
                }
            }
        }

        for r in 0..chunk_len {
            let fmap_val = fmap_values[r];
            let bucket = (fmap_val / rows_per_bucket).min(num_buckets as u64 - 1) as usize;
            scatter_writer.write_to_segment(
                fmap_col_idx, bucket, FlexType::Integer(fmap_val as i64),
            )?;
        }

        chunk_start = chunk_end;
    }

    scatter_writer.finish()
}
