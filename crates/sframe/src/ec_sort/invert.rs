use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::vfs::{ArcCacheFsVfs, VirtualFileSystem};
use sframe_query::planner::PlannerNode;
use sframe_storage::segment_reader::SegmentReader;
use sframe_storage::segment_writer::{BufferedSegmentWriter, SegmentWriter};
use sframe_storage::sframe_writer::{assemble_sframe_from_segments, generate_hash, segment_filename};
use sframe_types::error::Result;
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::sarray::SArray;
use crate::sframe::{AnonymousStore, SFrame};

use super::{
    compute_num_buckets, extract_physical_source, flush_to_segment_writer, read_fmap_chunk,
    CHUNK_SIZE,
};

/// Invert a permutation SArray.
///
/// Given `inverse_map` where `inverse_map[output_pos] = input_row`,
/// produces `forward_map` where `forward_map[input_row] = output_pos`.
///
/// Equivalent to `permute_sframe([0..N), inverse_map)` but without
/// needing to read any input data — values are synthesized from the
/// row index during scatter.
pub(super) fn invert_permutation(inverse_map: &SArray, num_rows: u64) -> Result<SArray> {
    if num_rows == 0 {
        return SArray::from_vec(Vec::new(), FlexTypeEnum::Integer);
    }

    let budget = sframe_config::global().sort_max_memory() / rayon::current_num_threads().max(1);
    let column_bytes = &[8usize]; // single integer column
    let num_buckets = compute_num_buckets(num_rows, column_bytes, budget);
    let rows_per_bucket = ((num_rows + num_buckets as u64 - 1) / num_buckets as u64).max(1);

    eprintln!(
        "[sframe] ec_sort invert_permutation: {num_rows} rows, {num_buckets} buckets"
    );

    let cache_fs = global_cache_fs();

    // The inverse_map is physical (from standard_sort output on CacheFs).
    // Build a single-column SFrame for plan extraction.
    let imap_sf = SFrame::new_with_columns(
        vec![inverse_map.clone()],
        vec!["__imap__".to_string()],
    );
    let imap_source = extract_physical_source(&imap_sf);

    // Scatter phase: for each row i, scatter value=i to bucket inverse_map[i]/rows_per_bucket.
    // Output has 2 columns: [value (the row index), forward_map (inverse_map value)].
    let scatter_dtypes = vec![FlexTypeEnum::Integer, FlexTypeEnum::Integer];

    let scatter_path = cache_fs.alloc_dir();
    let scatter_vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
    scatter_vfs.mkdir_p(&scatter_path)?;
    let scatter_prefix = format!("s_{}", generate_hash(&scatter_path));

    let mut segment_files = Vec::with_capacity(num_buckets);
    let segment_writers: Vec<Mutex<SegmentWriter<Box<dyn sframe_io::vfs::WritableFile>>>> =
        (0..num_buckets)
            .map(|seg_idx| {
                let seg_file = segment_filename(&scatter_prefix, seg_idx);
                let seg_path = format!("{scatter_path}/{seg_file}");
                let file = scatter_vfs.open_write(&seg_path)?;
                segment_files.push(seg_file);
                Ok(Mutex::new(SegmentWriter::new(file, scatter_dtypes.len())))
            })
            .collect::<Result<_>>()?;

    let flush_thresh = (64 * 1024 / 8usize).clamp(8, 256 * 1024); // ~8K rows for integers

    // Read inverse_map in chunks, scatter in parallel across chunks.
    let num_threads = rayon::current_num_threads().max(1);
    let concurrent_chunks = num_threads.max(1);
    let fmap_chunk_size = (budget / 2 / 8 / concurrent_chunks).max(1024) as u64;

    let mut batch_start = 0u64;
    while batch_start < num_rows {
        let mut batch_chunks: Vec<(u64, u64)> = Vec::new();
        let mut row = batch_start;
        for _ in 0..concurrent_chunks {
            if row >= num_rows { break; }
            let end = (row + fmap_chunk_size).min(num_rows);
            batch_chunks.push((row, end));
            row = end;
        }
        batch_start = row;

        // Read all inverse_map chunks for this batch
        let imap_arcs: Vec<Arc<Vec<u64>>> = batch_chunks
            .iter()
            .map(|&(cs, ce)| {
                let vals = read_fmap_chunk(&imap_sf, &imap_source, cs, ce)?;
                Ok(Arc::new(vals))
            })
            .collect::<Result<_>>()?;

        // Process all chunks in parallel — each work item handles one chunk.
        // With a single column there's no column dimension, but we still
        // get parallelism across chunks.
        let errors: Vec<Result<()>> = batch_chunks
            .par_iter()
            .enumerate()
            .map(|(ci, &(cs, _ce))| {
                let imap = &imap_arcs[ci];
                let mut seg_bufs_val: Vec<Vec<FlexType>> = vec![Vec::new(); num_buckets];
                let mut seg_bufs_fmap: Vec<Vec<FlexType>> = vec![Vec::new(); num_buckets];

                for (r, &imap_val) in imap.iter().enumerate() {
                    let global_row = cs + r as u64;
                    let bucket = (imap_val / rows_per_bucket)
                        .min(num_buckets as u64 - 1) as usize;

                    // Value column: the row index (synthesized, not read)
                    seg_bufs_val[bucket].push(FlexType::Integer(global_row as i64));
                    // Forward_map column: the inverse_map value
                    seg_bufs_fmap[bucket].push(FlexType::Integer(imap_val as i64));

                    if seg_bufs_val[bucket].len() >= flush_thresh {
                        let buf_v = std::mem::take(&mut seg_bufs_val[bucket]);
                        let buf_f = std::mem::take(&mut seg_bufs_fmap[bucket]);
                        flush_to_segment_writer(
                            &segment_writers, bucket, 0, &buf_v, FlexTypeEnum::Integer,
                        )?;
                        flush_to_segment_writer(
                            &segment_writers, bucket, 1, &buf_f, FlexTypeEnum::Integer,
                        )?;
                    }
                }

                // Flush remaining
                for seg in 0..num_buckets {
                    let buf_v = std::mem::take(&mut seg_bufs_val[seg]);
                    let buf_f = std::mem::take(&mut seg_bufs_fmap[seg]);
                    if !buf_v.is_empty() {
                        flush_to_segment_writer(
                            &segment_writers, seg, 0, &buf_v, FlexTypeEnum::Integer,
                        )?;
                        flush_to_segment_writer(
                            &segment_writers, seg, 1, &buf_f, FlexTypeEnum::Integer,
                        )?;
                    }
                }

                Ok(())
            })
            .collect();

        for err in errors { err?; }
    }

    // Finish scatter segment writers
    let mut scatter_seg_sizes = Vec::with_capacity(num_buckets);
    for sw_mutex in segment_writers {
        scatter_seg_sizes.push(sw_mutex.into_inner().unwrap().finish()?);
    }

    // Permute phase: for each bucket, read value + fmap columns,
    // permute value by fmap, write to output.
    let output_path = cache_fs.alloc_dir();
    let output_vfs: Arc<dyn VirtualFileSystem> = Arc::new(ArcCacheFsVfs(cache_fs.clone()));
    output_vfs.mkdir_p(&output_path)?;
    let output_prefix = format!("m_{}", generate_hash(&output_path));
    let output_dtypes = vec![FlexTypeEnum::Integer];

    let permute_results: Vec<Option<Result<(String, Vec<u64>, u64)>>> = (0..num_buckets)
        .into_par_iter()
        .map(|bucket_idx| {
            let bucket_rows = scatter_seg_sizes[bucket_idx].first().copied().unwrap_or(0);
            if bucket_rows == 0 { return None; }

            Some((|| -> Result<(String, Vec<u64>, u64)> {
                let seg_path = format!("{scatter_path}/{}", segment_files[bucket_idx]);
                let file = scatter_vfs.open_read(&seg_path)?;
                let file_size = file.size()?;
                let mut reader = SegmentReader::open(
                    Box::new(file), file_size, scatter_dtypes.clone(),
                )?;

                let values = reader.read_column(0)?;    // the row indices
                let fmap_col = reader.read_column(1)?;  // the inverse_map values
                let n = values.len();

                let bucket_start = bucket_idx as u64 * rows_per_bucket;
                let mut permuted = vec![FlexType::Integer(0); n];
                for i in 0..n {
                    let target = match &fmap_col[i] {
                        FlexType::Integer(v) => (*v as u64 - bucket_start) as usize,
                        _ => 0,
                    };
                    permuted[target] = values[i].clone();
                }

                let out_file_name = segment_filename(&output_prefix, bucket_idx);
                let out_path = format!("{output_path}/{out_file_name}");
                let out_file = output_vfs.open_write(&out_path)?;
                let mut out_writer = BufferedSegmentWriter::new(out_file, &output_dtypes);

                for chunk in permuted.chunks(CHUNK_SIZE) {
                    out_writer.write_column_block(0, chunk, FlexTypeEnum::Integer)?;
                }

                let sizes = out_writer.finish()?;
                Ok((out_file_name, sizes, n as u64))
            })())
        })
        .collect();

    // Clean up scatter scratch
    cache_fs.remove_dir(&scatter_path).ok();

    // Assemble output
    let mut out_seg_files = Vec::new();
    let mut out_seg_sizes = Vec::new();
    let mut total_rows_out = 0u64;
    for r in permute_results.into_iter().flatten() {
        let (f, s, n) = r?;
        if n > 0 {
            out_seg_files.push(f);
            out_seg_sizes.push(s);
            total_rows_out += n;
        }
    }

    assemble_sframe_from_segments(
        &*output_vfs, &output_path,
        &["__fwd__"], &[FlexTypeEnum::Integer],
        &out_seg_files, &out_seg_sizes,
        total_rows_out, &HashMap::new(),
    )?;

    let store: Arc<dyn Send + Sync> = Arc::new(AnonymousStore {
        path: output_path.clone(),
        cache_fs: cache_fs.clone(),
    });
    let plan = PlannerNode::sframe_source_cached(
        &output_path,
        vec!["__fwd__".to_string()],
        vec![FlexTypeEnum::Integer],
        total_rows_out,
        store,
    );

    Ok(SArray::from_plan(plan, FlexTypeEnum::Integer, Some(total_rows_out), 0))
}
