use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use sframe_io::cache_fs::global_cache_fs;
use sframe_io::vfs::VirtualFileSystem;
use sframe_query::planner::PlannerNode;
use sframe_storage::segment_reader::SegmentReader;
use sframe_storage::segment_writer::BufferedSegmentWriter;
use sframe_storage::sframe_writer::{assemble_sframe_from_segments, generate_hash, segment_filename};
use sframe_types::error::{Result, SFrameError};
use sframe_types::flex_type::{FlexType, FlexTypeEnum};

use crate::sarray::SArray;
use crate::sframe::{AnonymousStore, SFrame};

use super::CHUNK_SIZE;

/// Permute all buckets in parallel and assemble the output SFrame.
///
/// For each bucket (scatter segment), reads the forward map column to build
/// the local permutation, then reads value columns in groups that fit in
/// the memory budget, permutes them, and writes to output segments.
pub(super) fn permute_buckets(
    scatter_vfs: &Arc<dyn VirtualFileSystem>,
    scatter_base_path: &str,
    scatter_result: &sframe_storage::scatter_writer::ScatterResult,
    output_vfs: &Arc<dyn VirtualFileSystem>,
    output_base_path: &str,
    column_names: &[String],
    column_types: &[FlexTypeEnum],
    _num_rows: u64,
    rows_per_bucket: u64,
    column_bytes: &[usize],
    budget_per_thread: usize,
) -> Result<SFrame> {
    let num_buckets = scatter_result.segment_files.len();
    let num_input_cols = column_types.len();
    // The scatter segments have num_input_cols + 1 columns (last is forward_map)
    let mut scatter_col_types = column_types.to_vec();
    scatter_col_types.push(FlexTypeEnum::Integer);

    let data_prefix = format!("m_{}", generate_hash(output_base_path));

    // Process each bucket in parallel
    let results: Vec<Option<Result<(String, Vec<u64>, u64)>>> = (0..num_buckets)
        .into_par_iter()
        .map(|bucket_idx| {
            // Check if this bucket has any rows
            let bucket_rows = scatter_result.all_segment_sizes[bucket_idx]
                .first()
                .copied()
                .unwrap_or(0);
            if bucket_rows == 0 {
                return None;
            }

            Some((|| -> Result<(String, Vec<u64>, u64)> {
                permute_one_bucket(
                    scatter_vfs,
                    scatter_base_path,
                    &scatter_result.segment_files[bucket_idx],
                    &scatter_col_types,
                    output_vfs,
                    output_base_path,
                    &data_prefix,
                    bucket_idx,
                    num_input_cols,
                    column_types,
                    rows_per_bucket,
                    bucket_rows,
                    column_bytes,
                    budget_per_thread,
                )
            })())
        })
        .collect();

    // Collect results, skipping empty buckets
    let mut segment_files = Vec::new();
    let mut all_segment_sizes = Vec::new();
    let mut total_rows = 0u64;

    for result in results.into_iter().flatten() {
        let (seg_file, sizes, rows) = result?;
        if rows > 0 {
            segment_files.push(seg_file);
            all_segment_sizes.push(sizes);
            total_rows += rows;
        }
    }

    if total_rows == 0 {
        // Return empty SFrame
        let empty_columns: Vec<SArray> = column_types
            .iter()
            .enumerate()
            .map(|(_, &dt)| {
                SArray::from_vec(Vec::new(), dt).unwrap()
            })
            .collect();
        return Ok(SFrame::new_with_columns(
            empty_columns,
            column_names.to_vec(),
        ));
    }

    // Assemble SFrame metadata from segments
    let col_name_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
    assemble_sframe_from_segments(
        &**output_vfs,
        output_base_path,
        &col_name_refs,
        column_types,
        &segment_files,
        &all_segment_sizes,
        total_rows,
        &HashMap::new(),
    )?;

    // Construct SFrame backed by the output cache directory
    let cache_fs = global_cache_fs();
    let store: Arc<dyn Send + Sync> = Arc::new(AnonymousStore {
        path: output_base_path.to_string(),
        cache_fs: cache_fs.clone(),
    });
    let plan = PlannerNode::sframe_source_cached(
        output_base_path,
        column_names.to_vec(),
        column_types.to_vec(),
        total_rows,
        store,
    );

    let columns: Vec<SArray> = column_types
        .iter()
        .enumerate()
        .map(|(i, &dtype)| SArray::from_plan(plan.clone(), dtype, Some(total_rows), i))
        .collect();

    Ok(SFrame::new_with_columns(columns, column_names.to_vec()))
}

/// Permute a single bucket: read scatter segment, apply permutation, write output segment.
fn permute_one_bucket(
    scatter_vfs: &Arc<dyn VirtualFileSystem>,
    scatter_base_path: &str,
    scatter_seg_file: &str,
    scatter_col_types: &[FlexTypeEnum],
    output_vfs: &Arc<dyn VirtualFileSystem>,
    output_base_path: &str,
    output_data_prefix: &str,
    bucket_idx: usize,
    num_input_cols: usize,
    output_col_types: &[FlexTypeEnum],
    rows_per_bucket: u64,
    _bucket_rows: u64,
    column_bytes: &[usize],
    budget_per_thread: usize,
) -> Result<(String, Vec<u64>, u64)> {
    let seg_path = format!("{scatter_base_path}/{scatter_seg_file}");
    let bucket_start = bucket_idx as u64 * rows_per_bucket;

    // Open scatter segment and read the forward map column (last column)
    let file = scatter_vfs.open_read(&seg_path)?;
    let file_size = file.size()?;
    let mut seg_reader =
        SegmentReader::open(Box::new(file), file_size, scatter_col_types.to_vec())?;

    let fmap_values = seg_reader.read_column(num_input_cols)?;
    let num_rows_in_bucket = fmap_values.len();

    if num_rows_in_bucket == 0 {
        let seg_file = segment_filename(output_data_prefix, bucket_idx);
        let seg_out_path = format!("{output_base_path}/{seg_file}");
        let out_file = output_vfs.open_write(&seg_out_path)?;
        let seg_writer = BufferedSegmentWriter::new(out_file, output_col_types);
        let sizes = seg_writer.finish()?;
        return Ok((seg_file, sizes, 0));
    }

    // Build permutation: permutation[i] = local_target for scatter row i
    // fmap_values[i] is the global target row index.
    // local_target = fmap_values[i] - bucket_start
    let mut permutation = vec![0usize; num_rows_in_bucket];
    for (i, fmap_val) in fmap_values.iter().enumerate() {
        let global_target = match fmap_val {
            FlexType::Integer(v) => *v as u64,
            _ => {
                return Err(SFrameError::Type(
                    "forward_map must contain Integer values".into(),
                ))
            }
        };
        let local_target = (global_target - bucket_start) as usize;
        permutation[i] = local_target;
    }

    // Create output segment writer
    let seg_file = segment_filename(output_data_prefix, bucket_idx);
    let seg_out_path = format!("{output_base_path}/{seg_file}");
    let out_file = output_vfs.open_write(&seg_out_path)?;
    let mut seg_writer = BufferedSegmentWriter::new(out_file, output_col_types);

    // Process columns in groups that fit in memory budget
    let mut col_start = 0;
    while col_start < num_input_cols {
        // Determine how many columns fit in budget
        let mut col_end = col_start + 1; // At least one column per group
        let mut memory_estimate =
            column_bytes[col_start] * num_rows_in_bucket;

        while col_end < num_input_cols {
            let next_mem = column_bytes[col_end] * num_rows_in_bucket;
            if memory_estimate + next_mem > budget_per_thread {
                break;
            }
            memory_estimate += next_mem;
            col_end += 1;
        }

        // Re-open the scatter segment for reading value columns
        let file = scatter_vfs.open_read(&seg_path)?;
        let file_size = file.size()?;
        let mut col_reader =
            SegmentReader::open(Box::new(file), file_size, scatter_col_types.to_vec())?;

        // Read and permute each column in this group
        for col_idx in col_start..col_end {
            let values = col_reader.read_column(col_idx)?;

            // Apply permutation
            let mut permuted = vec![FlexType::Undefined; num_rows_in_bucket];
            for (src_idx, &dst_idx) in permutation.iter().enumerate() {
                permuted[dst_idx] = values[src_idx].clone();
            }

            // Write in chunks
            for chunk in permuted.chunks(CHUNK_SIZE) {
                seg_writer.write_column_block(col_idx, chunk, output_col_types[col_idx])?;
            }
        }

        col_start = col_end;
    }

    let sizes = seg_writer.finish()?;
    Ok((seg_file, sizes, num_rows_in_bucket as u64))
}
